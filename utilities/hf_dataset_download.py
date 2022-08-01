from datasets import load_dataset
# for tokenization 
from titans.utils import barrier_context


def main():
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    args = {'dataset_name': 'wikitext', 'dataset_config_name': 'wikitext-2-raw-v1', 'validation_split_percentage': 5}
    
    # logger.info("Prepare dataset")
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        # load manual files
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
            
def hf_tokenize():
      # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    logger.info("Model config is created")

    if args.model_name_or_path == 'facebook/13b':
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    logger.info(f"{tokenizer.__class__.__name__} is created")

    # build model
    shard_strategy = TensorShardStrategy()
    with ZeroInitContext(target_device=get_current_device(), shard_strategy=shard_strategy, shard_param=True):
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path, config=config)

        # enable graident checkpointing
        model.gradient_checkpointing_enable()
    logger.info(f'{model.__class__.__name__} is created')

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]


    # LIL MINI GUY!
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with barrier_context(executor_rank=0, parallel_mode=ParallelMode.DATA):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx.")
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                           f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.")
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)
               ] for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with barrier_context(executor_rank=0, parallel_mode=ParallelMode.DATA):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = get_dataloader(train_dataset,
                                      shuffle=True,
                                      add_sampler=True,
                                      collate_fn=default_data_collator,
                                      batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 batch_size=args.per_device_eval_batch_size)
    logger.info("Dataloaders are created")


if __name__ == "__main__":
  main()
  