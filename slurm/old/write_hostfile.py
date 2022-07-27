import os

# nodelist = "gpub[073,078]"
nodelist = str(os.environ["SLURM_JOB_NODELIST"])
write_dir = os.path.join(str(os.environ["BASE_DIR"]), 'nodelist.txt')

# process slurm nodelist into one node per newline.
prefix = nodelist.split('[')[0]
raw_list = nodelist.split('[')[1].strip('[]')
if ',' in raw_list:
  nodes = list(raw_list.split(','))
elif '-' in raw_list:
  nodes = list(raw_list.split('-'))
  
final = [prefix + node for node in nodes]

print("CHECK FOR CORRECTNESS: (only works on 2 nodes) Node list: ", final)
print("official: ", nodelist)

# save hostfile for 
with open(write_dir, 'w') as f:
  for node in final:
    f.write(node + '\n')
