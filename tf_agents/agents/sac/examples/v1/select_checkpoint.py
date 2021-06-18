import sys
import os

folder = sys.argv[1]
idx = sys.argv[2]

checkpoint_file = os.path.join(folder, 'train', 'checkpoint')
with open(checkpoint_file) as f:
    new_lines = []
    for line in f.readlines():
        if line.startswith('model_checkpoint_path:'):
            line = 'model_checkpoint_path: "ckpt-{}000"\n'.format(idx)
        new_lines.append(line)

with open(checkpoint_file, 'w') as f:
    for line in new_lines:
        f.write(line)

checkpoint_file = os.path.join(folder, 'train', 'policy', 'checkpoint')
with open(checkpoint_file) as f:
    new_lines = []
    for line in f.readlines():
        if line.startswith('model_checkpoint_path:'):
            line = 'model_checkpoint_path: "ckpt-{}000"\n'.format(idx)
        new_lines.append(line)

with open(checkpoint_file, 'w') as f:
    for line in new_lines:
        f.write(line)
