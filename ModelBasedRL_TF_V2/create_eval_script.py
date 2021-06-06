import argparse
import os

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', required=True)
  parser.add_argument('--task', required=True)
  parser.add_argument('--ids', nargs='+', required=True)
  args, remaining = parser.parse_known_args()

  commands = []
  for run_id in args.ids:
    run_path = os.path.join(args.logdir, args.task, run_id)
    seeds = os.listdir(run_path)
    if len(seeds) > 0:
      script_file_path = os.path.join(run_path, seeds[0], 'script.sh')
      with open(script_file_path) as f:
        commands.append(f.readlines()[1])
  eval_command = '\n'.join(commands).replace('dreamer.py', 'evaluation.py')
  script_path = os.path.join('scripts', 'eval_script.sh')
  with open(script_path, 'w') as f:
    f.write("#!/bin/bash")
    f.write("\n")
    f.write(eval_command)