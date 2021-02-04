import os
import subprocess
import time
import logging
import sys

garbage_list = ['.ipynb_checkpoints', 'generator.ipynb', 'generator.py']

current_dir = os.getcwd()

root_dir = os.path.dirname(current_dir)
notebook_dir = os.path.join(root_dir, 'notebook')
output_dir = os.path.join(root_dir, 'output')
time_w8 = 10


def shell_command(filename, output_dir = output_dir):
    format_file = "pdf"
    extension = f"--to {format_file} "
    option = "--ExecutePreprocessor.timeout=-1 "
    output = f"""--output-dir="{output_dir}" """
    hide = "--no-input "
    execute = "--execute "
    file = f'"{filename}"'
    
    command = f"""jupyter nbconvert {extension}{option}{output}{hide}{execute}{file} """.strip()
    return command

retry = True
while retry:
    try:
        subprocess.run(shell_command("Nuove Case.ipynb"), cwd = notebook_dir)
        retry = False
    except:
        time.sleep(time_w8)