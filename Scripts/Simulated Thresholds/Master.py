import os
import time
begin_time = time.time()
def print_folder_structure(root_path, indent=""):
    for root, dirs, files in os.walk(root_path):
        level = root.replace(root_path, "").count(os.sep)
        indent_level = "    " * level
        print(f"{indent_level}{os.path.basename(root)}/")
        sub_indent = "    " * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

# Replace with your folder path
folder_path = "Scripts"
print_folder_structure(folder_path)
#!/usr/bin/env python3
"""
Run all .py files inside 'Simulated Thresholds' except those in the
'Simulate_Thresholds' sub-directory.

Usage
-----
$ python run_all_st_scripts.py
"""

import subprocess
from pathlib import Path
import sys

def main():
    # --- Adjust these two paths if your layout differs -----------------------
    repo_root = Path(__file__).resolve().parent          # folder containing this runner
    target_dir = repo_root                               # current directory is Simulated Thresholds
    # -------------------------------------------------------------------------

    python_exe = sys.executable  # uses the current interpreter; override if needed

    for script in sorted(target_dir.rglob("*.py")):
        # Skip Master.py itself and any file that sits somewhere under Simulate_Thresholds/
        if script.name == "Master.py" or "Simulate_Thresholds" in script.parts:
            continue

        print(f"\n=== Running: {script.relative_to(repo_root)} ===")
        completed = subprocess.run(
            [python_exe, str(script)],
            capture_output=True,
            text=True
        )

        # Show standard output
        if completed.stdout:
            print("--- stdout ---")
            print(completed.stdout.rstrip())

        # Show errors, if any
        if completed.stderr:
            print("--- stderr ---")
            print(completed.stderr.rstrip())

        print(f"=== Finished {script.name} (return code {completed.returncode}) ===")

    print(f"\nAll requested scripts have been executed.")

if __name__ == "__main__":
    main()
end_time = time.time()
print(f"Time taken: {end_time - begin_time} seconds")