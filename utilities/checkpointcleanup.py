import os
import re
from pathlib import Path

def delete_old_checkpoints(checkpoints_dir):
    # Get all checkpoint files in the directory
    checkpoint_files = list(Path(checkpoints_dir).glob("*.pth"))
    
    # Dictionary to store the newest file for each architecture/run type
    newest_files = {}

    # Regex to extract the architecture/run type and version number
    pattern = re.compile(r"(.+)_([0-9]+)\.pth")

    for file in checkpoint_files:
        match = pattern.match(file.name)
        if match:
            run_type, version = match.groups()
            version = int(version)
            # Update the newest file for this run type
            if run_type not in newest_files or newest_files[run_type][1] < version:
                newest_files[run_type] = (file, version)

    # Delete all files except the newest ones
    for file in checkpoint_files:
        match = pattern.match(file.name)
        if match:
            run_type, _ = match.groups()
            if newest_files[run_type][0] != file:
                print(f"Deleting old checkpoint: {file}")
                file.unlink()

# Example usage
if __name__ == "__main__":
    delete_old_checkpoints("checkpoints/")
