import os

def load_latest_checkpoint(prefix, model, checkpoint_dir="checkpoints"):
    # Load latest model from checkpoint folder if it exists
    checkpoints = os.listdir(checkpoint_dir)
    if checkpoints:
        matching_checkpoints = [f"{checkpoint_dir}/{checkpoint}" for checkpoint in checkpoints if checkpoint.startswith(prefix)]
        if not matching_checkpoints:
            print(f"No checkpoints found with prefix '{prefix}' in directory '{checkpoint_dir}'.")
            return
        latest_checkpoint = max(matching_checkpoints, key=os.path.getctime)
        try:
            model.load(latest_checkpoint)
            print(f"Loaded checkpoint: {latest_checkpoint}")
        except EOFError:
            print(f"Failed to load checkpoint: {latest_checkpoint} (file might be corrupted or incomplete)")

