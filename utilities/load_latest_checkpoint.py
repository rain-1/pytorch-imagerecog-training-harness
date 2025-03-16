import os

def load_latest_checkpoint(model, checkpoint_dir="checkpoints"):
    # Load latest model from checkpoint folder if it exists
    checkpoints = os.listdir(checkpoint_dir)
    if checkpoints:
        latest_checkpoint = max([f"{checkpoint_dir}/{checkpoint}" for checkpoint in checkpoints], key=os.path.getctime)
        try:
            model.load(latest_checkpoint)
            print(f"Loaded checkpoint: {latest_checkpoint}")
        except EOFError:
            print(f"Failed to load checkpoint: {latest_checkpoint} (file might be corrupted or incomplete)")

