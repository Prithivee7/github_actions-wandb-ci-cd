import wandb
print(f"The version of wandb is {wandb.__version__}")
print("This is to check if actions are getting triggered")

print("Checking the version for testing")
assert wandb.__version__ == "0.14.2", f"Expected version 0.14.2 but got {wandb.__version__}"
