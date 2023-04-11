import wandb
print(f"The version of wandb is {wandb.__version__}")
print("This is to check if actions are getting triggered")

print("Checking the version")
assert wandb.__version__ == "2.1.01", f"Expected version 2.1.01 but got {wandb.__version__}"
