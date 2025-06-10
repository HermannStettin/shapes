import torch
import wandb
import tqdm
from config import PARAMS_CONFIG
from utils import (
    get_params,
    set_up_env,
    get_optimizer,
    load_checkpoint,
    save_checkpoint,
    )
from data import get_train_val_test_data
from models import ResNet
from trainer import (
    train,
    validate
)

def launch(
    env_params,
    data_params,
    model_params,
    trainer_params,
    optim_params,
    wandb_params,
):
    wandb_flag = wandb_params["wandb_flag"]
    if wandb_flag:
        wandb.login(key = wandb_params["wandb_key"])

        # In case of id != None we'll resume run logging in wandb
        wandb.init(
            project = wandb_params["wandb_project"],
            id = wandb_params["run_id"],
            resume = "allow"
        )
        # Specify your run name or use one created by wandb
        wandb.run.name = wandb_params["run_name"] if wandb_params["run_name"] else wandb.run.name
        wandb.config.update(model_params)
        wandb.config.update(data_params)
        wandb.config.update(optim_params)

    set_up_env(env_params)
    device = env_params["device"]

    print("data_params:\t", data_params)
    print("model_params:\t", model_params)
    print("optim_params:\t", optim_params)
    print("trainer_params:\t", trainer_params)

    train_data, val_data, test_data = get_train_val_test_data(
        data_params = data_params,
        batch_size = trainer_params["batch_size"],
        device = device,
    )

    model = ResNet(
        num_classes = 8,
        # "IMAGENET1K_V1"
        weights = model_params["weights"]
    ).to(device)

    print(model)

    optimizer = get_optimizer(
        model = model,
        optim = optim_params["optim"],
        lr = optim_params["lr"],
    )
    
    resume = trainer_params["resume"]
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        resume,
    )

    if trainer_params["full_eval_mode"]:
        loss_val, acc_val, f1_val = validate(
            model,
            val_data,
            device,
        )
        loss_test, acc_test, f1_test = validate(
            model,
            test_data,
            device,
        )

        print(f"Val: Average loss: {loss_val:.6f}, Accuracy: {acc_val:.2f}%, F1-Score {f1_val:.2f}\n")
        print(f"Test: Average loss: {loss_test:.6f}, Accuracy: {acc_test:.2f}%, F1-Score {f1_test:.2f}\n")
        
        return

    best_val_loss = None
    epochs = trainer_params["epochs"]
    for epoch in range(iter_init, epochs):
        print(f"=================== EPOCHS {epoch} ======================")
        train_epoch_loss, train_epoch_accuracy = train(model, optimizer, train_data, device)
        
        val_epoch_loss, val_epoch_accuracy, val_epoch_f1 = validate(model, val_data, device)

        if wandb_flag:
            wandb.log({
                "train_loss": train_epoch_loss,
                "train_acc": train_epoch_accuracy,
                "val_loss": val_epoch_loss,
                "val_acc": val_epoch_accuracy,
                "val_f1_score": val_epoch_f1
            })

        if best_val_loss is None or val_epoch_loss < best_val_loss:
            save_checkpoint(trainer_params["checkpoint_path"], model, optimizer, epoch)
            best_val_loss = val_epoch_loss

if __name__ == "__main__":
    launch(**get_params(params_config = PARAMS_CONFIG))