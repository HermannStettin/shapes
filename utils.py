import os
import torch
import argparse

def _parse_args(params_config, args):
    parser = argparse.ArgumentParser()
    for params_category in params_config:  # e.g., 'model_params'
        for param_flag, param_config in params_config[params_category].items():
            # e.g., param_flag = '--block-sz'
            parser.add_argument(param_flag, **param_config)
    return parser.parse_args(args)

def get_params(params_config, args=None):
    namespace = _parse_args(params_config, args)
    return {
        params_category: {
            param_config["dest"]: namespace.__getattribute__(param_config["dest"])
            for param_config in params_config[params_category].values()
        }
        for params_category in params_config
    }

def set_up_env(env_params):
    assert torch.cuda.is_available()
    env_params["device"] = torch.device("cuda")

def get_optimizer(model, optim, lr):
    if optim == "sgd":
        return torch.optim.SGD(model.params(), lr = lr, momentum = 0.9)
    elif optim == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = 1e-4)
    else:
        raise RuntimeError("wrong type of optimizer - must be 'sgd' or 'adamw'")

def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    if checkpoint_path:
        checkpoint_state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint_state, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, resume):
    if resume:
        if os.path.exists(checkpoint_path):
            checkpoint_state = torch.load(checkpoint_path, weights_only=True)
            iter_init = checkpoint_state["epoch"] + 1
            model.load_state_dict(checkpoint_state["model"])
            optimizer.load_state_dict(checkpoint_state["optim"])
            return iter_init
    return 0