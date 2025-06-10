
PARAMS_CONFIG = {
    "env_params": {

    },
    "data_params": {
        "--data": {
            "type": str,
            "required": True,
            "help": "Path to the dataset directory (must contain train.txt, valid.txt, test.txt).",
            "dest": "data_path",
        },
    },
    "model_params": {
        "--weights": {
            "type": str,
            "default": None,
            "help": "Pretrained weights to use. Possible options: None, IMAGENET1K_V1.",
            "dest": "weights",
        },
    },
    "trainer_params": {
        "--epochs": {
            "type": int,
            "default": 10,
            "help": "Number of training epochs.",
            "dest": "epochs",
        },
        "--batch-size": {
            "type": int,
            "default": 32,
            "help": "Batch size.",
            "dest": "batch_size",
        },
        "--checkpoint": {
            "type": str,
            "required": True,
            "help": "Path to save/load model checkpoint in the form 'path/model.pt'.",
            "dest": "checkpoint_path",
        },
        "--resume": {
            "action": "store_true",
            "default": False,
            "help": "Resume training from the checkpoint if it exists.",
            "dest": "resume",
        },
        "--full-eval-mode": {
            "action": "store_true",
            "default": False,
            "help": "Run in full evaluation mode.",
            "dest": "full_eval_mode",
        },
    },
    "optim_params": {
        "--lr": {
            "type": float,
            "default": 0.001,
            "help": "Learning rate.",
            "dest": "lr"
        },
        "--optim": {
            "type": str,
            "default": "adamw",
            "help": "Optimizer type: 'sgd' or 'adamw'.",
            "dest": "optim",
        },
    },
    "wandb_params": {
        "--use-wandb": {
            "action": "store_true",
            "default": False,
            "help": "Enable logging to Weights & Biases.",
            "dest": "wandb_flag",
        },
        "--wandb-key": {
             "type": str,
             "default": None,
             "help": "WandB API key.",
             "dest": "wandb_key",
        },
        "--project-name": {
            "type": str,
            "default": None,
            "help": "WandB project name.",
            "dest": "wandb_project",
        },
        "--run-name": {
            "type": str,
            "default": None,
            "help": "WandB run name.",
            "dest": "run_name",
        },
        "--run-id": {
            "type": str,
            "default": None,
            "help": "WandB run id for resuming existing run",
            "dest": "run_id",
        },
    }
}