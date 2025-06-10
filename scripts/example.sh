mkdir results/

args="
--data /kaggle/input/geometric-shapes-mathematics/dataset \
--weights IMAGENET1K_V1 \
--lr 1e-3 \
--epochs 5 \
--checkpoint results/model.pt
"

echo "Training..."
python3 train.py $args --use-wandb \
    --wandb-key <your wandb API key> \
    --project-name shapes \
    --run-name resnet-lr=1e-3 \

echo "Evaluation..."
python3 train.py $args --full-eval-mode --resume