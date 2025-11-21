TO START PROJECT IN NEW ENV:

$pip install transformers pandas tqdm scikit-learn numpy wandb datasets hf_transfer

$wandb login

$python cuda_test.py #test script on runpod

$python router_final.py OR ./start_training.sh

Run with DDP
$torchrun --standalone --nproc_per_node=2 router_acc_only.py