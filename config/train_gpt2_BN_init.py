# config for training GPT-2 (124M) with 4 A40s 
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_BN_init.py

wandb_log = True
wandb_project = 'nanoGPT_BN_testing'
wandb_run_name='gpt2-124M_BN_init'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 4

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

bn_init = True
bn_init_n_batches = 1000
