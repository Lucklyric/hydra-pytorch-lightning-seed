# hydra-pytorch-lightning-seed
A Project Template Seed using Hydra, PyTorch and PyTorch-Lightning

## Main Devstack
- python 3.6, [conda/anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/)
- [PyTorch-lightning](https://www.pytorchlightning.ai/)
- [hydra](https://hydra.cc/)

## This project seed includes a dummy DnCNN model for MNIST filtering
Structure
```bash
.
├── config      # hydra configures
├── data        # data scripts
├── __init__.py
├── model       # model scripts
├── test.py     # Test entrypoint
├── train.py    # Train entrypoint
```
### Setup python environment
```
# install requirements
conda env create -n mlseed -f ./envs/conda_env.yml
conda activate mlseed
cd project
```

### Training

Train with predefined configs - model: **dncnn_small** and data: **train_dummy_mnist**

```bash
python train.py \
    model=dncnn_small \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_small_mnist/' \
```

Train with predefined configs - model: **dncnn_small** and data: **train_dummy_mnist**

```bash
python train.py \
    model=dncnn_large \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_large_mnist/' \
```
Train with CLI custom configs by overriding existing configs. [hydra override grammar](https://hydra.cc/docs/advanced/override_grammar/basic)  
```bash
# override existing config in yaml file
python train.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_mid_mnist/' \

# or append new config for pl.trainer with +
python train.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    +pl_trainer.benchmark=True \
    data=train_dummy_mnist \
    processing_dir='./processing/train/dncnn_mid_mnist/' \
```

### Testing
Test with pretrained checkpoint file after training 

```bash
# Recommend to use absolute path for checkpoint_path then you do not need extract $PWD
python test.py \
    model=dncnn_small \
    model.num_features=32 \
    model.num_layers=5 \
    data=test_dummy_mnist \
    +pl_trainer.deterministic=True \
    checkpoint_path=$PWD'/processing/train/dncnn_mid_mnist/lightning_logs/version_0/checkpoints/epoch\=19-step\=8450.ckpt' \
    processing_dir='./processing/test/dncnn_mid_mnist/' \
```
