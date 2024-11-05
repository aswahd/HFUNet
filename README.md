# A Hybrid of Foundation Model and U-Net for Improved Image Segmentation and Out-of-Distribution Generalization

## Installation
To set up the environment, follow these steps:

1. Create a virtual environment:
  ```bash
  python3 -m venv venv
  ```

2. Activate the virtual environment:
  ```bash
  source venv/bin/activate
  ```

3. Install the required dependencies:
  ```bash
  pip3 install -r requirements.txt
  ```


## Training Configuration
In the `config` file, you can specify the model, dataset, and training parameters. The following is an example of the configuration file for training HFUNet on the ACDC dataset.

```yaml
checkpoint: "weights/mobile_sam.pt"
model: "hfunet_tiny"

# You can choose a bigger model by changing the model name and the checkpoint path
# checkpoint: "weights/sam2_hiera_tiny.pt"
# model: "hfunet_hiera_tiny"

dataset:
  name: acdc
  root: ./datasets/ACDCPreprocessed
  image_size: 1024
  split: 0.8  # training split
  seed: 42
  batch_size: 4
  num_workers: 4
  num_classes: 3

max_epochs: 800
save_path: checkpoints/ACDC
```

# Training
To start training, run the following command:

```bash
python train.py --config config/acdc_hfunet_tiny.yaml
```

# Example: Tiny ViT HFUNet for ACDC dataset
1. Download the preprocessed data from [ACDC dataset](https://drive.google.com/drive/folders/14WIOWTF1WWwMaHV7UVo5rjWujpUxGetJ?usp=sharing).
2. Extract the data to `./datasets/ACDCPreprocessed`.
3. Place the configuration file in the `config` folder.
4. Run the training script:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config config/acdc_hfunet_tiny.yaml
```




## Evaluation

During testing, images are evaluated at their original size rather than the size used during model training. To facilitate easy batching, set the batch size to 1.

Below is an example configuration for evaluating a test dataset:

```yaml
# Choose model
checkpoint: "weights/mobile_sam.pt"
model: "hfunet_tiny"

test_config:
  name: acdc_test  # Name of the test dataset
  root: /path/to/ACDCPreprocessed  # Path to the test dataset
  image_size: 256  # Size of the image that the model was trained on
  num_classes: 3  # Number of classes in the dataset
  checkpoint_path: path/to/checkpoint  
```

To evaluate the model, run the following command:

```bash
python test.py --config config/test_config.yaml
```

## Baselines
We compared HFUNet with the following baseline models:

- [UNet](https://arxiv.org/abs/1505.04597)
- [UNet++](https://arxiv.org/pdf/1807.10165.pdf)
- [DeeplabV3+](https://arxiv.org/abs/1802.02611)
- [MAnet](https://ieeexplore.ieee.org/abstract/document/9201310)
- [PAN](https://arxiv.org/abs/1805.10180)

All baseline models were trained using the same preprocessing, data augmentation, and hyperparameters with the [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) framework. You can easily train any model supported by this framework by specifying its name and other parameters in the configuration file, then running the training script:

Example configuration for training the baseline model:

```yaml
# Dataset configs
dataset:
  name: ACDC
  root: ./datasets/ACDCPreprocessed
  image_size: 256 
  split: 0.2  # training split
  seed: 42
  batch_size: 4
  num_workers: 4
  num_classes: 3 

max_epochs: 200
save_path: checkpoints/ACDC

# segmentation_models.pytorch specific configs
model: DeepLabV3Plus
encoder_name: "resnet18"
encoder_weights: "imagenet"
classes: 3 
in_channels: 3
```


```bash
python train.py --config config/baseline_config.yaml
```
