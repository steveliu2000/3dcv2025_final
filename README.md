## Installation
You need to have a working version of PyTorch and Pytorch3D installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:

```bash
conda create -n smirk python=3.9 -y
conda activate smirk
pip install "pip<23"
pip install -r requirements.txt
# install pytorch3d now
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
```

Then, in order to download the required models, run:

```bash
bash quick_install.sh
```
*The above installation includes downloading the [FLAME](https://flame.is.tue.mpg.de/) model. This requires registration. If you do not have an account you can register at [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)*

This command will also download the SMIRK pretrained model which can also be found on [Google Drive](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing) and [Google Drive](https://drive.google.com/drive/u/0/folders/1eqj7q9pXU5nXH4awhnRl_QwaWDnIrj3H).

## Demo 
We provide two demos. One that can be used to test the model on a single image,

```bash
python demo.py --input_path samples/test_images --out_path results/ --checkpoint pretrained_models/non_temporal.pt --crop
```

and one that can be used to test the model on a video,

```bash
python demo_video.py --input_path MEAD_samples/001.mp4 --out_path results/ --checkpoint pretrained_models/temporal.pt --crop --render_orig
```

## Training

### Dataset Preparation
Download the MEAD dataset from [here](https://wywu.github.io/projects/MEAD/MEAD.html).

After downloading the datasets we need to extract the landmarks using mediapipe and FAN. We provide the scripts for preprocessing in `datasets/preprocess_scripts`. Example usage:

```bash
python datasets/preprocess_scripts/apply_mediapipe_to_dataset.py --input_dir PATH_TO_MEAD --output_dir PATH_TO_MEAD/mediapipe_landmarks
# e.g. python datasets/preprocess_scripts/apply_mediapipe_to_dataset.py --input_dir MEAD_samples --output_dir MEAD_samples/mediapipe_landmarks
```

and for FAN landmarks we use the implementation in [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment).

```bash

git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

```bash
python datasets/preprocess_scripts/apply_fan_to_dataset.py --input_dir PATH_TO_MEAD --output_dir PATH_TO_MEAD/fan_landmarks
# e.g. python datasets/preprocess_scripts/apply_fan_to_dataset.py --input_dir MEAD_samples --output_dir MEAD_samples/fan_landmarks
```

### Training
Update the config files in configs with the correct paths to the datasets and their landmarks.

```bash
# With temporal loss
python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=true train.log_path=logs/temporal
# Without temporal loss
python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=false train.log_path=logs/non_temporal
```