## Installation
You need to have a working version of PyTorch and Pytorch3D installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:

```bash
conda create -n smirk python=3.9
pip install -r requirements.txt
# install pytorch3d now
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
```

Then, in order to download the required models, run:

```bash
bash quick_install.sh
```
*The above installation includes downloading the [FLAME](https://flame.is.tue.mpg.de/) model. This requires registration. If you do not have an account you can register at [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)*

This command will also download the SMIRK pretrained model which can also be found on [Google Drive](https://drive.google.com/file/d/1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE/view?usp=sharing).


## Training

Download the MEAD dataset from [here](https://wywu.github.io/projects/MEAD/MEAD.html).

After downloading the datasets we need to extract the landmarks using mediapipe and FAN. We provide the scripts for preprocessing in `datasets/preprocess_scripts`. Example usage:

```bash
python datasets/preprocess_scripts/apply_mediapipe_to_dataset.py --input_dir PATH_TO_MEAD --output_dir PATH_TO_MEAD/mediapipe_landmarks
```

and for FAN landmarks we use the implementation in [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).

```bash
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```

```bash
python datasets/preprocess_scripts/apply_fan_to_dataset.py --input_dir PATH_TO_MEAD --output_dir PATH_TO_MEAD/fan_landmarks
```

```bash
# With temporal loss
python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=true train.log_path=logs/temporal
# Without temporal loss
python train.py configs/config_train.yaml resume=pretrained_models/SMIRK_em1.pt train.loss_weights.emotion_loss=1.0 train.use_temporal_loss=false train.log_path=logs/non_temporal
```