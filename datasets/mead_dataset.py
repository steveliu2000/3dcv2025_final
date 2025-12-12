import os
import sys
import pickle
from datasets.data_utils import landmarks_interpolate
from datasets.base_dataset import BaseDataset
import numpy as np
import cv2
import torch

class MEADDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD'

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]
        mediapipe_landmarks_path = sample[2]

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))
            
        if not os.path.exists(mediapipe_landmarks_path):
            print('Mediapipe landmarks not found for %s'%(sample))
            return None

        mediapipe_landmarks = np.load(mediapipe_landmarks_path)

        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # select randomly one file from this subject
        if num_frames == 0:
            print('Video %s has no frames'%(sample))
            return None


        # pick random frame
        frame_idx = np.random.randint(0, num_frames)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image = video.read()
        if not ret:
            raise Exception('Video %s has no frames'%(sample))
        
        landmarks_fan = preprocessed_landmarks[frame_idx]
        landmarks_mediapipe = mediapipe_landmarks[frame_idx]
        if landmarks_fan.ndim == 3:
            landmarks_fan = landmarks_fan[0]

        data_dict = self.prepare_data(image=image, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)

        return data_dict
    

class MEADVideoDataset(BaseDataset):
    def __init__(self, data_list, config, test=False):
        super().__init__(data_list, config, test)
        self.name = 'MEAD'
        if hasattr(config.train, 'batch_size'):
            self.seq_len = config.train.batch_size
        self.seq_len = 32

    def __len__(self):
        return len(self.data_list)

    def __getitem_aux__(self, index):
        sample = self.data_list[index]

        landmarks_filename = sample[1]
        video_path = sample[0]
        mediapipe_landmarks_path = sample[2]

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                raise Exception('Video %s has no landmarks'%(sample))
            
        if not os.path.exists(mediapipe_landmarks_path):
            print('Mediapipe landmarks not found for %s'%(sample))
            return None

        mediapipe_landmarks = np.load(mediapipe_landmarks_path)

        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # select randomly one file from this subject
        if num_frames == 0:
            print('Video %s has no frames'%(sample))
            return None


        # pick random frame
        seq_len = self.seq_len
        if num_frames < seq_len:
            seq_len = num_frames

        start_idx = np.random.randint(0, num_frames - seq_len + 1)

        data_dicts = []

        video.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for i in range(seq_len):
            ret, frame = video.read()
            if not ret:
                raise Exception(f"Video {sample} reading failed at frame {start_idx + i}")
        
            landmarks_fan = preprocessed_landmarks[start_idx + i]
            landmarks_mediapipe = mediapipe_landmarks[start_idx + i]
            if landmarks_fan.ndim == 3:
                landmarks_fan = landmarks_fan[0]

            data_dict = self.prepare_data(image=frame, landmarks_fan=landmarks_fan, landmarks_mediapipe=landmarks_mediapipe)
            data_dicts.append(data_dict)

        return data_dicts
    
    def __getitem__(self, index):
        while True:
            try:
                data_dicts = self.__getitem_aux__(index)   # list of seq_len dict

                # 檢查 data_dicts 是否為 list
                if data_dicts is None or not isinstance(data_dicts, (list, tuple)):
                    print("Error: data_dicts is None or not list. Trying again...")
                    index = np.random.randint(0, len(self.data_list))
                    continue

                # 所有 seq frames 都要通過檢查
                all_valid = True
                for data_dict in data_dicts:
                    if data_dict is None:
                        all_valid = False
                        break

                    landmarks = data_dict.get("landmarks_fan", None)
                    if landmarks is None or landmarks.shape[-2] != 68:
                        all_valid = False
                        break

                if all_valid:
                    tensor_dict = {}
                    keys = data_dicts[0].keys()
                    for key in keys:
                        tensor_dict[key] = torch.stack([d[key].clone().detach() if isinstance(d[key], torch.Tensor) else torch.tensor(d[key]) for d in data_dicts])

                    return tensor_dict

                print("Some seq frames invalid. Trying again...")
                index = np.random.randint(0, len(self.data_list))

            except Exception as e:
                print("Error in loading data. Trying again...", e)
                index = np.random.randint(0, len(self.data_list))
    

def get_datasets_MEAD(config=None):
    # Assuming you're currently in the directory where the files are located
    # files = [f for f in os.listdir(config.dataset.MEAD_fan_landmarks_path)]

    # this is the split used in the paper, randomly selected
    # train_subjects = ['M003', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025', 'M026', 'M027', 'M029', 'M030', 'M031', 'M032', 'M033', 'M034', 'M035', 'M037', 'M039', 'M040', 'M041', 'W009', 'W011', 'W014', 'W015', 'W016', 'W019', 'W021', 'W023', 'W024', 'W025', 'W026', 'W035', 'W036', 'W037', 'W038', 'W040']
    # val_subjects = ['M013', 'M023', 'M042', 'W018', 'W028']
    # test_subjects = ['M005', 'M022', 'M028', 'W029', 'W033']
    train_subjects = ['M013', 'M007', 'M009', 'M011', 'M012', 'M019', 'M024', 'M025']
    val_subjects = ['M003', 'M023']
    test_subjects = ['M005', 'M022']

    # assert each subject is in exactly one split
    assert len(set(train_subjects).intersection(val_subjects)) == 0
    assert len(set(train_subjects).intersection(test_subjects)) == 0
    assert len(set(val_subjects).intersection(test_subjects)) == 0

    train_list = []
    for part in train_subjects:
        root_dir = os.path.join(config.dataset.MEAD_path, part, 'video')
        landmarks_root = os.path.join(config.dataset.MEAD_fan_landmarks_path, part, 'video')
        mediapipe_root = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, part, 'video')
        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith(('.mp4', '.avi')):
                    relative_path = os.path.relpath(root, root_dir)
                    
                    landmarks_path = os.path.join(landmarks_root, relative_path, file_name.split(".")[0] + ".pkl")
                    folder_path = os.path.join(root_dir, relative_path, file_name)
                    mediapipe_landmarks_path = os.path.join(mediapipe_root, relative_path, file_name.split(".")[0] + ".npy")
                    train_list.append([folder_path, landmarks_path, mediapipe_landmarks_path])

    val_list = []
    for part in val_subjects:
        root_dir = os.path.join(config.dataset.MEAD_path, part, 'video')
        landmarks_root = os.path.join(config.dataset.MEAD_fan_landmarks_path, part, 'video')
        mediapipe_root = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, part, 'video')
        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith(('.mp4', '.avi')):
                    relative_path = os.path.relpath(root, root_dir)

                    landmarks_path = os.path.join(landmarks_root, relative_path, file_name.split(".")[0] + ".pkl")
                    folder_path = os.path.join(root_dir, relative_path, file_name)
                    mediapipe_landmarks_path = os.path.join(mediapipe_root, relative_path, file_name.split(".")[0] + ".npy")
                    val_list.append([folder_path, landmarks_path, mediapipe_landmarks_path])


    test_list = []
    for part in test_subjects:
        root_dir = os.path.join(config.dataset.MEAD_path, part, 'video')
        landmarks_root = os.path.join(config.dataset.MEAD_fan_landmarks_path, part, 'video')
        mediapipe_root = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, part, 'video')
        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if file_name.lower().endswith(('.mp4', '.avi')):
                    relative_path = os.path.relpath(root, root_dir)

                    landmarks_path = os.path.join(landmarks_root, relative_path, file_name.split(".")[0] + ".pkl")
                    folder_path = os.path.join(root_dir, relative_path, file_name)
                    mediapipe_landmarks_path = os.path.join(mediapipe_root, relative_path, file_name.split(".")[0] + ".npy")
                    test_list.append([folder_path, landmarks_path, mediapipe_landmarks_path])

    # for file in files:
    #     if file.split('_')[0] in train_subjects:
    #         landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".npy")
    #         folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
    #         mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
    #         train_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    # val_list = []
    # for file in files:
    #     if file.split('_')[0] in val_subjects:
    #         landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".npy")
    #         folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
    #         mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
    #         val_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    # test_list = []
    # for file in files:
    #     if file.split('_')[0] in test_subjects:
    #         landmarks_path = os.path.join(config.dataset.MEAD_fan_landmarks_path, file.split(".")[0] + ".npy")
    #         folder_path = os.path.join(config.dataset.MEAD_path,file.split(".")[0]+".mp4")
    #         mediapipe_landmarks_path = os.path.join(config.dataset.MEAD_mediapipe_landmarks_path, file.split(".")[0] + ".npy")
    #         test_list.append([folder_path, landmarks_path, mediapipe_landmarks_path, file.split('_')[0]])

    if hasattr(config.train, 'use_sequencial_data') and config.train.use_sequencial_data:
        return MEADVideoDataset(train_list, config), MEADVideoDataset(val_list, config, test=True), MEADVideoDataset(test_list, config, test=True) #, train_list, val_list, test_list
    else:
        return MEADDataset(train_list, config), MEADDataset(val_list, config, test=True), MEADDataset(test_list, config, test=True) #, train_list, val_list, test_list

