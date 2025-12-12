import torch
import os


def load_dataloaders(config):
    from datasets.mead_dataset import get_datasets_MEAD
    # ----------------------- initialize datasets ----------------------- #
    train_dataset_MEAD, val_dataset_MEAD, test_dataset_MEAD = get_datasets_MEAD(config)
    
    def collate_fn(batch):
        # filter none
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)
        
    if config.train.use_sequencial_data:
        train_loader = torch.utils.data.DataLoader(train_dataset_MEAD,
                                                batch_size=1,
                                                num_workers=config.train.num_workers)
        
        val_loader = torch.utils.data.DataLoader(val_dataset_MEAD,
                                                batch_size=1,
                                                num_workers=config.train.num_workers,
                                                shuffle=False, drop_last=True)
    else:    
        train_loader = torch.utils.data.DataLoader(train_dataset_MEAD,
                                                batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers,
                                                collate_fn=collate_fn)
        
        val_loader = torch.utils.data.DataLoader(val_dataset_MEAD,
                                                batch_size=config.train.batch_size,
                                                num_workers=config.train.num_workers,
                                                shuffle=False, drop_last=True,
                                                collate_fn=collate_fn)

    return train_loader, val_loader





def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks

def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks

