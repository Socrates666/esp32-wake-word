import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, features_list, labels_list):
        self.features = features_list
        self.labels = labels_list
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label

class AudioFrameDataset(AudioDataset):
    def __format__(self, format_spec):
        return super().__format__(format_spec)
    def __init__(self, features_list, labels_list):
        super().__init__(features_list, labels_list)
        frame_list = []
        for t in self.features:
            for frame in t.T:
                frame_list.append(frame)
        self.features = frame_list
        l_list = []
        for t in self.labels:
            for index, i in enumerate(range(63)):
                l_list.append(torch.tensor((1, index/63, (index+1)/63)))
        self.labels = l_list


