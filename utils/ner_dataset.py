from torch.utils.data import Dataset

class TranDataset(Dataset):
    """Dataset Class for word-level model

    args:
        data_tensor (ins_num, seq_length): words
        label_tensor (ins_num, seq_length): labels
        mask_tensor (ins_num, seq_length): padding masks
    """
    def __init__(self, data_tensor, label_tensor, action_tensor):
        #assert data_tensor.size(0) == label_tensor.size(0)
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.action_tensor = action_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index], self.action_tensor[index]

    def __len__(self):
        return len(self.data_tensor)
        #return self.data_tensor.size(0)
