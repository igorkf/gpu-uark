import torch
import torch.nn as nn
import torch.nn.functional as F


class DNA_Net(nn.Module):
    def __init__(self, input_dim):
        super(DNA_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
    

def mean_mse(y_true, y_pred, envs):
    unique_envs = torch.unique(envs)
    mse_value = 0
    for env in unique_envs:
        mask = envs == env
        mse_value += torch.mean((y_true[mask] - y_pred[mask]) ** 2)
    return mse_value / len(unique_envs)


def mean_r(y_true, y_pred, envs):
    """
    Calculate the mean Pearson's Correlation Coefficient (PCC) using a cosine similarity trick.
    """
    unique_envs = torch.unique(envs)
    pcc_value = 0
    for env in unique_envs:
        mask = envs == env
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]
        pcc_value += F.cosine_similarity(
            y_true_masked - y_true_masked.mean(),
            y_pred_masked - y_pred_masked.mean(),
            dim=0,
        )
    return pcc_value / len(unique_envs)
