import os
import time
import torch
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch import Tensor, nn
import matplotlib.pyplot as plt
from datetime import datetime
from torchinfo import summary  # 需要安装 torchinfo 包
from torch.utils.data import random_split, DataLoader, Dataset
from scipy.stats import spearmanr
import warnings

import torch.optim

warnings.filterwarnings("ignore")

import grelu.lightning
import grelu.data
import grelu.model.models
from grelu.sequence.format import (
    indices_to_one_hot,
    strings_to_indices,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneExpressionDataset_self(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        # One-hot encode
        seq = self.sequences[idx]
        seq_strings = strings_to_indices(seq)
        seq_one_hot = indices_to_one_hot(seq_strings)
        
        label = self.labels[idx]

        return seq_one_hot, label

class PoissonLoss(nn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, predictions, targets):
        # 计算泊松损失
        loss = predictions - targets * torch.log(predictions + 1e-8)
        return loss.mean()  # 返回平均损失


parser = argparse.ArgumentParser(description="Run the model")
parser.add_argument('--expression_type', type=str, required=True,  help="translation of power or standard")
args = parser.parse_args()

expression_type = args.expression_type
print(f"--> expression_type: {expression_type}")

model = grelu.model.models.DNA_Conv1D_Model(expression_type).to(device)

if expression_type == "power": 
    mmc2_gene_expression_file = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_expression_power_transformed.npy"
elif expression_type == "standard":
    mmc2_gene_expression_file = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_expression_standard_transformed.npy"
elif expression_type == "minmax":
    mmc2_gene_expression_file = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_expression_minmax_transformed.npy"
elif expression_type == "identity":
    mmc2_gene_expression_file = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_expression_identity_transformed.npy"
elif expression_type == "log":
    mmc2_gene_expression_file = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_expression_log_transformed.npy"

mmc2_gene_sequence_file   = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_sequence.fasta"
mmc2_gene_location_file   = "/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/data/mmc2_78_gene_location.csv"

# select sequence with length less than 100000 
threshold = 100000 
df_mmc2_gene_location = pd.read_csv(mmc2_gene_location_file)
cell_type_list = df_mmc2_gene_location.iloc[:, 0].values
df_mmc2_gene_location['len'] = pd.to_numeric(df_mmc2_gene_location['len'], errors='coerce')  # 如果有非数值的值，设置为NaN
df_mmc2_gene_location = df_mmc2_gene_location[df_mmc2_gene_location['len'] < threshold]
selected_gene = df_mmc2_gene_location["gene_id"].to_list()
 
# load gene expression 
mmc2_gene_expression = np.load(mmc2_gene_expression_file, allow_pickle=True)
mmc2_gene_expression_dict = dict(zip(mmc2_gene_expression.item()["genes"], mmc2_gene_expression.item()["expression"]))

# load gene sequence 
mmc2_gene_sequence_dict = {}
for record in SeqIO.parse(mmc2_gene_sequence_file, "fasta"):
    gene_id = record.id  
    gene_sequence = str(record.seq) 
    mmc2_gene_sequence_dict[gene_id] = gene_sequence

dataset_dict = {
                'gene': [],
                'sequence': [],
                'expression': []
            }
dataset_df = pd.DataFrame(dataset_dict)

for gene in selected_gene:
    sequence = mmc2_gene_sequence_dict[gene]
    expression = mmc2_gene_expression_dict[gene]

    dataset_dict['gene'].append(gene)
    dataset_dict['sequence'].append(sequence)
    dataset_dict['expression'].append(expression)

# tarin_step
total_dataset = GeneExpressionDataset_self(dataset_dict['sequence'], dataset_dict['expression'])
train_size = int(0.85 * len(total_dataset))
test_size = len(total_dataset)-train_size

print(f"train_size: {train_size}; test_size: {test_size}")

train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

if expression_type == "minmax":
    criterion = torch.nn.MSELoss()  
else:
    criterion = PoissonLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

def calculate_spearman(real_activate, fake_activate):
    spearman_coefficients = []
    real_tensor = torch.stack(real_activate)  # shape: [num_samples, 78]
    fake_tensor = torch.stack(fake_activate)

    for col in range(real_tensor.shape[1]):
        real_col = real_tensor[:, col].numpy()  
        fake_col = fake_tensor[:, col].numpy()  

        coefficient, _ = spearmanr(real_col, fake_col)
        spearman_coefficients.append(coefficient)
   
    return spearman_coefficients


def Measure_Metric(model, total_loader):
    model.eval()  
    real_activate = []
    fake_activate = []
    with torch.no_grad():  
        for inputs, targets in total_loader:
            
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            outputs = model(inputs)
            real_activate.extend(targets)
            fake_activate.extend(outputs.to('cpu'))

    spearman_coefficients = calculate_spearman(real_activate, fake_activate)

    return spearman_coefficients, real_activate, fake_activate


def plt_figure(save_figure_path, cell_type, real_col, fake_col, spearmanr):

    os.makedirs(save_figure_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_col, fake_col, color="blue", alpha=0.5)

    plt.title(f"{cell_type}_{spearmanr:.5f}", fontsize=16)
    plt.xlabel("real", fontsize=14)
    plt.ylabel("fake", fontsize=14)
    plt.xlim(0, 200)  
    plt.ylim(0, 200)  

    plt.savefig(f'{save_figure_path}/{cell_type}_{spearmanr:.5f}.png')
    plt.close()


def Tra_Val_plt_figure(save_figure_path, cell_type, train_spear, tra_real_col, tra_fake_col, test_spear, test_real_col, test_fake_col):

    os.makedirs(save_figure_path, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))  # 1 行 2 列的子图

    axs[0].scatter(tra_real_col, tra_fake_col, color="blue", alpha=0.5)
    axs[0].set_title(f'Train_{cell_type}_{train_spear:.5f}')
    axs[0].set_xlabel('Real')
    axs[0].set_ylabel('Predicted')
    axs[0].set_xlim(0, 200)  
    axs[0].set_ylim(0, 200) 

    axs[1].scatter(test_real_col, test_fake_col)
    axs[1].set_title(f'Test_{cell_type}_{test_spear:.5f}')
    axs[1].set_xlabel('Real')
    axs[1].set_ylabel('Predicted')
    axs[1].set_xlim(0, 200)  
    axs[1].set_ylim(0, 200) 

    plt.tight_layout()
    plt.savefig(f'{save_figure_path}/{cell_type}_Tra:{train_spear:.2f}_Val:{test_spear:.2f}.png')
    plt.close()


def Visualization_Result(folder_path, train_spearman, train_real_activate, train_fake_activate, test_spearman, test_real_activate, test_fake_activate):
    for index in range(len(cell_type_list)):

        cell_type = cell_type_list[index]

        train_spear = train_spearman[index]
        train_real_activate_tensor = torch.stack(train_real_activate)  
        train_fake_activate_tensor = torch.stack(train_fake_activate)  
        tra_real_col = train_real_activate_tensor[:, index].numpy()  
        tra_fake_col = train_fake_activate_tensor[:, index].numpy() 

        test_spear  = test_spearman[index]
        test_real_activate_tensor = torch.stack(test_real_activate)  
        test_fake_activate_tensor = torch.stack(test_fake_activate)  
        test_real_col = test_real_activate_tensor[:, index].numpy()  
        test_fake_col = test_fake_activate_tensor[:, index].numpy() 

        save_figure_path = os.path.join(folder_path, "figure")
        Tra_Val_plt_figure(save_figure_path, cell_type, train_spear, tra_real_col, tra_fake_col, test_spear, test_real_col, test_fake_col)
        
        
def validate_model(model, val_loader, criterion):
    model.eval()  
    val_loss = 0.0
    real_activate = []
    fake_activate = []
    with torch.no_grad():  
        for inputs, targets in val_loader:
            
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            targets_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets_tensor)
            val_loss += loss.item()

            real_activate.extend(targets)
            fake_activate.extend(outputs.to('cpu'))
            
    avg_val_loss = val_loss / len(val_loader)
    spearman_coefficients = calculate_spearman(real_activate, fake_activate)

    return avg_val_loss, spearman_coefficients

def Construct_dir():
    year = datetime.now().year
    month = datetime.now().month 
    day = datetime.now().day
    hour = datetime.now().hour
    minute = datetime.now().minute

    folder_path = f"/geniusland/home/liuxianliang1/code/z_test_demo/Heidelberg/code/model/CNN_Model/{year}_{month}_{day}_{hour}_{minute}_{expression_type}"
    os.makedirs(folder_path, exist_ok=True)

    return folder_path


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()
    best_spearman = 0.0
    start_time = time.time()
    folder_path = Construct_dir()
    log_filename = os.path.join(folder_path, "log.txt")
    log_file = open(log_filename, 'w')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            targets = torch.tensor(targets, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, spearman_coefficients = validate_model(model, val_loader, criterion)
        end_time = time.time()

        top_5 = sorted(spearman_coefficients, reverse=True)[:5]
        bottom_5 = sorted(spearman_coefficients)[:5]
        content_1 = f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Trainer_Loss: {loss.item():.7f}, Val_Loss: {val_loss:.7f}; time: {end_time-start_time}'
        content_2 = f"--> Top 5 Spearman coefficients: {top_5}"
        content_3 = f"--> Low 5 Spearman coefficients: {bottom_5}"
        log_file.write(content_1+"\n"+content_2+"\n"+content_3+"\n")
        log_file.flush()
        start_time = time.time()

        current_max_spearman = max(spearman_coefficients)
        
        if current_max_spearman > best_spearman:
            current_max_spearman = best_spearman
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,  
                'loss': loss,   
            }
            torch.save(checkpoint, f"{folder_path}/checkpoint.ckpt")

            train_spearman, train_real_activate, train_fake_activate = Measure_Metric(model, train_loader)
            test_spearman, test_real_activate, test_fake_activate = Measure_Metric(model, val_loader)
            Visualization_Result(folder_path, train_spearman, train_real_activate, train_fake_activate, test_spearman, test_real_activate, test_fake_activate)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20)




