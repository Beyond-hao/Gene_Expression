import os
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from torchinfo import summary  # 需要安装 torchinfo 包
from torch.utils.data import random_split, DataLoader
import warnings

warnings.filterwarnings("ignore")

import grelu.lightning
import grelu.data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Run the model")
parser.add_argument('--model_type', type=str, required=True, help="Type of Model")
parser.add_argument('--n_transformers', type=int, required=True, help="Number of transformers")
parser.add_argument('--task', type=str, required=True, help="Type of task")
parser.add_argument('--loss', type=str, required=True, help="Type of loss")
parser.add_argument('--expression_type', type=str, required=True,  help="translation of power or standard")
parser.add_argument('--channels', type=int, default=1536, help="Channels")
parser.add_argument('--n_tasks', type=int, default=86, help="Channels")
args = parser.parse_args()

model_type = args.model_type
n_transformers = args.n_transformers
task = args.task
loss = args.loss
expression_type = args.expression_type
channels = args.channels
n_tasks = args.n_tasks

print(f"****--parameter--****")
print(f"--> n_transformers: {n_transformers}")
print(f"--> task: {task}")
print(f"--> loss: {loss}")
print(f"--> expression_type: {expression_type}")
print(f"--> channels: {channels}")

model_params = {
    'model_type': model_type, # Type of model
    'n_tasks': n_tasks, # Number of cell types to predict
    'crop_len':0, # No cropping of the model output
    'n_transformers': n_transformers, # Number of transformer layers; the published Enformer model has 11
    'channels': channels
}

train_params = {
    'task':task, # binary, multiclass, or regressionn
    'loss': loss, # poisson, mse and poisson_multinomial
    'lr':1e-4, # learning rate
    'logger': 'csv', # Logs will be written to a CSV file
    'batch_size': 4,
    'num_workers': 2,
    'devices': 0, # GPU index
    'save_dir': "tutorial_enformer",
    'optimizer': 'adam',
    'max_epochs': 10,
    'checkpoint': True, # Save checkpoints
    'train_label_len ': 86,
    'train_seq_len': 196608
}

model = grelu.lightning.LightningModel(model_params, train_params)

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
df_mmc2_gene_location['len'] = pd.to_numeric(df_mmc2_gene_location['len'], errors='coerce')  # 如果有非数值的值，设置为NaN
df_mmc2_gene_location = df_mmc2_gene_location[df_mmc2_gene_location['len'] < threshold]
selected_gene = df_mmc2_gene_location["gene_id"].to_list()

# load gene expression
mmc2_gene_expression = np.load(mmc2_gene_expression_file, allow_pickle=True)
mmc2_gene_expression_dict = dict(zip(mmc2_gene_expression.item()["genes"], mmc2_gene_expression.item()["expression"]))

# load gene sequence
mmc2_gene_sequence_dict = {}
for record in SeqIO.parse(mmc2_gene_sequence_file, "fasta"):
    gene_id = record.id  # 基因的 ID（通常是描述信息）
    gene_sequence = str(record.seq)  # 基因序列，转换为字符串
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


total_dataset = grelu.data.dataset.GeneExpressionDataset(dataset_dict['sequence'], dataset_dict['expression'])
train_size = int(0.99 * len(total_dataset))
test_size = len(total_dataset)-train_size

print(f"train_size: {train_size}; test_size: {test_size}")

train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

trainer = model.train_on_dataset(
    train_dataset=train_dataset,
    val_dataset=test_dataset
)

