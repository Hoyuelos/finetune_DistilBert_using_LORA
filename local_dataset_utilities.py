import os
import os.path as op
import sys
import tarfile
import time

from datasets import load_dataset
import numpy as np
import pandas as pd
from packaging import version
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import urllib


# def reporthook(count, block_size, total_size):
#     global start_time
#     if count == 0:
#         start_time = time.time()
#         return
#     duration = time.time() - start_time
#     progress_size = int(count * block_size)
#     speed = progress_size / (1024.0**2 * duration)
#     percent = count * block_size * 100.0 / total_size

#     sys.stdout.write(
#         f"\r{int(percent)}% | {progress_size / (1024.**2):.2f} MB "
#         f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
#     )
#     sys.stdout.flush()


# def download_dataset():
#     source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#     target = "aclImdb_v1.tar.gz"

#     if os.path.exists(target):
#         os.remove(target)

#     if not os.path.isdir("aclImdb") and not os.path.isfile("aclImdb_v1.tar.gz"):
#         urllib.request.urlretrieve(source, target, reporthook)

#     if not os.path.isdir("aclImdb"):

#         with tarfile.open(target, "r:gz") as tar:
#             tar.extractall()


def download_dataset():
    dataset = load_dataset("financial_phrasebank", "sentences_50agree")
    return dataset

def load_dataset_into_to_dataframe():

    labels = {"positive": 2, "negative": 0, "neutral":1}

    df = pd.DataFrame()

    financial_datatset = download_dataset()
    df_financial = pd.DataFrame(financial_datatset)

    # with tqdm(total=50000) as pbar:
    #     for s in ("test", "train"):
    #         for l in ("pos", "neg"):
    #             path = os.path.join(basepath, s, l)
    #             for file in sorted(os.listdir(path)):
    #                 with open(os.path.join(path, file), "r", encoding="utf-8") as infile:
    #                     txt = infile.read()

    #                 if version.parse(pd.__version__) >= version.parse("1.3.2"):
    #                     x = pd.DataFrame(
    #                         [[txt, labels[l]]], columns=["review", "sentiment"]
    #                     )
    #                     df = pd.concat([df, x], ignore_index=False)

    #                 else:
    #                     df = df.append([[txt, labels[l]]], ignore_index=True)
    #                 pbar.update()

    
    for index, row in df_financial.iterrows():
        dict = row['train']
        text = dict['sentence']
        label = dict['label']
        df = df._append([[text, label]], ignore_index=True)

    df.columns = ["text", "label"]
    
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))

    print("Class distribution:")
    np.bincount(df["label"].values)

    # save the entire data file in data directory
    df.to_csv('data/data.csv', sep='\t')

    return df


def partition_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()

    #df_train = df_shuffled.iloc[:35_000]
    df_train = df_shuffled.iloc[:3395]          # 70% of entire data
    df_val = df_shuffled.iloc[3395:3880]        # 10% of data
    df_test = df_shuffled.iloc[3880:]           # 20% of entire data

    if not op.exists("data"):
        os.makedirs("data")
    df_train.to_csv(op.join("data", "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(op.join("data", "val.csv"), index=False, encoding="utf-8")
    df_test.to_csv(op.join("data", "test.csv"), index=False, encoding="utf-8")


# class IMDBDataset(Dataset):
#     def __init__(self, dataset_dict, partition_key="train"):
#         self.partition = dataset_dict[partition_key]

#     def __getitem__(self, index):
#         return self.partition[index]

#     def __len__(self):
#         return self.partition.num_rows
    
class FinancialDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, index):
        return self.partition[index]

    def __len__(self):
        return self.partition.num_rows


def get_dataset():
    files = ("test.csv", "train.csv", "val.csv")
    download = True

    for f in files:
        if not os.path.exists(f):
            download = False

    if download is False:
        #download_dataset()
        df = load_dataset_into_to_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv(op.join("data", "train.csv"))
    df_val = pd.read_csv(op.join("data", "val.csv"))
    df_test = pd.read_csv(op.join("data", "test.csv"))

    return df_train, df_val, df_test


def tokenization():
    financial_dataset = load_dataset(
        "csv",
        data_files={
            "train": op.join("data", "train.csv"),
            "validation": op.join("data", "val.csv"),
            "test": op.join("data", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=512)

    financial_tokenized = financial_dataset.map(tokenize_text, batched=True, batch_size=None)
    financial_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return financial_tokenized


def setup_dataloaders(financial_tokenized):
    train_dataset = FinancialDataset(financial_tokenized, partition_key="train")
    val_dataset = FinancialDataset(financial_tokenized, partition_key="validation")
    test_dataset = FinancialDataset(financial_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=8,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=4
    )
    return train_loader, val_loader, test_loader
