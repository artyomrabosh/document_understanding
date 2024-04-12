from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import RobertaTokenizerFast
import os
import pandas as pd
from typing import List, Tuple
    



def pdf_to_tokens(path: str) -> list:
    pass


class DocBankNoImageDataset(Dataset):
    @staticmethod
    def get_vocab(labels: List[List[str]]) -> Tuple[dict, dict]:
        labels_set = set([label for page in labels for label in page])
        label2id = {label: i for i, label in enumerate(labels_set)}
        id2label = {i: label for i, label in enumerate(labels_set)}

        return label2id, id2label

    def __init__(
        self,
        data_dir: str,
        tokenizer: RobertaTokenizerFast = None,
        max_files: int = 100,
        file_extension: str = 'txt'
    ):
        self.data_dir = data_dir
        assert os.path.isdir(self.data_dir)

        self.data = self._get_files(max_files, file_extension)
        self.words, self.bboxes, self.labels = self._load_data()
        self.label2id, self.id2label = self.get_vocab(self.labels)

        if tokenizer is None:
            tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")

        self.tokens = self._encode_words(tokenizer)

    def _get_files(self, max_files: int, file_extension: str) -> List[str]:
        return [
            os.path.join(self.data_dir, path)
            for path in os.listdir(self.data_dir)[:max_files]
            if os.path.isfile(os.path.join(self.data_dir, path)) and path.endswith(file_extension)
        ]

    def _load_data(self) -> Tuple[List[List[str]], List[List[List[int]]], List[List[str]]]:
        words, bboxes, labels = [], [], []

        for txt in self.data:
            with open(txt) as f:
                lines = f.readlines()
                data = [line.split('\t') for line in lines]
                line_words = [line[0] for line in data]
                line_bboxes = [[int(coord) for coord in line[1:5]] for line in data]
                line_labels = [line[-1] for line in data]
                words.append(line_words)
                bboxes.append(line_bboxes)
                labels.append(line_labels)

        return words, bboxes, labels

    def _encode_words(self, tokenizer: RobertaTokenizerFast) -> List[List[int]]:
        return [[tokenizer.encode(word) for word in page] for page in self.words]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.bboxes[idx], self.labels[idx]

def collate_fn_docbank(tokens, bboxes, labels):
    pass

class SpbuDataset:
    def __init__(self, data_dir: str = os.path.join('data', 'spbu', 'latex')):
        self.data_dir = data_dir
        self.folders = self._get_folders()
        self.data = self._load_data()

    def _get_folders(self) -> list:
        return [folder for folder in os.listdir(self.data_dir) if folder.startswith('work_')]

    def _load_data(self) -> list:
        data = []

        for folder in self.folders:
            folder_path = os.path.join(self.data_dir, folder)
            df = pd.read_csv(os.path.join(folder_path, 'text.csv'), sep='\t')
            df_toc = pd.read_csv(os.path.join(folder_path, 'toc.csv'), sep='\t')
            pdf_path = os.path.join(folder_path, f'{folder}.pdf')

            data.append((df, df_toc, pdf_path))

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        return self.data[idx]