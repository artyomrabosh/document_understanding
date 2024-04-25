from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import RobertaTokenizerFast
import os
import pandas as pd
from typing import List, Tuple, Dict
import torch
import pdfplumber
import bisect
import numpy as np 


label2id = {
    'paragraph': 0,
    'title': 1,
    'equation': 2,
    'reference': 3,
    'section': 4,
    'list': 5,
    'table': 6,
    'caption': 7,
    'author': 8,
    'abstract': 9,
    'footer': 10,
    'date': 11,
    'figure': 12,
    'service': 12
    }

def fill_with_nearest_label(row, pagedata: pd.DataFrame):
    if pd.isna(row['label']):
        # Find the index of the current row
        current_index = pagedata.index[pagedata['token'] == row['token']].tolist()[0]

        # Find the index of the next non-NaN token
        next_non_nan_index = pagedata[current_index:].first_valid_index('label')

        if next_non_nan_index is not None:
            # Return the label of the next non-NaN token
            return pagedata.loc[next_non_nan_index, 'label']
        else:
            return 'Service'  # or another default label if there are no more non-NaN tokens
    else:
        return row['label']

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


class SpbuDataset:
    """
    A custom dataset class to load SPBU data, which consists of text CSV files, table of contents CSV files,
    and corresponding PDF files.
    """

    def __init__(self, data_dir: str = os.path.join('data', 'spbu', 'latex')):
        """
        Initialize the SpbuDataset object.

        Args:
            data_dir (str, optional): The path to the directory containing the SPBU data. Defaults to os.path.join('data', 'spbu', 'latex').
        """
        self.data_dir = data_dir
        self.folders = self._get_folders()
        self.data = self._load_data()

        self.cumulative_pages = [0]
        self.page_sizes = []

        for item in self.data:
            df, _, pdf_path = item
            self.cumulative_pages.append(self.cumulative_pages[-1] + len(df['page'].unique()))
            with pdfplumber.open(pdf_path) as doc:
                for page in doc.pages:
                    width, height = page.width, page.height
                    self.page_sizes.append((width, height))


    def _get_folders(self) -> List[str]:
        """
        Get a list of folders in the data directory that start with 'work_'.

        Returns:
            List[str]: A list of folder names.
        """
        return [folder for folder in os.listdir(self.data_dir) if folder.startswith('work_')]

    def _load_data(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
        """
        Load the data from the text and table of contents CSV files and store the corresponding PDF paths.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame, str]]: A list of tuples containing text DataFrame, table of contents DataFrame, and PDF path.
        """
        data = []

        for folder in self.folders:
            folder_path = os.path.join(self.data_dir, folder)
            df = pd.read_csv(os.path.join(folder_path, 'text.csv'), sep='\t')
            df_toc = pd.read_csv(os.path.join(folder_path, 'toc.csv'), sep='\t')
            pdf_path = os.path.join(folder_path, f'{folder}.pdf')

            data.append((df, df_toc, pdf_path))

        return data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.page_sizes)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, List], Tuple[int, int]]:
        """
        Retrieve tokens and other data for a specific global page index.

        Args:
            idx (int): The global page index to retrieve data for.

        Returns:
            page (Dict[str, List]): A dictionary containing tokens and other data from the specified page.
            page_size (Tuple[int, int]): The dimensions of the specified page.
        """
        if not isinstance(idx, int) or idx < 0 or idx >= self.cumulative_pages[-1]:
            raise IndexError("Invalid index. Expected a non-negative integer less than the total number of pages.")

        # Find the item and the local page number corresponding to the global page index
        item_idx = bisect.bisect_right(self.cumulative_pages, idx) - 1
        page_number = idx - self.cumulative_pages[item_idx] + 1
        page_size = self.page_sizes[idx]

        df, _, _ = self.data[item_idx]

        # Find the rows where the page number matches
        pagedata = df[df['page'] == page_number]

        if len(pagedata) > 0:
            def fill_with_nearest_label(row, pagedata: pd.DataFrame = pagedata):
                if pd.isna(row['label']):
                    # Find the index of the current row
                    current_index = pagedata.index[pagedata['token'] == row['token']].tolist()[0]
                
                    # Find the index of the next non-NaN token
                    next_non_nan_index = df['label'][current_index:].first_valid_index()
                    if next_non_nan_index is not None:
                        # Return the label of the next non-NaN token
                        new_label = df.loc[next_non_nan_index, 'label']
                        new_block_id = df.loc[next_non_nan_index, 'block_id']
                        return new_label, new_block_id 
                    else:
                        return 'Service', -1  # or another default label if there are no more non-NaN tokens
                else:
                    return row['label'], row['block_id']
            pagedata['label'], pagedata['block_id'] = zip(*pagedata.apply(fill_with_nearest_label, axis=1))

            page = {
                "words": list(pagedata['token']),
                "labels": list(pagedata['label']),
                "block_ids": list(pagedata['block_id']),
                "bbox": [t for t in zip(pagedata.x0, pagedata.y0, pagedata.x1, pagedata.y1)],
            }

            return page, page_size

        raise ValueError(f"Page {page_number} not found in item {item_idx} of the dataset.")