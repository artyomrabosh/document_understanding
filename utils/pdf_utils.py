import pdfplumber
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pdf2image
from pdfminer.layout import LTChar, LTLine
import re
from tqdm import tqdm
from collections import Counter
from vila.pdftools.pdfplumber_extractor import PDFPlumberTokenExtractor
from vila.pdftools.pdf_extractor import PDFExtractor
import layoutparser as lp
import pandas as pd

COLORS = ['blue', 'green', 'orange', 'violet', 'red', 'brown']
# data_dir = os.path.join('data', 'spbu', 'pdf')
# poppler_path = r"C:\poppler-24.02.0\Library\bin"

def load_tokens_labeled(work_path):
    pass


class TrainingPDFExtractor(PDFExtractor):
    def __init__(self, pdf_extractor_name, **kwargs):
        self.vision_model = lp.EfficientDetLayoutModel("lp://PubLayNet", 
                                                       extra_config={"output_confidence_threshold": 0.1}) 

        
    def load_tokens_and_image(
        self, work_idx: int, **kwargs):
        ground_truth = pd.read_csv(f'data/spbu/latex/work_{work_idx}/text.csv', sep='\t', index_col=0)
        
        if resize_layout:
            for image, page in zip(page_images, pdf_tokens):
                width, height = image.size
                resize_factor = width / page.width, height / page.height
                page.tokens = page.tokens.scale(resize_factor)
                page.image_height = height
                page.image_width = width

        elif resize_image:
            page_images = [
                image.resize((int(page.width), int(page.height)))
                if page.width != image.size[0]
                else image
                for image, page in zip(page_images, pdf_tokens)
            ]

        return pdf_tokens, page_images


def plot_page(img_path, bboxes, labels, id2label=None):
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for box, label in zip(bboxes, labels):
        general_box = box.copy()
        general_box[0] *= 1.7
        general_box[2] *= 1.7
        general_box[1] *= 2.2
        general_box[3] *= 2.2
        draw.rectangle(general_box, outline=COLORS[label], width=2)
        if id2label:
            draw.text((general_box[0] + 10, general_box[1] - 10), id2label[label], fill=COLORS[label], font=font)
    
    image.show()