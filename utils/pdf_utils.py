import pdfplumber
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pdf2image
from pdfminer.layout import LTChar, LTLine
import re
from tqdm import tqdm
from collections import Counter

COLORS = ['blue', 'green', 'orange', 'violet', 'red', 'brown']
data_dir = os.path.join('data', 'spbu', 'pdf')
poppler_path = r"C:\poppler-24.02.0\Library\bin"

def within_bbox(bbox_bound, bbox_in):
    assert bbox_bound[0] <= bbox_bound[2]
    assert bbox_bound[1] <= bbox_bound[3]
    assert bbox_in[0] <= bbox_in[2]
    assert bbox_in[1] <= bbox_in[3]

    x_left = max(bbox_bound[0], bbox_in[0])
    y_top = max(bbox_bound[1], bbox_in[1])
    x_right = min(bbox_bound[2], bbox_in[2])
    y_bottom = min(bbox_bound[3], bbox_in[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox_in_area = (bbox_in[2] - bbox_in[0]) * (bbox_in[3] - bbox_in[1])

    if bbox_in_area == 0:
        return False

    iou = intersection_area / float(bbox_in_area)

    return iou > 0.95

def extract_tokens_from_page(page):
    tokens = []
    
    width = int(page.width)
    height = int(page.height)
        
    words = page.extract_words(x_tolerance=1.5)
    figures = page.images
    
    lines = []
    for obj in page.layout._objs:
        if not isinstance(obj, LTLine):
            continue
        lines.append(obj)
    
    for word in words:
        word_bbox = (float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom']))
        objs = []
        for obj in page.layout._objs:
            if not isinstance(obj, LTChar):
                continue
            obj_bbox = (obj.bbox[0], float(height) - obj.bbox[3],
                        obj.bbox[2], float(height) - obj.bbox[1])
            if within_bbox(word_bbox, obj_bbox):
                objs.append(obj)
        fontname = []
        for obj in objs:
            fontname.append(obj.fontname)
        if len(fontname) != 0:
            c = Counter(fontname)
            fontname, _ = c.most_common(1)[0]
        else:
            fontname = 'default'

        # format word_bbox
        f_x0 = min(1000, max(0, int(word_bbox[0] / width * 1000)))
        f_y0 = min(1000, max(0, int(word_bbox[1] / height * 1000)))
        f_x1 = min(1000, max(0, int(word_bbox[2] / width * 1000)))
        f_y1 = min(1000, max(0, int(word_bbox[3] / height * 1000)))
        word_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

        # plot annotation
        x0, y0, x1, y1 = word_bbox
        x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
            y1 * height / 1000)

        word_bbox = tuple([str(t) for t in word_bbox])
        word_text = re.sub(r"\s+", "", word['text'])
        tokens.append((word_text,) + word_bbox + (fontname,))
    for figure in figures:
        figure_bbox = (float(figure['x0']), float(figure['top']), float(figure['x1']), float(figure['bottom']))

        # format word_bbox
        f_x0 = min(1000, max(0, int(figure_bbox[0] / width * 1000)))
        f_y0 = min(1000, max(0, int(figure_bbox[1] / height * 1000)))
        f_x1 = min(1000, max(0, int(figure_bbox[2] / width * 1000)))
        f_y1 = min(1000, max(0, int(figure_bbox[3] / height * 1000)))
        figure_bbox = tuple([f_x0, f_y0, f_x1, f_y1])

        # plot annotation
        x0, y0, x1, y1 = figure_bbox
        x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
            y1 * height / 1000)
        figure_bbox = tuple([str(t) for t in figure_bbox])
        word_text = '##LTFigure##'
        fontname = 'default'
        tokens.append((word_text,) + figure_bbox + (fontname,))
    
    return tokens

def save_images(data_dir, path):
    
    pdf_name = path[:-4]
    try:
        pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, path))
    except Exception:
        try:
            pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, path), poppler_path=poppler_path)
        except:
            return False
    for i, image in enumerate(pdf_images):
        image.save(os.path.join(data_dir, pdf_name, pdf_name + f'_{i}.jpg'))
    return True

def save_tokens(tokens, page_num, pdf_dir, pdf_name):
    with open(os.path.join(pdf_dir, pdf_name + f'_{page_num}.txt'), 'w', encoding='utf8') as f:
        for token in tokens:
            f.write('\t'.join(token) + '\n')

def extract_pages_from_pdf(data_dir, path):
    if path[-3:] != 'pdf':
        return
    
    pdf_name = path[:-4]
    pdf_dir = os.path.join(data_dir, pdf_name)
    try:
        os.mkdir(pdf_dir)
    except:
        pass

    if not save_images(data_dir, path):
        return
    
    try:
        pdf = pdfplumber.open(os.path.join(data_dir, path))
    except Exception:
        return
    
    
    pages = pdf.pages
    
    for page_num, page in enumerate(pages):
        tokens = extract_tokens_from_page(page)
        save_tokens(tokens, page_num, pdf_dir, pdf_name)
    return

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