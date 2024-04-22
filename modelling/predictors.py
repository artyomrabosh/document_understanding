from vila.predictors import HierarchicalPDFPredictor

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

def normalize_bbox(
    bbox,
    page_width,
    page_height,
    target_width=1000,
    target_height=1000,
):
    """
    Normalize bounding box to the target size.
    """

    x1, y1, x2, y2 = bbox

    # Right now only execute this for only "large" PDFs
    # TODO: Change it for all PDFs

    if x1 > page_width or x2 > page_width or y1 > page_height or y2 > page_height:
        return (0, 0, 0, 0)

    if x1 > x2:
        logger.debug(f"Incompatible x coordinates: x1:{x1} > x2:{x2}")
        x1, x2 = x2, x1

    if y1 > y2:
        logger.debug(f"Incompatible y coordinates: y1:{y1} > y2:{y2}")
        y1, y2 = y2, y1

    if page_width > target_width or page_height > target_height:

        # Aspect ratio preserving scaling
        scale_factor = target_width / page_width if page_width > page_height else target_height / page_height

        logger.debug(f"Scaling page as page width {page_width} is larger than target width {target_width} or height {page_height} is larger than target height {target_height}")
        
        x1 = float(x1) * scale_factor
        x2 = float(x2) * scale_factor

        y1 = float(y1) * scale_factor
        y2 = float(y2) * scale_factor

    return (x1, y1, x2, y2)

class SpbuPredictor(HierarchicalPDFPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def preprocess_pdf_data(self, pdf_data, page_size, replace_empty_unicode):
        _labels = pdf_data.get("labels")
        pdf_data["labels"] = [label2id[l.lower()] for l in _labels]
        page_width, page_height = page_size
        _words = pdf_data["words"]
        if replace_empty_unicode:
            pdf_data["words"] = replace_unicode_tokens(
                pdf_data["words"],
                False,
                "[UNK]",
            )

        _bbox = pdf_data["bbox"]
        
        pdf_data["bbox"] = [
            normalize_bbox(box, page_width, page_height) for box in pdf_data["bbox"]
        ]
        sample = self.preprocessor.preprocess_sample(pdf_data)

        # Change back to the original pdf_data
        pdf_data["words"] = _words
        pdf_data["bbox"] = _bbox

        return sample
