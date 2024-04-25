from vila.predictors import HierarchicalPDFPredictor
import layoutparser as lp
import itertools

# TODO WANDB integration

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
    'figure': 12, # doesn't fit for our dataset
    'service': 12 # service tokens, created by latex
    }

def flatten_group_level_prediction(batched_group_pred, batched_group_word_count):
    final_flatten_pred = []
    for group_pred, group_word_count in zip(
        batched_group_pred, batched_group_word_count
    ):
        assert len(group_pred) == len(group_word_count)
        for (pred, label), (line_id, count) in zip(group_pred, group_word_count):
            final_flatten_pred.append([[pred, label, line_id]] * count)
    return list(itertools.chain.from_iterable(final_flatten_pred))

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
        x1, x2 = x2, x1

    if y1 > y2:
        y1, y2 = y2, y1

    if page_width > target_width or page_height > target_height:

        # Aspect ratio preserving scaling
        scale_factor = target_width / page_width if page_width > page_height else target_height / page_height

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
        if pdf_data['labels'] and pdf_data['labels'][0]:
            pdf_data["labels"] = [label2id[l.lower()] for l in _labels]
        elif pdf_data['labels'] and not pdf_data['labels'][0]:
            pdf_data["labels"] = [0] * len(_labels)
        page_width, page_height = page_size
        _words = pdf_data["words"]
        _bbox = pdf_data["bbox"]
        
        pdf_data["bbox"] = [
            normalize_bbox(box, page_width, page_height) for box in pdf_data["bbox"]
        ]
        sample = self.preprocessor.preprocess_sample(pdf_data)

        # Change back to the original pdf_data
        pdf_data["words"] = _words
        pdf_data["bbox"] = _bbox

        return sample
    
    def postprocess_model_outputs(self, pdf_data, model_inputs, model_predictions, return_type):
        encoded_labels = model_inputs["labels"]
        # for k in model_inputs:
        #     print(k, model_inputs[k])
        # print(encoded_labels)
        # print(model_predictions) 
        if len(model_predictions) == 1:
            true_predictions = [
                [(p, l) for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(model_predictions[0], encoded_labels)
            ]
        else:
            true_predictions = [
                [(p, l) for (p, l) in zip(prediction.reshape(-1), label) if l != -100]
                for prediction, label in zip(model_predictions, encoded_labels)
            ]
        flatten_predictions = flatten_group_level_prediction(
            true_predictions, model_inputs["group_word_count"]
        )
        preds = [label for label, _, _ in flatten_predictions]
        # preds = [self.id2label.get(ele[0], ele[0]) for ele in flatten_predictions]
        # We don't need assertion here because flatten_group_level_prediction
        # already guarantees the length of preds is the same as pdf_data["words"]

        if return_type == "list":
            return preds

        elif return_type == "layout":
            words = pdf_data["words"]
            bboxes = pdf_data["bbox"]

            generated_tokens = []
            for word, pred, bbox in zip(words, preds, bboxes):
                generated_tokens.append(
                    lp.TextBlock(block=lp.Rectangle(*bbox), text=word, type=pred)
                )
