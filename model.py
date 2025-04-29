from transformers import AutoModelForTokenClassification

from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT


def initialize_model():

    model_checkpoint = "Babelscape/wikineural-multilingual-ner"


    model = AutoModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_checkpoint,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        num_labels=len(ID_TO_LABEL),
        ignore_mismatched_sizes=True
    )

    return model