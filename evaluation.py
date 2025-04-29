import evaluate
import numpy as np
from transformers.trainer_utils import EvalPrediction

from constants import ID_TO_LABEL

seq_eval = evaluate.load("seqeval")


def compute_metrics(eval_results: EvalPrediction) -> dict[str, float]:


    pred_logits, gold_tags = eval_results
    pred_tags = np.argmax(pred_logits, axis=2)


    gold_sequences = []
    pred_sequences = []

    for batch_idx in range(len(gold_tags)):

        gold_seq = []
        pred_seq = []

        for pos_idx in range(len(gold_tags[batch_idx])):
            if gold_tags[batch_idx][pos_idx] != -100:
                gold_seq.append(ID_TO_LABEL[gold_tags[batch_idx][pos_idx]])
                pred_seq.append(ID_TO_LABEL[pred_tags[batch_idx][pos_idx]])

        gold_sequences.append(gold_seq)
        pred_sequences.append(pred_seq)


    eval_scores = seq_eval.compute(
        predictions=pred_sequences,
        references=gold_sequences,
        scheme="IOB2",
        mode="strict",
        zero_division=0
    )

    # 4. 返回聚合指标
    return {
        "precision": eval_scores["overall_precision"],
        "recall": eval_scores["overall_recall"],
        "f1": eval_scores["overall_f1"]
    }