import numpy as np
import evaluate


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {**accuracy_metric.compute(predictions=predictions, references=labels),
            **f1_metric.compute(predictions=predictions, references=labels, average='weighted'),
            **precision_metric.compute(predictions=predictions, references=labels, average='weighted'),
            **recall_metric.compute(predictions=predictions, references=labels, average='weighted')}
