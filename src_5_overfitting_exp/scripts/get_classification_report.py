import numpy as np
import pandas as pd

# Define the confusion matrix
_confusion_matrix = np.array([
    [0, 44, 128, 124, 54, 6, 0],
    [1, 28, 324, 299, 88, 0, 0],
    [3, 90, 66, 201, 273, 24, 0],
    [1, 274, 581, 1063, 37, 4, 0],
    [0, 57, 62, 95, 11, 1, 0],
    [0, 22, 211, 276, 216, 0, 0],
    [0, 16, 69, 164, 44, 1, 0]
])

# Define traffic classes
traffic_classes = ["discord", "others", "telegram", "microsoft_teams", "whatsapp", "facebook_messenger", "signal"]

# Define the function
def getClassificationReport(
    _confusion_matrix=_confusion_matrix, 
    traffic_classes=traffic_classes, 
    byte_count=False
):
    total_flows_bytes = np.sum(_confusion_matrix, axis=1)
    recall = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=1)
    precision = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=0)
    _weighted_precision = [
        [a * b for a, b in zip(*l)] 
        for l in zip(np.array(_confusion_matrix).T.tolist(), 
                    [[x / y for y in total_flows_bytes] for x in total_flows_bytes])
    ]
    weighted_precision = np.diag(np.array(_confusion_matrix).T.tolist()) / [sum(l) for l in _weighted_precision]
    f1_Score = [2 * (a * b) / (a + b) if (a + b) > 0 else 0 for a, b in zip(weighted_precision, recall)]
    df_conf_matrix = pd.DataFrame(_confusion_matrix, columns=traffic_classes, index=traffic_classes)

    if byte_count:
        df_conf_matrix['total_bytes'] = total_flows_bytes
    else:
        df_conf_matrix['total_flows'] = total_flows_bytes
    df_conf_matrix['recall'] = np.round(recall * 100, 2)
    df_conf_matrix['precision'] = np.round(precision * 100, 2)

    df_conf_matrix['weighted_precision'] = np.round(weighted_precision * 100, 2)
    df_conf_matrix["f1_score"] = np.round(np.array(f1_Score) * 100, 2)
    return df_conf_matrix

# Generate and print the report
classification_report = getClassificationReport(_confusion_matrix=_confusion_matrix, traffic_classes=traffic_classes)
print(classification_report)

classification_report.to_csv('/home/mpaul/projects/mpaul/mai/results/results_jan10/cross_dataset_v2/4_csv')
