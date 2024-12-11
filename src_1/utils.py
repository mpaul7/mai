import click
import numpy as np
import pandas as pd

from sklearn import pipeline
from importlib import import_module


def import_class(name):
    components = name.split('.')
    mod = import_module('.'.join(components[:-1]))
    return getattr(mod, components[-1])


def get_pipeline(config):
    """Create sklearn pipeline based on configuration."""
    stages = []
    xgb = False
    for stage in config['stages']:
        name = stage.get('name', stage['type'].split('.')[-1].lower())
        clf = stage['type'].split('.')[-1].lower()
        if 'xgb' in clf:
            xgb = True
        klass = import_class(stage['type'])
        stages.append((name, klass(**stage['kwargs'])))

    return pipeline.Pipeline(stages), xgb


def display_report(cm=None, type=None):
    _cm = cm.copy()
    _cm.loc['average'] = _cm[['recall', 'precision', 'weighted_precision', 'f1_score']].mean(axis=0).round(decimals=2)
    report_average = _cm.iloc[[-1], [-4, -3,-2, -1]]
    nl = '\n'
    click.echo(f"{nl}Evaluation Classification Summary Report{nl}{'=' * 41}{nl}{report_average}{nl}")
    click.echo(f"{nl}Evaluation Confusion Matrix{nl}{'='*28}{nl}{cm}{nl}")


def get_classification_report(_confusion_matrix=None, traffic_classes=None, byte_count=False):
    total_flows_bytes = np.sum(_confusion_matrix, axis=1)
    recall = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=1)
    precision = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=0)
    _weighted_precision = [[a * b for a, b in zip(*l)] for l in zip(np.array(_confusion_matrix).T.tolist(),
                                                                    [[x / y for y in total_flows_bytes] for x in
                                                                    total_flows_bytes])]
    weighted_precision = np.diag(np.array(_confusion_matrix).T.tolist()) / [sum(l) for l in _weighted_precision]
    f1_Score = [2 * (a * b) / (a + b) for a, b in zip(weighted_precision, recall)]
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

def plot_bargraph_cm(apps, scores):
    import matplotlib.pyplot as plt

    # Data for plotting (services and their respective F1 scores)
    # apps = [
    #     "Amazon Prime Music", "Amazon Prime Video", "Apple Music", "BitTorrent",
    #     "Dailymotion", "Deezer", "Discord", "Dropbox", "Facebook Messenger",
    #     "Gotomeeting", "Instagram", "Line", "Microsoft Teams", "Netflix",
    #     "Signal", "Skype", "Slack", "Snapchat", "Spotify", "Tango", "Telegram",
    #     "Threema", "TikTok", "Twitch", "Viber", "Webex", "Wechat", "Whatsapp",
    #     "Youtube", "Zoom", "iMessage"
    # ]
    # scores = [
    #     30.23, 80.16, 86.38, 91.51, 66.08, 69.29, 84.58, 37.85, 67.02, 94.79,
    #     83.55, 90.79, 55.09, 89.12, 82.61, 59.48, 66.97, 72.80, 77.24, 76.36,
    #     88.34, 69.58, 77.10, 72.97, 85.16, 83.45, 94.00, 67.53, 71.16, 73.61,
    #     98.42
    # ]

    # Plotting the bar graph
    plt.figure(figsize=(12, 8))
    # plt.barh(apps, scores, color='skyblue')
    plt.barh(apps, scores, color='skyblue')
    plt.xlabel('F1 Score')
    plt.title('F1 Score by Service/Application')
    plt.gca().invert_yaxis()  # Invert y-axis to have highest values at the top
    plt.tight_layout()

    # Display the plot
    plt.show()
