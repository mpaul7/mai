import pandas as pd
import numpy as np

def getClassificationReport(confusion_matrix, traffic_classes, byte_count=False):
    """
    Generate a classification report from a confusion matrix.
    
    Args:
        confusion_matrix: numpy array containing the confusion matrix
        traffic_classes: list of class labels
        byte_count: boolean indicating if counts are in bytes (True) or flows (False)
    
    Returns:
        pandas DataFrame containing precision, recall, F1-score and totals
    """
    # Calculate row sums (total actual values per class)
    total_counts = np.sum(confusion_matrix, axis=1)
    
    # Calculate recall and precision
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    
    # Calculate weighted precision
    class_weights = [[count / total for total in total_counts] for count in total_counts]
    weighted_matrix = [[a * b for a, b in zip(*row)] for row in 
                      zip(np.array(confusion_matrix).T.tolist(), class_weights)]
    weighted_precision = np.diag(np.array(confusion_matrix).T) / [sum(row) for row in weighted_matrix]
    
    # Calculate F1 scores
    f1_scores = 2 * (weighted_precision * recall) / (weighted_precision + recall)
    
    # Create DataFrame with results
    df_results = pd.DataFrame(confusion_matrix, 
                            columns=traffic_classes, 
                            index=traffic_classes)
    
    # Add metrics columns
    df_results['total_' + ('bytes' if byte_count else 'flows')] = total_counts
    df_results['recall'] = np.round(recall * 100, 2)
    df_results['precision'] = np.round(precision * 100, 2)
    df_results['weighted_precision'] = np.round(weighted_precision * 100, 2)
    df_results['f1_score'] = np.round(f1_scores * 100, 2)
    
    return df_results