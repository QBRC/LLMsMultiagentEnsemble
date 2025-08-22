import os

import numpy as np
import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, recall_score, precision_score, jaccard_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # for AUROC plots
# import plotly.graph_objects as go
# import plotly.express as px
# from sklearn.metrics import roc_curve, roc_auc_score


""" define function to construct and plot Confusion Matrix, and calculate metrics """

def evaluate_attribute(df,col_ref,col_pred,plot_title,predictor,metrics_for_vals=[],exclude_uncern_ref=False):
    """Evaluate classification by creating confusion matrix and calculating metrics.

    Keyword arguments:
    df -- the DataFrame contains classification data
    col_ref -- the col for reference (or ground truth)
    col_pred -- the col for prediction
    plot_title -- the title of confusion matrix
    predictor -- the predictor that made the prediction
    metrics_for_vals -- a list of values, which the value-centered metrics will be calculated, such as Recall of <value>. (default [])
    exclude_uncern_ref -- Boolean variable, True: evaluation excludes rows, where the reference has any uncertain values
    """
    # Use category variable
    ref = df[col_ref].astype(str)
    pred = df[col_pred].astype(str)

    # # to use binary value, mapping col_ref and col_pred to boolean values
    # ref = df[col_ref].map({'Yes': True, 'No': False})
    # pred = df[col_pred].map({'Yes': True, 'No': False})

    df_metrics = plot_cm_plus_metrics( 
                        reference=ref, 
                        prediction=pred, 
                        plot_title=plot_title,
                        predictor=predictor, 
                        metrics_for_vals=metrics_for_vals,
                        # df_metrics=df_metrics
                    )
    if exclude_uncern_ref:   
        # Evaluate without uncertain values in reference
        df1 = df[~df[col_ref].astype(str).isin(["", 'nan', 'NA', 'Not available', 'Not applicable', 'Cannot be determined', 'Unknown',
                                                  'Absent', 'Information Absent', 'No input data', 'No Input Data', 'No input', 'No Input'])]
    
        ref = df1[col_ref].astype(str)
        pred = df1[col_pred].astype(str)
        coverage = len(df1)/len(df)
        df_metrics = plot_cm_plus_metrics( 
                            reference=ref, 
                            prediction=pred, 
                            plot_title=plot_title,
                            subtitle="Exclude Reference Uncertainty",
                            predictor=predictor, 
                            metrics_for_vals=metrics_for_vals,
                            # df_metrics=df_metrics
                        )
    return df_metrics 


def plot_cm_plus_metrics(
             reference, 
             prediction, 
             plot_title, 
             predictor, 
             metrics_for_vals='micro',
             coverage=1.0,
             subtitle=None,
             labels=None, 
             path_plots=None, 
             df_metrics=None,
             ndigits=4,
             # verbos=True
            ):

    # plot confusion matrix
    plot_cm(
        reference = reference,
        prediction = prediction,
        title = plot_title,
        predictor = predictor,
        subtitle = subtitle,
        labels = labels,
        path_plots = path_plots,
    )
    
    # calculate metrics     
    accuracy = round(accuracy_score(reference, prediction),ndigits)
    
    f1_m = round(f1_score(reference, prediction, average='micro', zero_division=0),ndigits)
    f1_w = round(f1_score(reference, prediction, average='weighted', zero_division=0),ndigits)
    f1 = f1_score(reference, prediction, average=None, zero_division=0)
        
    kappa = round(cohen_kappa_score(reference, prediction),ndigits)
    
    # Jaccard index (Critical Success Index), accurcy of positive prediction: TP/(TP+FP+FN)
    jaccard_m = jaccard_score(reference, prediction, average='micro')
    jaccard_w = jaccard_score(reference, prediction, average='weighted')
    jaccard = jaccard_score(reference, prediction, average=None)
    
    recall_m = round(recall_score(y_true=reference, y_pred=prediction, average='micro'),ndigits) 
    recall_w = round(recall_score(y_true=reference, y_pred=prediction, average='weighted'),ndigits) 
    recall = recall_score(y_true=reference, y_pred=prediction, average=None) 

    # specificity = specificity_score(y_true=reference, y_pred=prediction)
    # print(f"Specificity: {specificity}")
        
    precision_m = round(precision_score(y_true=reference, y_pred=prediction, average='micro'),ndigits)
    precision_w = round(precision_score(y_true=reference, y_pred=prediction, average='weighted'),ndigits)
    precision = precision_score(y_true=reference, y_pred=prediction, average=None)

    df_row = pd.DataFrame({"Predictor": [predictor], 
                           "Accuracy": [accuracy], 
                           "Kappa": [kappa],
                           "F1 (micro)": [f1_m], 
                           "Recall (micro)": [recall_m], 
                           "Precision (micro)": [precision_m], 
                           "Jaccard (micro)": [jaccard_m],
                           "F1 (weighted)": [f1_w], 
                           "Recall (weighted)": [recall_w], 
                           "Precision (weighted)": [precision_w], 
                           "Jaccard (weighted)": [jaccard_w]
                          })
    
    if metrics_for_vals=='micro': # default setting to calculate general overall meterics
        # df_row = pd.DataFrame({"Predictor": [predictor], 
        #                        "Accuracy": [accuracy], 
        #                        "Kappa": [kappa],
        #                        "F1 (micro)": [f1_m], 
        #                        "Recall (micro)": [recall_m], 
        #                        "Precision (micro)": [precision_m], 
        #                        "Jaccard (micro)": [jaccard_m]
        #                       })
        print(f"Accuracy: {accuracy}")
        print(f"Kappa: {kappa}")
        print(f"F1 (micro): {f1_m}")
        print(f"Recall (micro): {recall_m}")    
        print(f"Precision (micro): {precision_m}")
        print(f"Jaccard score (micro): {jaccard_m}")            
    elif metrics_for_vals=='weighted': # default setting to calculate general overall meterics
        # df_row = pd.DataFrame({"Predictor": [predictor], 
        #                        "Accuracy": [accuracy], 
        #                        "Kappa": [kappa],
        #                        "F1 (weighted)": [f1_w], 
        #                        "Recall (weighted)": [recall_w], 
        #                        "Precision (weighted)": [precision_w], 
        #                        "Jaccard (weighted)": [jaccard_w]
        #                       })
        print(f"Accuracy: {accuracy}")
        print(f"Cohen Kappa: {kappa}")
        print(f"F1 (weighted): {f1_w}")
        print(f"Recall (weighted): {recall_w}")    
        print(f"Precision (weighted): {precision_w}")
        print(f"Jaccard score (weighted): {jaccard_w}")            
    else:
        # df_row = pd.DataFrame({"Predictor": [predictor], 
        #                        "Accuracy": [accuracy], 
        #                        "Kappa": [kappa]
        #                       })
        print(f"Accuracy: {accuracy}")
        print(f"Cohen Kappa: {kappa}")
        for v in metrics_for_vals:
            try:
                # gether labels which may or may not present in different subset of data
                vlist = sorted(list(pd.concat([pd.Series(reference.unique()), pd.Series(prediction.unique())], axis=0).unique()))
                if v in vlist:
                    f1_v = round(f1[vlist.index(v)], ndigits)
                    recall_v = round(recall[vlist.index(v)], ndigits)
                    precision_v = round(precision[vlist.index(v)], ndigits)
                    jaccard_v = round(jaccard[vlist.index(v)], ndigits)
                else:
                    f1_v = np.nan
                    recall_v = np.nan
                    precision_v = np.nan
                    jaccard_v = np.nan

                df_row_v = pd.DataFrame({"Predictor": [predictor], 
                                       f"F1 ({v})": [f1_v], 
                                       f"Recall ({v})": [recall_v], 
                                       f"Precision ({v})": [precision_v], 
                                       f"Jaccard ({v})": [jaccard_v],
                                      })
                df_row = df_row.merge(df_row_v,on='Predictor',how='left')
                
                print(f"F1 ({v}): {f1_v}")
                print(f"Recall ({v}): {recall_v}")    
                print(f"Precision ({v}): {precision_v}")
                print(f"Jaccard score ({v})): {jaccard_v}")
            except Exception as e:
                print(f"Error: {e}")
                

    # if coverage < 1.0:
    #     print(f"Coverage: {coverage}")
    print(f"Cases: {len(prediction)}")
    
    if df_metrics is not None:
        df_metrics = pd.concat([df_metrics, df_row], ignore_index=True)
        return df_metrics
    else:
        return df_row

def plot_cm(reference, 
            prediction, 
            title, 
            predictor, 
            labels = None, # Specify labels in an intended order; None: use default order 
            subtitle = None,
            path_plots = None, # specify folder to save plot; None: not save
           ):

    if labels is None:
        labels = list(set(reference.astype(str)) | set(prediction.astype(str)))
        labels.sort()
        labels = move_unknown_to_list_end(labels, ['Other', 'Not specified', 'Cannot be determined'])
    else:
        try:
            if set(labels) != set(reference).uniion(set(prediction)):
                raise Error('Parameter labels do not match the reference and prediction!')
        except Error as e:
            print(f"Error: {e}")
    # print(f" Number of labels: {len(labels)}")
    
    # Calculate the confusion matrix
    cm = confusion_matrix(reference, prediction, labels = labels)

    # Visualize the confusion matrix using a heatmap
    fig_size = (lambda x: 3 if x <=2 else (x if x > 2 and x <= 20 else 20))(len(labels))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=True, xticklabels=labels, yticklabels=labels)
    if subtitle is not None:
        title += " - " + subtitle
    plt.title(title)
    plt.xlabel(f"Value by {predictor}")
    plt.ylabel("Reference")
    fig = plt.gcf()
    plt.show()
    if path_plots is not None:
        filepath = (path_plots + 'CMx_'  
                    + attribute 
                    + '.pdf')                   
        fig.savefig(filepath, format='pdf')
    plt.close()

def move_unknown_to_list_end(list, end_item_list):
    for item in end_item_list:
        if item in list:
            index = list.index(item)
            list.pop(index)
            list.append(item)
    return list

def unique_values(df, attribute):
    if attribute in ['pid', 'subtype']:
        return list(np.unique(list(df[attribute])))
    else:
        return list(np.unique(list(df[attribute]) + list(df[attribute + '_gpt'])))

def none_to_string(x):
    return '' if x is None else x


def specificity_score(y_true, y_pred, labels=None):
    """
    Calculates the specificity for each class in a multi-class classification scenario
    using a one-vs-rest (OvR) approach.

    Specificity (True Negative Rate) is defined as: TN / (TN + FP)

    Args:
        y_true (array-like): True labels, shape (n_samples,).
        y_pred (array-like): Predicted labels, shape (n_samples,).
        labels (list, optional): List of labels to index the matrix. This may be used
                                 to reorder or select a subset of labels.
                                 If None, all unique labels in y_true and y_pred
                                 will be used in sorted order.

    Returns:
        numpy.ndarray: An array of specificity scores, where the order of scores
                       corresponds to the order of classes in the `labels` parameter.
                       Returns NaN for classes if (TN + FP) is zero for that class.
       alternatively:
        dict: A dictionary where keys are class labels and values are their
              corresponding specificity scores. Returns NaN for classes if
              (TN + FP) is zero for that class.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # Convert to numpy arrays for consistent indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if labels is None:
        # Get all unique labels from both true and predicted values
        all_labels = np.unique(np.concatenate((y_true, y_pred)))
        labels = sorted(all_labels) # Ensure consistent order

    specificity_list = []
    # specificity_scores = {}

    for label in labels:
        # Create binary versions of y_true and y_pred for the current class
        # (One-vs-Rest approach)
        binary_y_true = (y_true == label).astype(int)
        binary_y_pred = (y_pred == label).astype(int)

        # Compute the confusion matrix for the binary classification
        # For a binary classification (positive=1, negative=0):
        # TN (True Negative): actual 0, predicted 0
        # FP (False Positive): actual 0, predicted 1
        # FN (False Negative): actual 1, predicted 0
        # TP (True Positive): actual 1, predicted 1
        # The confusion_matrix for binary classification returns:
        # [[TN, FP],
        #  [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(binary_y_true, binary_y_pred, labels=[0, 1]).ravel()

        # Calculate specificity: TN / (TN + FP)
        denominator = tn + fp
        if denominator == 0:
            specificity = np.nan # Avoid division by zero, indicate undefined
        else:
            specificity = tn / denominator

        specificity_list.append(specificity)
        # specificity_scores[label] = specificity
        
    return np.array(specificity_list)        
    # return specificity_scores
