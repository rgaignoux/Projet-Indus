import numpy as np
import cv2
import argparse
import os
import glob


def calculate_precision_recall_f1(pred_mask, true_mask):
    """
    Fonction permettant de calculer la précision, le rappel et le score F1 pour une seule paire de masques binaires.

    Paramétres:
        pred_mask (numpy.ndarray): mask prédit 0/1
        true_mask (numpy.ndarray): vrai mask 0/1

    Return:
        dict: un dictionnaire qui contient les 3 informations
    """
    # Flatten des images pour comparaison binaire
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # calcul de TP, FP, FN
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))

    # Calcul des métriques
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def evaluate_masks(pred_masks, true_masks):
    """
    Fonction permettant de calculer la précision, le rappel et le score F1 pour une liste de masques binaires prédits et vrais.

    Paramétres:
        pred_masks (numpy.ndarray): liste de masks prédits 0/1
        true_masks (numpy.ndarray): liste de vrais masks 0/1

    Returns:
        dict: A dictionary containing average precision, recall, and F1 score.
    """
    #Vérification de la bonne initialisation des paramétres
    assert len(pred_masks) == len(true_masks)
    #Liste des métriques
    precision_list = []
    recall_list = []
    f1_list = []
    #Parcourt des masks
    for pred_mask, true_mask in zip(pred_masks, true_masks):
        metrics = calculate_precision_recall_f1(pred_mask, true_mask)
        precision_list.append(metrics["precision"])
        recall_list.append(metrics["recall"])
        f1_list.append(metrics["f1_score"])

    # Moyenne des métriques
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)

    #Robin's time afficher une courbe? :( 
    return {
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1
    }


folder="Metrics/toComputeMetrics/"

true_mask_paths = glob.glob(".\\toComputeMetrics\\truth\\*.png")  # Tous les fichiers masques vérités terrain
pred_mask_paths = []

for true_path in true_mask_paths:
    filename = os.path.basename(true_path).replace("segm_", "segm_road_")
    pred_path = os.path.join(".\\toComputeMetrics\\mask", filename)
    pred_mask_paths.append(pred_path)

for k in range(len(true_mask_paths)):
    print(true_mask_paths[k])
    print(pred_mask_paths[k])
    print()

pred_masks = [(cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) for path in pred_mask_paths]
true_masks = [(cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) for path in true_mask_paths]
results = evaluate_masks(pred_masks, true_masks)

print("Evaluation Results:")
print(f"Average Precision: {results['average_precision']:.4f}")
print(f"Average Recall: {results['average_recall']:.4f}")
print(f"Average F1 Score: {results['average_f1_score']:.4f}")
