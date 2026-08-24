import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import morphology
from scipy.ndimage import binary_fill_holes
#from tiatoolbox.tools.tissuemask import OtsuTissueMasker
from typing import Dict, List, Tuple
import pandas as pd
import seaborn as sns
from openslide import OpenSlide
from src.preprocess.profiling import profile

def otsu_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_mask = np.logical_not(binary)
    return inverted_mask   

def otsu_wo_blur(image):# without blur
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_mask = np.logical_not(binary)
    return inverted_mask

@profile
def double_pass_tissue_detection(thumbnail, return_masks=False):
    """
    parameters:
    thumbnail: RGB (H, W, 3) image
    return_masks: return all the masks in the process for debugging if True
    
    return:
    mask，background: 0, tissue: 255
    """
    # ==================== first pass: FilterGrays ====================
    # 1. sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(thumbnail, -1, kernel)
    
    # 2. filter gray pixels
    def filter_grays(rgb, tolerance=15, output_type="uint8"):
        """filter gray pixels（R≈G≈B）"""
        (h, w, c) = rgb.shape
        rgb_int = rgb.astype(int)
        rg_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 1]) <= tolerance
        rb_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 2]) <= tolerance
        gb_diff = np.abs(rgb_int[:, :, 1] - rgb_int[:, :, 2]) <= tolerance
        result = ~(rg_diff & rb_diff & gb_diff)
        
        if output_type == "bool":
            return result
        elif output_type == "float":
            return result.astype(float)
        else:  # "uint8"
            return result.astype("uint8")
    
    mask_pass1 = filter_grays(sharpened_image, tolerance=15, output_type="uint8")
    
    # 3. Morphological filtering reduces noise
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(mask_pass1, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)
    
    # 4. remove small objects
    cleaned_mask = morphology.remove_small_objects(
        cleaned_mask.astype(bool), 
        min_size=5000
    )
    mask_pass1 = cleaned_mask.astype(np.uint8)  # 0 or 1
    
    # ==================== second pass: DownsampleKMeans ====================
    # 1. downsampling for KMeans
    small_image = cv2.resize(
        thumbnail,
        (thumbnail.shape[1] // 4, thumbnail.shape[0] // 4),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 2. data for KMeans
    pixel_values = small_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # 3. KMeans (k=2)
    k = 2
    _, labels, centers = cv2.kmeans(
        data=pixel_values,
        K=k,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )
    
    # 4. Determine the tissue cluster (assuming tissue is darker)
    tissue_cluster = np.argmin(np.mean(centers, axis=1))
    
    # 5. Create a small-sized binary mask
    mask_small = np.zeros_like(labels, dtype=np.uint8)
    mask_small[labels == tissue_cluster] = 1
    mask_small = mask_small.reshape(small_image.shape[:2])
    
    # 6. Upsample to the original size
    mask_pass2 = cv2.resize(
        mask_small,
        (thumbnail.shape[1], thumbnail.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 7. Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask_pass2 = cv2.morphologyEx(mask_pass2, cv2.MORPH_CLOSE, kernel)
    mask_pass2 = cv2.morphologyEx(mask_pass2, cv2.MORPH_OPEN, kernel)
    
    # ==================== merging 2 pass ====================
    # 1. logic OR merge 2 masks
    combined_mask = cv2.bitwise_or(mask_pass1, mask_pass2)
    
    # 2. Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # transform to 0-255
    final_mask = (combined_mask > 0).astype(np.uint8) * 255
    
    if return_masks:
        return {
            'final_mask': final_mask,
            'mask_pass1': mask_pass1 * 255,
            'mask_pass2': mask_pass2 * 255,
            'combined_mask': combined_mask * 255,
            'sharpened': sharpened_image,
            'small_image': small_image
        }
    else:
        return final_mask
        
@profile
def double_pass_and_p1wr(thumbnail, return_masks=False):

    # ==================== first pass: FilterGrays ====================
    # 1. sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(thumbnail, -1, kernel)
    
    # 2. filter gray pixels
    def filter_grays(rgb, tolerance=15, output_type="uint8"):
        """filter gray pixels（R≈G≈B）"""
        (h, w, c) = rgb.shape
        rgb_int = rgb.astype(int)
        rg_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 1]) <= tolerance
        rb_diff = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 2]) <= tolerance
        gb_diff = np.abs(rgb_int[:, :, 1] - rgb_int[:, :, 2]) <= tolerance
        result = ~(rg_diff & rb_diff & gb_diff)
        
        if output_type == "bool":
            return result
        elif output_type == "float":
            return result.astype(float)
        else:  # "uint8"
            return result.astype("uint8")
    
    mask_pass1 = filter_grays(sharpened_image, tolerance=15, output_type="uint8")
    
    # ==================== second pass: DownsampleKMeans ====================
    # 1. downsampling for KMeans
    small_image = cv2.resize(
        thumbnail,
        (thumbnail.shape[1] // 4, thumbnail.shape[0] // 4),
        interpolation=cv2.INTER_LINEAR
    )
    
    # 2. data for KMeans
    pixel_values = small_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # 3. KMeans (k=2)
    k = 2
    _, labels, centers = cv2.kmeans(
        data=pixel_values,
        K=k,
        bestLabels=None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )
    
    # 4. Determine the tissue cluster (assuming tissue is darker)
    tissue_cluster = np.argmin(np.mean(centers, axis=1))
    
    # 5. Create a small-sized binary mask
    mask_small = np.zeros_like(labels, dtype=np.uint8)
    mask_small[labels == tissue_cluster] = 1
    mask_small = mask_small.reshape(small_image.shape[:2])
    
    # 6. Upsample to the original size
    mask_pass2 = cv2.resize(
        mask_small,
        (thumbnail.shape[1], thumbnail.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 7. Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask_pass2 = cv2.morphologyEx(mask_pass2, cv2.MORPH_CLOSE, kernel)
    mask_pass2 = cv2.morphologyEx(mask_pass2, cv2.MORPH_OPEN, kernel)
    
    # ==================== merging 2 pass ====================
    # 1. logic AND merge 2 masks
    combined_mask = cv2.bitwise_and(mask_pass1, mask_pass2)
    
    # 2. Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # transform to 0-255
    final_mask = (combined_mask > 0).astype(np.uint8) * 255
    
    if return_masks:
        return {
            'final_mask': final_mask,
            'mask_pass1': mask_pass1 * 255,
            'mask_pass2': mask_pass2 * 255,
            'combined_mask': combined_mask * 255,
            'sharpened': sharpened_image,
            'small_image': small_image
        }
    else:
        return final_mask

def compute_detailed_metrics(ground_truth_mask: np.ndarray, predicted_mask: np.ndarray) -> Dict[str, float]:
    """
    return:
    {
        "iou",
        "dice",
        "precision",
        "recall",
        "accuracy",
        "specificity",
        "f1_score"
    }
    """
    # match dimensions
    if ground_truth_mask.shape != predicted_mask.shape:
        predicted_mask = cv2.resize(
            predicted_mask,
            (ground_truth_mask.shape[1], ground_truth_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # convert to binary
    gt_binary = (ground_truth_mask == 255).astype(np.uint8)
    pred_binary = (predicted_mask == 255).astype(np.uint8)
    
    # Calculate the confusion matrix
    true_positive = np.sum((gt_binary == 1) & (pred_binary == 1))  
    false_positive = np.sum((gt_binary == 0) & (pred_binary == 1))  
    false_negative = np.sum((gt_binary == 1) & (pred_binary == 0))  
    true_negative = np.sum((gt_binary == 0) & (pred_binary == 0))  
    
    # IoU
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = 1.0 if union == 0 else intersection / union
    
    # Dice（F1 score）
    dice = 1.0 if (true_positive + false_positive + false_negative) == 0 else \
           2 * true_positive / (2 * true_positive + false_positive + false_negative)
    
    # Precision
    precision = 1.0 if (true_positive + false_positive) == 0 else \
                true_positive / (true_positive + false_positive)
    
    # Recall
    recall = 1.0 if (true_positive + false_negative) == 0 else \
             true_positive / (true_positive + false_negative)
    
    # Accuracy
    total_pixels = true_positive + false_positive + false_negative + true_negative
    accuracy = (true_positive + true_negative) / total_pixels
    
    # Specificity
    specificity = 1.0 if (true_negative + false_positive) == 0 else \
                  true_negative / (true_negative + false_positive)
    
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "f1_score": float(dice), 
        
        # confusion matrix values
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_negative": int(true_negative),
        
        # pixel statistics
        "total_pixels": int(total_pixels),
        "ground_truth_foreground_pixels": int(np.sum(gt_binary)),
        "predicted_foreground_pixels": int(np.sum(pred_binary))
    }



def load_and_resize_image(image_path: str, target_width: int = 512, resize: bool = False) -> np.ndarray:
    """  
    return:
    resized RGB image
    """
    if image_path.lower().endswith('.svs'):
        # WSI svs file
        slide = OpenSlide(image_path)
        
        # thumbnail
        width, height = slide.dimensions
        if resize:
            target_height = int(target_width * height / width)
            thumbnail = slide.get_thumbnail((target_width, target_height))
        else:
            thumbnail = slide.get_thumbnail((width, height))
        image = np.array(thumbnail)
        
        # RGB format
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        slide.close()
    else:
        # other format
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # size resize
        # height, width = image.shape[:2]
        # target_height = int(target_width * height / width)
        # image = cv2.resize(image, (target_width, target_height), 
        #                   interpolation=cv2.INTER_AREA)
    
    return image


def load_and_resize_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    return:
    resized binary mask (0 or 255)
    """
    # load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask: {mask_path}")
    
    # binary format（0 or 255）
    if mask.max() > 1:
        # foreground
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # resize（nearest neighbor interpolation to maintain the binary property）
    mask = cv2.resize(mask, (target_size[1], target_size[0]), 
                     interpolation=cv2.INTER_NEAREST)
    
    return mask


def batch_evaluate_methods(image_dir: str, mask_dir: str, target_width: int = 512, resize: bool = False) -> Dict:
    """
    return:
    dictionary containing all the assessment results
    """
    # all image files in the directory
    image_extensions = ['.svs', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} image files")
    
    # methods to evaluate
    methods = {
        'Simple_Otsu': otsu_wo_blur,
        'Otsu_with_Blur': otsu_blur,
        'Toolbox_Otsu': complex_otsu,
        'Double_Pass': double_pass_tissue_detection,
        'Double_Pass_AND': double_pass_and_p1wr
    }
    
    # save results
    all_results = {method_name: [] for method_name in methods.keys()}
    
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file}")
        
        try:
            # 1. image
            image_path = os.path.join(image_dir, image_file)
            image = load_and_resize_image(image_path, target_width,resize)
            
            # 2. mask
            mask_name = os.path.splitext(image_file)[0] + '.svs_mask.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(mask_path):
                print(f"Warning: Cannot find corresponding mask: {mask_path}")
                continue
            
            mask = load_and_resize_mask(mask_path, image.shape[:2])
            
            # 3. predict and evaluate each method
            for method_name, method_func in methods.items():
                try:
                    pred_mask = method_func(image)
                    
                    # pred mask should be binary
                    if pred_mask.max() <= 1:
                        pred_mask = pred_mask * 255
                    
                    metrics = compute_detailed_metrics(mask, pred_mask)
                    
                    metrics['image_name'] = image_file
                    metrics['image_size'] = image.shape
                    
                    all_results[method_name].append(metrics)
                    
                    print(f"  {method_name}: IoU={metrics['iou']:.3f}, Dice={metrics['dice']:.3f}")
                    
                except Exception as e:
                    print(f"Error when processing {method_name}: {e}")
                    all_results[method_name].append({
                        'image_name': image_file,
                        'error': str(e)
                    })
        
        except Exception as e:
            print(f"Error when processing {image_file}: {e}")
    
    return all_results


def calculate_average_metrics(all_results: Dict) -> pd.DataFrame:
    """
    return:
    A DataFrame containing the average metrics of each method
    """
    avg_metrics = []
    
    for method_name, results in all_results.items():
        # filter out results with errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            print(f"Warning: {method_name} has no valid results.")
            continue
        
        # average value of each metric
        metrics_summary = {'Method': method_name, 'Number_of_Images': len(valid_results)}
        
        metric_keys = ['iou', 'dice', 'precision', 'recall', 'accuracy', 'specificity']
        
        for key in metric_keys:
            values = [r[key] for r in valid_results]
            metrics_summary[f'mean_{key}'] = np.mean(values)
            metrics_summary[f'std_{key}'] = np.std(values)
            metrics_summary[f'min_{key}'] = np.min(values)
            metrics_summary[f'max_{key}'] = np.max(values)
        
        avg_metrics.append(metrics_summary)
    
    df = pd.DataFrame(avg_metrics)
    
    # sorted by average IoU
    if not df.empty and 'mean_iou' in df.columns:
        df = df.sort_values('mean_iou', ascending=False)
    
    return df

def compute_aggregated_confusion_matrix(all_results, method_name):
    """
    return:
    aggregated_confusion_matrix: 2x2 numpy array
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_pixels = 0
    
    # get method results
    method_results = all_results.get(method_name, [])
    
    # filter out results with errors
    valid_results = [r for r in method_results if 'error' not in r]
    
    if not valid_results:
        print(f"Warning: {method_name} has no valid results.")
        return None, None
    
    # accumulate confusion matrix values
    for result in valid_results:
        total_tp += result.get('true_positive', 0)
        total_fp += result.get('false_positive', 0)
        total_fn += result.get('false_negative', 0)
        total_tn += result.get('true_negative', 0)
    
    total_pixels = total_tp + total_fp + total_fn + total_tn
    
    aggregated_metrics = {
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_tn': total_tn,
        'total_pixels': total_pixels,
        
        # overall metrics
        'overall_accuracy': (total_tp + total_tn) / total_pixels if total_pixels > 0 else 0,
        'overall_precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'overall_recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
        'overall_specificity': total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0,
        
        # IoU & Dice
        'overall_iou': total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0,
        'overall_dice': 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0,
    }
    
    # build confusion matrix (2x2)
    confusion_matrix = np.array([
        [total_tn, total_fp],
        [total_fn, total_tp]
    ])
    
    return confusion_matrix, aggregated_metrics

def plot_detailed_confusion_matrices(all_results, save_path=None):
    """
    plot detailed confusion matrices with raw counts and percentages for each method
    """
    # order of methods
    method_order = ['Simple_Otsu', 'Otsu_with_Blur', 'Toolbox_Otsu', 'Double_Pass']
    
    # subplots
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    all_aggregated_metrics = {}
    
    for idx, method_name in enumerate(method_order):
        row = idx // 2
        col = (idx % 2) * 2 
        
        cm, metrics = compute_aggregated_confusion_matrix(all_results, method_name)
        
        if cm is None:
            axes[row, col].text(0.5, 0.5, f"No data for {method_name}", 
                               ha='center', va='center', fontsize=14)
            axes[row, col].set_title(method_name, fontsize=16, fontweight='bold')
            axes[row, col].axis('off')
            continue
        
        all_aggregated_metrics[method_name] = metrics
        
        # left: raw counts
        ax1 = axes[row, col]
        # right: percentage
        ax2 = axes[row, col+1]
        
        # 1. plot raw counts confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=ax1, cbar_kws={'label': 'Pixel Count'},
                   annot_kws={'size': 12})
        
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xticklabels(['Background', 'Tissue'])
        ax1.set_yticklabels(['Background', 'Tissue'])
        ax1.set_title(f'{method_name}\nRaw Counts', fontsize=14, fontweight='bold')
        
        # 2. plot percentage confusion matrix
        cm_percent = cm / cm.sum() if cm.sum() > 0 else cm
        sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Greens', 
                   ax=ax2, cbar_kws={'label': 'Percentage (%)'},
                   annot_kws={'size': 12})
        
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xticklabels(['Background', 'Tissue'])
        ax2.set_yticklabels(['Background', 'Tissue'])
        
        # add overall metrics text
        metrics_text = (f"Overall Metrics:\n"
                       f"IoU: {metrics['overall_iou']:.4f}\n"
                       f"Dice: {metrics['overall_dice']:.4f}\n"
                       f"Precision: {metrics['overall_precision']:.4f}\n"
                       f"Recall: {metrics['overall_recall']:.4f}")
        
        ax2.text(0.95, 0.05, metrics_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_title(f'{method_name}\nPercentage (%)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Detailed Confusion Matrices Analysis (Aggregated over 26 images)', 
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed confusion matrices are saved to: {save_path}")
    
    plt.show()
    
    return all_aggregated_metrics

def create_confusion_matrix_summary(all_results, output_path=None):
    """
    create a summary table of confusion matrices and metrics for each method
    """
    method_order = ['Simple_Otsu', 'Otsu_with_Blur', 'Toolbox_Otsu', 'Double_Pass','Double_Pass_AND']
    
    summary_data = []
    
    for method_name in method_order:
        cm, metrics = compute_aggregated_confusion_matrix(all_results, method_name)
        
        if cm is None:
            continue
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        
        total_pixels = metrics['total_pixels']
        background_pixels = tn + fp  
        tissue_pixels = fn + tp 
        
        summary_data.append({
            'Method': method_name,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Total Pixels': total_pixels,
            
            'TN (raw)': tn,
            'FP (raw)': fp,
            'FN (raw)': fn,
            'TP (raw)': tp,
            
            'TN (%)': tn/total_pixels*100 if total_pixels > 0 else 0,
            'FP (%)': fp/total_pixels*100 if total_pixels > 0 else 0,
            'FN (%)': fn/total_pixels*100 if total_pixels > 0 else 0,
            'TP (%)': tp/total_pixels*100 if total_pixels > 0 else 0,
            
            'Overall IoU': metrics['overall_iou'],
            'Overall Dice': metrics['overall_dice'],
            'Overall Accuracy': metrics['overall_accuracy'],
            'Overall Precision': metrics['overall_precision'],
            'Overall Recall': metrics['overall_recall'],
            'Overall Specificity': metrics['overall_specificity'],
            
            'Actual Background': background_pixels,
            'Actual Tissue': tissue_pixels,
            'Background %': background_pixels/total_pixels*100 if total_pixels > 0 else 0,
            'Tissue %': tissue_pixels/total_pixels*100 if total_pixels > 0 else 0,
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # ordered by IoU
    if not df_summary.empty and 'Overall IoU' in df_summary.columns:
        df_summary = df_summary.sort_values('Overall IoU', ascending=False)
    
    print("=" * 80)
    print("CONFUSION MATRIX SUMMARY (Aggregated over 26 images)")
    print("=" * 80)
    
    for _, row in df_summary.iterrows():
        print(f"\n{row['Method']}:")
        print(f"  Confusion Matrix (counts):")
        print(f"            Predicted")
        print(f"            Bkg   Tissue")
        print(f"  Actual Bkg  {row['TN']:7d}  {row['FP']:7d}")
        print(f"        Tissue {row['FN']:7d}  {row['TP']:7d}")
        print(f"\n  Overall Metrics:")
        print(f"    IoU: {row['Overall IoU']:.4f}, Dice: {row['Overall Dice']:.4f}")
        print(f"    Accuracy: {row['Overall Accuracy']:.4f}")
        print(f"    Precision: {row['Overall Precision']:.4f}, Recall: {row['Overall Recall']:.4f}")
        print(f"    Specificity: {row['Overall Specificity']:.4f}")
        print(f"  Pixel Distribution:")
        print(f"    Background: {row['Actual Background']:,} pixels ({row['Background %']:.1f}%)")
        print(f"    Tissue: {row['Actual Tissue']:,} pixels ({row['Tissue %']:.1f}%)")
    
    if output_path:
        df_summary.to_csv(output_path, index=False)
        print(f"\nSummary Table saved to: {output_path}")
    
    return df_summary

def plot_comparison_boxplots(all_results: Dict, save_path: str = None):
    import seaborn as sns
    
    plot_data = []
    
    method_order = ['Simple_Otsu', 'Otsu_with_Blur', 'Toolbox_Otsu', 'Double_Pass']
    
    for method_name in method_order:
        if method_name not in all_results:
            continue
            
        # filter out results with errors
        valid_results = [r for r in all_results[method_name] if 'error' not in r]
        
        for result in valid_results:
            plot_data.append({
                'Method': method_name,
                'IoU': result['iou'],
                'Dice': result['dice'],
                'Precision': result['precision'],
                'Recall': result['recall']
            })
    
    if not plot_data:
        print("No valid data to plot.")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    # style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall']
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # boxplot
        sns.boxplot(x='Method', y=metric, data=df_plot, 
                   order=method_order, ax=ax, width=0.6)
        
        # scatter plot with jitter
        sns.stripplot(x='Method', y=metric, data=df_plot, 
                     order=method_order, ax=ax, 
                     color='black', alpha=0.5, size=4, jitter=True)
        
        ax.set_title(f'{metric} Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        ax.set_ylim(0, 1.05)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    plt.suptitle('Comparison of Tissue Segmentation Methods', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Boxplot is saved to: {save_path}")
    
    plt.show()

def plot_comparison_violinplots(all_results: Dict, save_path: str = None):
    plot_data = []
    
    method_order = ['Simple_Otsu', 'Otsu_with_Blur', 'Toolbox_Otsu', 'Double_Pass', 'Double_Pass_AND']
    
    for method_name in method_order:
        if method_name not in all_results:
            continue
            
        valid_results = [r for r in all_results[method_name] if 'error' not in r]
        
        for result in valid_results:
            plot_data.append({
                'Method': method_name,
                'IoU': result['iou'],
                'Dice': result['dice'],
                'Precision': result['precision'],
                'Recall': result['recall']
            })
    
    if not plot_data:
        print("No valid data to plot.")
        return
    
    df_plot = pd.DataFrame(plot_data)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = ['IoU', 'Dice', 'Precision', 'Recall']
    
    method_display_names = {
        'Simple_Otsu': 'Simple Otsu',
        'Otsu_with_Blur': 'Otsu with Blur', 
        'Toolbox_Otsu': 'Toolbox Otsu',
        'Double_Pass': 'Double Pass',
        'Double_Pass_AND': 'Double Pass AND'
    }
    
    df_plot['Method'] = df_plot['Method'].map(method_display_names)
    
    display_order = [method_display_names[m] for m in method_order if m in method_display_names]
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        sns.violinplot(x='Method', y=metric, data=df_plot, 
                      order=display_order, ax=ax, cut=0, inner="box",
                      palette="husl")
        
        # add mean value points
        for i, method in enumerate(display_order):
            method_data = df_plot[df_plot['Method'] == method][metric].values
            if len(method_data) > 0:
                mean_val = np.mean(method_data)
                ax.scatter(i, mean_val, color='white', s=100, zorder=3, 
                          edgecolors='black', linewidth=2, marker='o')
                # add mean value text
                ax.text(i, mean_val + 0.02, f'{mean_val:.3f}', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(f'{metric} Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        ax.set_ylim(0, 1.05)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparison of Tissue Segmentation Methods (Violin Plots)', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Violin plot is saved to: {save_path}")
    
    plt.show()

def save_results_to_csv(all_results: Dict, avg_df: pd.DataFrame, output_dir: str):

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. save detailed results for each method
    for method_name, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            # transform to DataFrame
            df_method = pd.DataFrame(valid_results)
            
            # save to CSV
            csv_path = os.path.join(output_dir, f'{method_name}_detailed_results.csv')
            df_method.to_csv(csv_path, index=False)
            print(f"{method_name} saved to: {csv_path}")
    
    # 2. save average metrics
    if not avg_df.empty:
        avg_path = os.path.join(output_dir, 'average_metrics.csv')
        avg_df.to_csv(avg_path, index=False)
        print(f"Average metrics saved to: {avg_path}")
    
    # 3. save all results combined
    all_data = []
    for method_name, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        for result in valid_results:
            result['Method'] = method_name
            all_data.append(result)
    
    if all_data:
        df_all = pd.DataFrame(all_data)
        all_path = os.path.join(output_dir, 'all_methods_results.csv')
        df_all.to_csv(all_path, index=False)
        print(f"All results saved to: {all_path}")

'''
if __name__ == "__main__":

    IMAGE_DIR = "./dataset/wsi/LIHC"  
    MASK_DIR = "./dataset/wsi/LIHC_binary_mask" 
    OUTPUT_DIR = "./src/out/masking/output"     
    RESIZE = False    
    TARGET_WIDTH = 512                    
    
    print("=" * 60)
    print("Evaluating tissue segmentation methods...")
    print(f"Image Directory: {IMAGE_DIR}")
    print(f"Mask Directory: {MASK_DIR}")
    print(f"Target Width: {TARGET_WIDTH}")
    print("=" * 60)
    
    # 1. evaluate all methods on all images
    print("\nStep 1: processing all images...")
    all_results = batch_evaluate_methods(IMAGE_DIR, MASK_DIR, TARGET_WIDTH, RESIZE)
    
    # 2. average metrics
    print("\nStep 2: calculating average metrics...")
    avg_df = calculate_average_metrics(all_results)
    
    # 3. avguage metrics display
    print("\n" + "=" * 60)
    print("average metrics:")
    print("=" * 60)
    
    if not avg_df.empty:
        print("\nPerformance ranking of each method（descending average IoU）:")
        for idx, row in avg_df.iterrows():
            print(f"\n{row['Method']}:")
            print(f"  average IoU: {row['mean_iou']:.4f} ± {row['std_iou']:.4f}")
            print(f"  average Dice: {row['mean_dice']:.4f} ± {row['std_dice']:.4f}")
            print(f"  average Precision: {row['mean_precision']:.4f} ± {row['std_precision']:.4f}")
            print(f"  average Recall: {row['mean_recall']:.4f} ± {row['std_recall']:.4f}")
            print(f"  Number of Images: {row['Number_of_Images']}")
    else:
        print("No valid average metrics to display.")
    
    # confusion matrix analysis
    print("\n" + "="*60)
    print("Plotting confusion matrix ..")
    print("="*60)

    aggregated_metrics2 = plot_detailed_confusion_matrices(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, "detailed_confusion_matrices.png")
    )
    
    # generate confusion matrix summary
    print("\n" + "="*60)
    print("Generating confusion matrix summary...")
    print("="*60)
    
    df_summary = create_confusion_matrix_summary(
        all_results,
        output_path=os.path.join(OUTPUT_DIR, "confusion_matrix_summary.csv")
    )
    
    # compare error types of each method
    print("\n" + "="*60)
    print("Error types analysis:")
    print("="*60)
    
    for method_name in ['Simple_Otsu', 'Otsu_with_Blur', 'Toolbox_Otsu', 'Double_Pass', 'Double_Pass_AND']:
        if method_name in aggregated_metrics2:
            metrics = aggregated_metrics2[method_name]
            fp_rate = metrics['total_fp'] / metrics['total_pixels'] * 100
            fn_rate = metrics['total_fn'] / metrics['total_pixels'] * 100
            
            print(f"\n{method_name}:")
            print(f"  FP rate: {fp_rate:.3f}%")
            print(f"  FN rate: {fn_rate:.3f}%")
            print(f"  Total error pixels: {metrics['total_fp'] + metrics['total_fn']:,}")
            print(f"  Error rate: {(metrics['total_fp'] + metrics['total_fn']) / metrics['total_pixels'] * 100:.3f}%")

    # 4. boxplots
    print("\nStep 3: plotting boxplots...")
    boxplot_path = os.path.join(OUTPUT_DIR, "method_comparison_boxplots.png")
    plot_comparison_boxplots(all_results, boxplot_path)
    
    # 5. violin plots
    print("\nStep 4: plotting violin plots...")
    violin_path = os.path.join(OUTPUT_DIR, "method_comparison_violinplots.png")
    plot_comparison_violinplots(all_results, violin_path)
    
    # 6. save to CSV
    print("\nStep 5: saving results to files...")
    save_results_to_csv(all_results, avg_df, OUTPUT_DIR)
    
    # 7. best method statistics
    print("\n" + "=" * 60)
    print("Best method statistics:")
    print("=" * 60)
    
    # best method counts
    if not avg_df.empty:
        method_names = list(all_results.keys())
        
        best_counts = {method: 0 for method in method_names}
        
        # image names
        all_images = set()
        for results in all_results.values():
            for r in results:
                if 'error' not in r:
                    all_images.add(r['image_name'])
        
        # highest IoU per image
        for image_name in all_images:
            best_iou = -1
            best_method = None
            
            for method_name in method_names:
                method_results = all_results[method_name]
                for r in method_results:
                    if 'error' not in r and r['image_name'] == image_name:
                        if r['iou'] > best_iou:
                            best_iou = r['iou']
                            best_method = method_name
                        break
            
            if best_method:
                best_counts[best_method] += 1
        
        print("\nCount of best method per image:")
        for method, count in sorted(best_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(all_images) * 100 if all_images else 0
            print(f"  {method}: {count} times ({percentage:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 60)
'''
