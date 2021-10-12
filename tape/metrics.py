from typing import Sequence, Union
import numpy as np
import scipy.stats

from .registry import registry
from sklearn.preprocessing import normalize


@registry.register_metric('mse')
def mean_squared_error(target: Sequence[float],
                       prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(target: Sequence[float],
                        prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(target: Sequence[float],
              prediction: Sequence[float]) -> float:
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]]) -> float:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total

@registry.register_metric('mean_spectral_angle')
def masked_spectral_distance(true: Sequence[float], pred: Sequence[float], epsilon : float = np.finfo(np.float16).eps):
    true = np.asarray(true)
    pred = np.asarray(pred)
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = normalize(pred_masked)
    true_norm = normalize(true_masked)
    product = np.sum(pred_norm * true_norm, axis=1)
    
    
    arccos = np.arccos(product)
    spectral_distance = 2 * arccos / np.pi
    spectral_distance = 1 - spectral_distance
    spectral_distance = np.nan_to_num(spectral_distance)
    return np.mean(spectral_distance)


@registry.register_metric('fdr')
def masked_spectral_distance(true: Sequence[float], pred: Sequence[float], epsilon : float = np.finfo(np.float16).eps):
    true = np.asarray(true)
    pred = np.asarray(pred)
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = normalize(pred_masked)
    true_norm = normalize(true_masked)

    pred_norm_bool = pred_norm > 0
    true_norm_bool = true_norm > 0

    tp = np.sum((pred_norm_bool == 1) & (true_norm_bool == 1), axis=1)
    fp = np.sum((pred_norm_bool == 1) & (true_norm_bool == 0), axis=1)
    return np.mean(fp/(tp+fp))

@registry.register_metric('fnr')
def masked_spectral_distance(true: Sequence[float], pred: Sequence[float], epsilon : float = np.finfo(np.float16).eps):
    true = np.asarray(true)
    pred = np.asarray(pred)
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)
    
    pred_norm = normalize(pred_masked)
    true_norm = normalize(true_masked)

    pred_norm_bool = pred_norm > 0
    true_norm_bool = true_norm > 0

    tp = np.sum((pred_norm_bool == 1) & (true_norm_bool == 1), axis=1)
    fn = np.sum((pred_norm_bool == 0) & (true_norm_bool == 1), axis=1)
    return np.mean(fn/(tp+fn))