from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# --------------------------------------------------------------------------- #
#                                 Data models                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ThresholdSpec:
    label: str
    threshold: float
    predictions: np.ndarray


@dataclass
class ThresholdResult:
    threshold_label: str
    threshold: float
    f1: float
    recall: float
    precision: float
    adjusted_precision: float
    balanced_accuracy: float
    actual_positive_rate: float


# --------------------------------------------------------------------------- #
#                                   Helpers                                   #
# --------------------------------------------------------------------------- #

def _unpack_model(_model):
    """Разворачивает bundle (model + калибровка + порог) или чистую модель."""
    if isinstance(_model, dict):
        scale = _model.get('calibration_scale', 1.0)
        decision_threshold = _model.get('decision_threshold', 0.5)
        model = _model['model']
    else:
        scale = 1.0
        decision_threshold = 0.5
        model = _model
    return model, scale, decision_threshold


def _calibrated_proba(model, X, scale: float) -> np.ndarray:
    raw = model.predict_proba(X)[:, 1]
    return np.clip(raw * scale, 0.0, 1.0)


def _best_ba_threshold(y_proba: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """
    Возвращает (порог, BA) для порога, максимизирующего Balanced Accuracy.

    Векторизованно: сортируем по вероятности, накапливаем TP/FP при движении
    порога справа налево (от самого высокого значения вероятности).
    """
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        # BA не определена; возвращаем дефолт, чтобы не ломать пайплайн
        return 0.5, float('nan')

    order = np.argsort(y_proba, kind='stable')[::-1]      # по убыванию proba
    y_sorted = y_true[order]
    p_sorted = y_proba[order]

    # Идём по порогу сверху вниз: сначала все предсказаны как 0,
    # затем постепенно добавляем объекты как "1".
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / P
    tnr = 1.0 - fp / N                                     # = (N - fp) / N
    ba = 0.5 * (tpr + tnr)

    # Порог должен меняться только на уникальных значениях вероятности,
    # иначе получим промежуточные (бессмысленные) точки внутри плато.
    # Берём последний индекс в каждой группе одинаковых proba.
    _, last_idx = np.unique(p_sorted, return_index=True)   # индексы первых вхождений
    # np.unique на убывающем массиве даёт уникальные значения в порядке возрастания
    # и первые индексы в этом (уже убывающем) массиве — это концы групп.
    valid_idx = np.sort(last_idx)

    ba_at_thresholds = ba[valid_idx]
    best_pos = int(np.argmax(ba_at_thresholds))
    best_idx = valid_idx[best_pos]

    best_ba = float(ba_at_thresholds[best_pos])
    best_threshold = float(p_sorted[best_idx])
    return best_threshold, best_ba


def _topk_threshold(y_proba: np.ndarray, target_rate: float) -> tuple[float, np.ndarray]:
    """Порог, при котором доля позитивов ≈ target_rate (top-k по вероятности)."""
    if not 0 <= target_rate <= 1:
        raise ValueError('Positive rates must be between 0 and 1')

    n = len(y_proba)
    n_positive = int(n * target_rate)

    sorted_idx = np.argsort(-y_proba, kind='stable')
    predictions = np.zeros(n, dtype=int)
    predictions[sorted_idx[:n_positive]] = 1

    if n_positive == 0:
        threshold = np.inf
    else:
        threshold = float(y_proba[sorted_idx[n_positive - 1]])
    return threshold, predictions


def _adjusted_precision(P: float, N: int, M: int, new_N: int, new_M: int) -> float:
    """Precision, пересчитанная под новый баланс классов (new_N : new_M)."""
    if N == 0 or M == 0 or np.isnan(P):
        return float('nan')
    numerator = P * (new_N / N)
    denominator = numerator + (1 - P) * (new_M / M)
    return numerator / denominator if denominator else 0.0


def _plot_confusion(y_true, y_pred, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sn.set(font_scale=1.4)
    sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.show()


def _plot_distribution(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    title: str,
    actual_positive_rate: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_proba[y_true == 0], bins=50, color='blue', alpha=0.5,
            label='Class 0', edgecolor='black')
    ax.hist(y_proba[y_true == 1], bins=50, color='green', alpha=0.5,
            label='Class 1', edgecolor='black')
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold:.3f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Number of Predictions')
    ax.set_title(
        f'Combined Distribution - {title}\n'
        f'Actual Positive Rate: {actual_positive_rate * 100:.1f}%'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _build_threshold_specs(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    decision_threshold: float,
    target_positive_rates: Iterable[float],
) -> List[ThresholdSpec]:
    """Собирает все варианты порогов, которые нужно протестировать."""
    specs: List[ThresholdSpec] = []

    # 1) Фиксированный порог из bundle
    specs.append(ThresholdSpec(
        label='Fixed threshold',
        threshold=decision_threshold,
        predictions=(y_proba >= decision_threshold).astype(int),
    ))

    # 2) Порог по максимальному Balanced Accuracy
    ba_threshold, ba_value = _best_ba_threshold(y_proba, y_true)
    print(f"Best BA threshold = {ba_threshold:.4f} (BA = {ba_value:.4f})")
    specs.append(ThresholdSpec(
        label='Best BA',
        threshold=ba_threshold,
        predictions=(y_proba >= ba_threshold).astype(int),
    ))

    # 3) Квантильные пороги под целевые доли позитивов
    for rate in target_positive_rates:
        threshold, preds = _topk_threshold(y_proba, rate)
        specs.append(ThresholdSpec(
            label=f'{rate * 100:.0f}% Positive',
            threshold=threshold,
            predictions=preds,
        ))

    return specs


def _evaluate_spec(
    spec: ThresholdSpec,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    N_1: int,
    N_0: int,
    *,
    plot: bool,
) -> ThresholdResult:
    preds = spec.predictions
    actual_positive_rate = float(preds.mean())

    f1 = f1_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    precision = precision_score(y_true, preds, zero_division=0)
    precision_adj = _adjusted_precision(precision, N_1, N_0, new_N=1000, new_M=9000)
    ba = balanced_accuracy_score(y_true, preds)

    print(f"\n{'=' * 60}")
    print(f"Testing with threshold = {spec.threshold:.4f} ({spec.label})")
    print(f"{'=' * 60}")
    print(f"CatBoost result: F1 = {f1:.2f}, Recall = {recall:.2f}, Precision = {precision:.2f}")
    print(f"Adjusted Precision = {precision_adj:.2f}, Balanced Accuracy = {ba:.2f}")
    print(f"Actual Positive Rate: {actual_positive_rate * 100:.1f}%")

    # save_value_to_csv(
    #     ba,
    #     threshold_label=spec.label,
    #     threshold=spec.threshold,
    #     actual_positive_rate=actual_positive_rate,
    # )

    if plot:
        _plot_confusion(y_true, preds, spec.label)
        _plot_distribution(y_proba, y_true, spec.threshold, spec.label, actual_positive_rate)

    return ThresholdResult(
        threshold_label=spec.label,
        threshold=spec.threshold,
        f1=f1,
        recall=recall,
        precision=precision,
        adjusted_precision=precision_adj,
        balanced_accuracy=ba,
        actual_positive_rate=actual_positive_rate,
    )


def _print_summary(results: Sequence[ThresholdResult]) -> None:
    header = (
        f"{'Threshold Type':<20} {'Threshold':<10} {'F1':<8} "
        f"{'Recall':<8} {'Precision':<10} {'Adjusted':<10} "
        f"{'BA':<8} {'Pos Rate':<10}"
    )
    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 60}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.threshold_label:<20} {r.threshold:<10.4f} "
            f"{r.f1:<8.3f} {r.recall:<8.3f} {r.precision:<10.3f} "
            f"{r.adjusted_precision:<10.3f} {r.balanced_accuracy:<8.3f} "
            f"{r.actual_positive_rate * 100:<10.1f}%"
        )


# --------------------------------------------------------------------------- #
#                                 Main entry                                  #
# --------------------------------------------------------------------------- #

def test(
    _model,
    trn,
    trg,
    target_positive_rates: Sequence[float] = (0.12,),
    *,
    plot: bool = True,
):
    """
    Оценивает модель при нескольких стратегиях выбора порога:
      * фиксированный порог из bundle (`decision_threshold`, по умолчанию 0.5);
      * порог, максимизирующий Balanced Accuracy на `trn`;
      * квантильные пороги под заданные доли позитивов (`target_positive_rates`).

    Возвращает список ThresholdResult.
    """
    model, scale, decision_threshold = _unpack_model(_model)
    print(f"Calibration scale = {scale:.4f}")

    y_proba = _calibrated_proba(model, trn, scale)
    y_true = np.asarray(trg)

    N_1 = int((y_true == 1).sum())
    N_0 = int((y_true == 0).sum())
    if N_1 == 0 or N_0 == 0:
        print("WARNING: в trg присутствует только один класс — часть метрик будет NaN.")

    specs = _build_threshold_specs(
        y_proba, y_true, decision_threshold, target_positive_rates,
    )

    results = [
        _evaluate_spec(spec, y_proba, y_true, N_1, N_0, plot=plot)
        for spec in specs
    ]

    _print_summary(results)
    return results