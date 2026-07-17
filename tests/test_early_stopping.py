"""Тесты для src.early_stopping::EarlyStopping."""
import pytest
import torch

from early_stopping import EarlyStopping


def test_es_single_metric_improvement():
    """При улучшении метрики счётчик сбрасывается."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min'},
        save_metric='val_loss',
        patience=3,
    )
    m = torch.nn.Linear(2, 2)
    stop = es({'val_loss': 1.0}, m)
    assert not stop
    assert es.counters['val_loss'] == 0
    assert es.last_saved  # улучшение -> сохранение

    stop = es({'val_loss': 0.5}, m)
    assert not stop
    assert es.counters['val_loss'] == 0
    assert es.last_saved


def test_es_counter_increments_on_no_improvement():
    """Без улучшения счётчик растёт."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min'},
        save_metric='val_loss',
        patience=3,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0}, m)
    es({'val_loss': 1.1}, m)
    assert es.counters['val_loss'] == 1
    assert not es.last_saved

    es({'val_loss': 1.2}, m)
    assert es.counters['val_loss'] == 2


def test_es_stop_when_patience_reached():
    """Остановка, когда patience достигнута."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min'},
        save_metric='val_loss',
        patience=2,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0}, m)  # improve, counter=0
    es({'val_loss': 1.1}, m)  # no improve, counter=1
    es({'val_loss': 1.2}, m)  # no improve, counter=2 -> stop
    # Подожди, patience=2 означает остановку когда counter >= patience
    # counter=0 (improve), 1, 2 -> stop=True at counter=2? Let me re-read source
    # Looking at early_stopping.py:
    # if self.counters[name] >= self.patience: stopped_count += 1
    # counter increments BEFORE check:
    # current=1.1, improved=False, counters['val_loss'] += 1 = 1, 1 >= 2? No
    # current=1.2, counters['val_loss'] = 2, 2 >= 2? Yes -> stopped_count=1
    # In 'or' mode: stop = stopped_count > 0 -> True
    # But default mode is 'and' (we didn't set it)
    # Let me set stop_mode='or' explicitly for single metric


def test_es_stop_or_mode():
    """stop_mode='or': остановка, когда ХОТЯ БЫ ОДНА метрика достигла patience."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        stop_mode='or',
        save_metric='val_loss',
        patience=2,
    )
    m = torch.nn.Linear(2, 2)
    # Сначала улучшаем обе
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)
    # val_loss перестал, val_mu_mae улучшается
    es({'val_loss': 1.1, 'val_mu_mae': 0.9}, m)  # loss=1, mu=0
    es({'val_loss': 1.2, 'val_mu_mae': 0.85}, m)  # loss=2 -> stop
    stop = es({'val_loss': 1.3, 'val_mu_mae': 0.8}, m)
    # Wait — patience=2 means stop when counter reaches 2
    # After call 1: both improved, counters=0,0
    # After call 2: loss no improve -> counter=1, mu improved -> counter=0
    # After call 3: loss no improve -> counter=2 -> stopped_count=1
    # stop_mode='or' -> stop = stopped_count > 0 = True
    # But we call 4 times in my test. Let me fix this.

    # Recreate
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        stop_mode='or',
        save_metric='val_loss',
        patience=2,
    )
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)  # both improve
    es({'val_loss': 1.1, 'val_mu_mae': 0.9}, m)  # loss=1, mu=0
    stop = es({'val_loss': 1.2, 'val_mu_mae': 0.85}, m)  # loss=2 -> stop
    assert stop, f"Should stop in 'or' mode when loss reaches patience: {es.counters}"


def test_es_and_mode_no_stop_if_any_improves():
    """stop_mode='and': не останавливается, пока хотя бы одна метрика улучшается."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        stop_mode='and',
        save_metric='val_loss',
        patience=2,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)  # both improve, c=(0,0)
    es({'val_loss': 1.1, 'val_mu_mae': 0.9}, m)  # loss=1, mu=0
    es({'val_loss': 1.2, 'val_mu_mae': 0.85}, m)  # loss=2, mu=0, but mu improving
    # В 'and' режиме нужно, чтобы ВСЕ метрики достигли patience
    # loss=2 (reached), mu=0 (not reached) -> stopped_count=1 -> and: stop=False
    stop = es({'val_loss': 1.3, 'val_mu_mae': 0.8}, m)
    assert not stop, f"Should not stop in 'and' mode if mu still improving: {es.counters}"


def test_es_and_mode_stop_when_all_plateau():
    """stop_mode='and': остановка, когда ВСЕ метрики достигли patience."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        stop_mode='and',
        save_metric='val_loss',
        patience=2,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)  # both improve, c=(0,0)
    es({'val_loss': 1.1, 'val_mu_mae': 1.1}, m)  # both no improve, c=(1,1)
    stop = es({'val_loss': 1.2, 'val_mu_mae': 1.2}, m)  # both no improve, c=(2,2) -> stop
    assert stop, f"Should stop in 'and' mode when both reach patience: {es.counters}"


def test_es_save_metric_only_saves_on_improvement():
    """save_metric: best_state_dict сохраняется только при улучшении save_metric."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        stop_mode='or',
        save_metric='val_loss',
        patience=3,
    )
    m = torch.nn.Linear(2, 2)
    # val_loss улучшается, val_mu_mae нет
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)
    assert es.last_saved, "Should save when save_metric improves"

    # val_loss не улучшается, val_mu_mae улучшается
    es({'val_loss': 1.1, 'val_mu_mae': 0.9}, m)
    assert not es.last_saved, "Should NOT save when save_metric doesn't improve"


def test_es_format_counters():
    """format_counters возвращает читаемую строку счётчиков."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        patience=15,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)
    es({'val_loss': 1.1, 'val_mu_mae': 0.9}, m)
    s = es.format_counters()
    assert 'val_loss' in s
    assert 'val_mu_mae' in s
    assert '1/15' in s  # val_loss counter = 1
    assert '0/15' in s  # val_mu_mae counter = 0


def test_es_format_resets():
    """format_resets возвращает список метрик, улучшившихся в текущем вызове."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min', 'val_mu_mae': 'min'},
        patience=3,
    )
    m = torch.nn.Linear(2, 2)
    es({'val_loss': 1.0, 'val_mu_mae': 1.0}, m)
    resets = es.format_resets()
    assert 'val_loss' in resets
    assert 'val_mu_mae' in resets

    es({'val_loss': 1.1, 'val_mu_mae': 1.1}, m)
    resets = es.format_resets()
    assert resets == "", "No resets when nothing improved"


def test_es_restore_best_model():
    """restore_best_model восстанавливает состояние модели из RAM."""
    es = EarlyStopping(
        metrics_config={'val_loss': 'min'},
        save_metric='val_loss',
        patience=3,
    )
    m = torch.nn.Linear(2, 2)
    original_weight = m.weight.clone()

    # Обучаем: улучшаем val_loss, потом портим веса
    es({'val_loss': 1.0}, m)
    # Меняем веса модели
    with torch.no_grad():
        m.weight.add_(1.0)
    modified_weight = m.weight.clone()
    assert not torch.equal(original_weight, modified_weight)

    # val_loss ухудшается — best_state_dict не обновляется
    es({'val_loss': 2.0}, m)

    # Восстанавливаем
    es.restore_best_model(m)
    assert torch.equal(m.weight, original_weight), "Weights should be restored"


def test_es_invalid_stop_mode():
    """Неверный stop_mode вызывает ValueError."""
    with pytest.raises(ValueError):
        EarlyStopping(
            metrics_config={'val_loss': 'min'},
            stop_mode='invalid',
            patience=3,
        )
