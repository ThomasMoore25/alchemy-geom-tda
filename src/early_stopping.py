"""Multi-metric Early Stopping для нескольких целевых метрик.

Возможности:
  - Отслеживание нескольких метрик (val_loss, val_mu_mae, val_alpha_mae, val_gap_mae)
  - stop_mode='and': ждёт, пока ВСЕ метрики перестанут улучшаться
  - stop_mode='or': срабатывает, когда ХОТЯ БЫ ОДНА метрика перестала улучшаться
  - save_metric: метрика, по которой сохраняется best checkpoint
  - format_counters(): строка вида [val_loss: 3/15 | val_mu_mae: 0/15 | ...]
  - format_resets(): строка вида RESET:val_mu_mae,val_alpha_mae (метрики, улучшившиеся в текущем вызове)
  - last_saved: флаг, что save_metric улучшился → сохранён best ckpt
  - restore_best_model(): восстановление лучшего состояния модели из RAM
"""
class EarlyStopping:
    def __init__(self, metrics_config, stop_mode='and', save_metric=None, patience=3, min_delta=0.0):
        if stop_mode not in ('and', 'or'):
            raise ValueError(f"stop_mode must be 'and' or 'or', got: {stop_mode}")
        self.metrics_config = metrics_config
        self.stop_mode = stop_mode
        self.save_metric = save_metric
        self.patience = patience
        self.min_delta = min_delta

        self.best_values = {}
        self.counters = {}
        for name, direction in metrics_config.items():
            self.best_values[name] = float('inf') if direction == 'min' else float('-inf')
            self.counters[name] = 0
        self.best_state_dict = None

        # v27: результаты последнего вызова __call__
        self.last_reset_metrics = []   # какие метрики улучшились → RESET
        self.last_saved = False        # был ли сохранён best ckpt

    def __call__(self, metrics_dict, model):
        improved_any = False
        stopped_count = 0
        self.last_reset_metrics = []  # сбрасываем перед новым проходом
        self.last_saved = False

        for name, direction in self.metrics_config.items():
            current = metrics_dict.get(name)
            if current is None:
                continue
            best = self.best_values[name]
            improved = current < best - self.min_delta if direction == 'min' else current > best + self.min_delta

            if improved:
                self.best_values[name] = current
                self.counters[name] = 0
                improved_any = True
                self.last_reset_metrics.append(name)  # v27: запоминаем RESET
            else:
                self.counters[name] += 1
                if self.counters[name] >= self.patience:
                    stopped_count += 1

        # Логика сохранения
        if self.save_metric is None:
            save = improved_any
        else:
            curr = metrics_dict.get(self.save_metric)
            best = self.best_values[self.save_metric]
            # Внимание: best_values уже обновлён выше, если было улучшение.
            # Поэтому сравниваем с current и previous best через приближение:
            # save = True iff save_metric в last_reset_metrics
            save = self.save_metric in self.last_reset_metrics

        if save:
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.last_saved = True  # v27: флаг для лога

        total = len(self.metrics_config)
        # stop_mode проверяется в __init__, здесь безопасно
        if self.stop_mode == 'and':
            stop = stopped_count == total
        else:  # 'or'
            stop = stopped_count > 0
        return stop

    def format_counters(self) -> str:
        """v27: строка вида [val_loss: 3/15 | val_mu_mae: 0/15 | ...]."""
        parts = []
        for name in self.metrics_config:
            c = self.counters.get(name, 0)
            parts.append(f"{name}: {c}/{self.patience}")
        return "[" + " | ".join(parts) + "]"

    def format_resets(self) -> str:
        """v27: строка вида RESET:val_mu_mae,val_alpha_mae или пустая."""
        if not self.last_reset_metrics:
            return ""
        return "RESET:" + ",".join(self.last_reset_metrics)

    def restore_best_model(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    def state_dict(self):
        return {'best_values': self.best_values, 'counters': self.counters, 'best_state_dict': self.best_state_dict}

    def load_state_dict(self, state):
        self.best_values = state['best_values']
        self.counters = state['counters']
        self.best_state_dict = state.get('best_state_dict')
