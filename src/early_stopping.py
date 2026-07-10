"""Early Stopping — останавливает обучение, если метрики не улучшаются."""
class EarlyStopping:
    def __init__(self, metrics_config, stop_mode='and', save_metric=None, patience=3, min_delta=0.0):
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

    def __call__(self, metrics_dict, model):
        improved_any = False
        stopped_count = 0

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
            else:
                self.counters[name] += 1
                if self.counters[name] >= self.patience:
                    stopped_count += 1

        if self.save_metric is None:
            save = improved_any
        else:
            curr = metrics_dict.get(self.save_metric)
            best = self.best_values[self.save_metric]
            save = curr is not None and (curr < best - self.min_delta if self.metrics_config[self.save_metric] == 'min' else curr > best + self.min_delta)

        if save:
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        total = len(self.metrics_config)
        if self.stop_mode == 'and':
            stop = stopped_count == total
        elif self.stop_mode == 'or':
            stop = stopped_count > 0
        else:
            raise ValueError("stop_mode must be 'and' or 'or'")
        return stop

    def restore_best_model(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

    def state_dict(self):
        return {'best_values': self.best_values, 'counters': self.counters, 'best_state_dict': self.best_state_dict}

    def load_state_dict(self, state):
        self.best_values = state['best_values']
        self.counters = state['counters']
        self.best_state_dict = state.get('best_state_dict')
