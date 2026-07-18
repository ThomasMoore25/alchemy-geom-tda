"""src — основной пакет (v32+).

Удобный импорт:
    from src.data import ATOM_TYPES, parse_sdf, mol_to_arrays
    from src.dataset import AlchemyDataset
    from src.metrics import mae, rmse, r2_score
    from src.utils import seed_everything, get_device
    from src.early_stopping import EarlyStopping
    from src.models import EGNNModel, build_egnn
    from src.tda import extract_tda_features, FiLMModulation
"""

__version__ = "33.0.0"
