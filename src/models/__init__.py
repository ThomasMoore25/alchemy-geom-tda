"""src.models — экспорты моделей (v32+).

Удобный импорт:
    from src.models import EGNNModel, EGNNTDA, EGNNVectorModel, EGNNVectorTDA
    from src.models import EGNNTensorModel, EGNNTensorTDA  # часть B
    from src.models import FCNNBaseline, SchNetWrapper
    from src.models import build_fcnn, build_schnet, build_egnn, build_egnn_tda
    from src.models import build_egnn_vector, build_egnn_vector_tda
    from src.models import build_egnn_tensor, build_egnn_tensor_tda
"""

from .egnn import EGNNModel, build_egnn
from .egnn_tda import EGNNTDA, build_egnn_tda
from .egnn_tensor import EGNNTensorModel, build_egnn_tensor
from .egnn_tensor_tda import EGNNTensorTDA, build_egnn_tensor_tda
from .egnn_vector import EGNNVectorModel, build_egnn_vector
from .egnn_vector_tda import EGNNVectorTDA, build_egnn_vector_tda
from .fcnn import FCNNBaseline, build_fcnn
from .knn import knn_graph_pytorch
from .schnet import SchNetWrapper, build_schnet

__all__ = [
    # Classes
    "FCNNBaseline", "SchNetWrapper",
    "EGNNModel", "EGNNTDA", "EGNNVectorModel", "EGNNVectorTDA",
    "EGNNTensorModel", "EGNNTensorTDA",
    # Builders
    "build_fcnn", "build_schnet",
    "build_egnn", "build_egnn_tda",
    "build_egnn_vector", "build_egnn_vector_tda",
    "build_egnn_tensor", "build_egnn_tensor_tda",
    # Utilities
    "knn_graph_pytorch",
]
