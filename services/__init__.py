"""
Services de l'application LLM Comparator
Contient la logique métier centralisée

Services disponibles :
- ModelConfigService : Gestion de la configuration des modèles
- HistoryService : Persistance des données d'historique  
- EvaluationService : Orchestration des évaluations et comparaisons
"""

from .model_config_service import ModelConfigService, get_model_config_service
from .history_service import HistoryService, get_history_service
from .evaluation_service import EvaluationService, get_evaluation_service

__version__ = "1.1.0"
__all__ = [
    "ModelConfigService", "get_model_config_service",
    "HistoryService", "get_history_service", 
    "EvaluationService", "get_evaluation_service"
]