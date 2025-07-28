"""
Service de gestion de la configuration des modèles LLM
Centralise toute la logique liée aux modèles et leur configuration
"""

import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Classe pour représenter les informations d'un modèle"""
    name: str
    id: str
    provider: str
    provider_to_display: str
    available: bool
    free: bool
    web_search: bool
    
    @classmethod
    def from_dict(cls, name: str, data: dict) -> 'ModelInfo':
        """Crée une instance ModelInfo depuis un dictionnaire"""
        return cls(
            name=name,
            id=data.get("id", ""),
            provider=data.get("provider", ""),
            provider_to_display=data.get("provider_to_display", "Unknown"),
            available=data.get("available", True),
            free=data.get("free", False),
            web_search=data.get("web_search", False)
        )
    
    def to_dict(self) -> dict:
        """Convertit l'instance en dictionnaire (pour compatibilité)"""
        return {
            "id": self.id,
            "provider": self.provider,
            "provider_to_display": self.provider_to_display,
            "available": self.available,
            "free": self.free,
            "web_search": self.web_search
        }


@dataclass
class ModelStats:
    """Statistiques sur les modèles"""
    total_models: int
    available_models: int
    free_models: int
    paid_models: int
    web_search_models: int
    providers: Set[str]
    
    def __str__(self) -> str:
        return (f"Total: {self.total_models}, "
                f"Disponibles: {self.available_models}, "
                f"Gratuits: {self.free_models}, "
                f"Payants: {self.paid_models}, "
                f"Web Search: {self.web_search_models}, "
                f"Providers: {len(self.providers)}")


@dataclass
class FilterCriteria:
    """Critères de filtrage des modèles"""
    pricing: Optional[str] = None  # "All", "Free Only", "Paid Only"
    web_search: Optional[str] = None  # "All", "With Web Search", "Without Web Search"
    provider: Optional[str] = None  # "All" ou nom du provider
    available_only: bool = True


class ModelConfigService:
    """Service de gestion de la configuration des modèles LLM"""
    
    def __init__(self, config_path: str = "config/llm_config.yaml"):
        self.config_path = Path(config_path)
        self._models_dict: Optional[Dict[str, ModelInfo]] = None
        self._raw_config: Optional[dict] = None
    
    def load_models_config(self) -> Dict[str, ModelInfo]:
        """
        Charge la configuration des modèles depuis le fichier YAML
        
        Returns:
            Dict[str, ModelInfo]: Dictionnaire des modèles configurés
            
        Raises:
            FileNotFoundError: Si le fichier de configuration n'existe pas
            yaml.YAMLError: Si le fichier YAML est mal formé
            ValueError: Si la configuration est invalide
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Fichier de configuration non trouvé: {self.config_path}")
        
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._raw_config = yaml.safe_load(f)
            
            if not self._raw_config or "models" not in self._raw_config:
                raise ValueError("Configuration invalide: section 'models' manquante")
            
            # Convertir en objets ModelInfo
            models_dict = {}
            for model_data in self._raw_config["models"]:
                if "name" not in model_data:
                    continue  # Ignorer les modèles sans nom
                
                name = model_data["name"]
                models_dict[name] = ModelInfo.from_dict(name, model_data)
            
            self._models_dict = models_dict
            return models_dict
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Erreur lors du chargement du YAML: {e}")
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement de la configuration: {e}")
    
    def get_models_dict(self) -> Dict[str, ModelInfo]:
        """
        Retourne le dictionnaire des modèles (charge si nécessaire)
        
        Returns:
            Dict[str, ModelInfo]: Dictionnaire des modèles
        """
        if self._models_dict is None:
            self.load_models_config()
        return self._models_dict
    
    def get_raw_models_dict(self) -> Dict[str, dict]:
        """
        Retourne le dictionnaire des modèles au format original (compatibilité)
        
        Returns:
            Dict[str, dict]: Dictionnaire des modèles au format original
        """
        models_dict = self.get_models_dict()
        return {name: model.to_dict() for name, model in models_dict.items()}
    
    def get_available_models(self) -> Dict[str, ModelInfo]:
        """
        Retourne uniquement les modèles disponibles
        
        Returns:
            Dict[str, ModelInfo]: Modèles disponibles
        """
        models_dict = self.get_models_dict()
        return {name: model for name, model in models_dict.items() if model.available}
    
    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """
        Retourne les modèles d'un provider spécifique
        
        Args:
            provider (str): Nom du provider
            
        Returns:
            List[ModelInfo]: Liste des modèles du provider
        """
        models_dict = self.get_models_dict()
        return [model for model in models_dict.values() 
                if model.provider_to_display.lower() == provider.lower()]
    
    def get_all_providers(self) -> Set[str]:
        """
        Retourne la liste de tous les providers configurés
        
        Returns:
            Set[str]: Ensemble des providers
        """
        models_dict = self.get_models_dict()
        return {model.provider_to_display for model in models_dict.values()}
    
    def validate_model_config(self, model_data: dict) -> bool:
        """
        Valide la configuration d'un modèle
        
        Args:
            model_data (dict): Données du modèle à valider
            
        Returns:
            bool: True si la configuration est valide
        """
        required_fields = ["name", "id", "provider"]
        return all(field in model_data for field in required_fields)
    
    def get_model_statistics(self) -> ModelStats:
        """
        Calcule les statistiques des modèles configurés
        
        Returns:
            ModelStats: Statistiques des modèles
        """
        models_dict = self.get_models_dict()
        
        total_models = len(models_dict)
        available_models = sum(1 for model in models_dict.values() if model.available)
        free_models = sum(1 for model in models_dict.values() if model.free)
        paid_models = sum(1 for model in models_dict.values() if not model.free)
        web_search_models = sum(1 for model in models_dict.values() if model.web_search)
        providers = self.get_all_providers()
        
        return ModelStats(
            total_models=total_models,
            available_models=available_models,
            free_models=free_models,
            paid_models=paid_models,
            web_search_models=web_search_models,
            providers=providers
        )
    
    def filter_models(self, criteria: FilterCriteria) -> Dict[str, ModelInfo]:
        """
        Filtre les modèles selon les critères spécifiés
        
        Args:
            criteria (FilterCriteria): Critères de filtrage
            
        Returns:
            Dict[str, ModelInfo]: Modèles filtrés
        """
        models_dict = self.get_models_dict()
        
        # Commencer avec tous les modèles ou seulement les disponibles
        if criteria.available_only:
            filtered_models = self.get_available_models()
        else:
            filtered_models = models_dict.copy()
        
        # Filtre par prix
        if criteria.pricing == "Free Only":
            filtered_models = {name: model for name, model in filtered_models.items() 
                             if model.free}
        elif criteria.pricing == "Paid Only":
            filtered_models = {name: model for name, model in filtered_models.items() 
                             if not model.free}
        
        # Filtre par web search
        if criteria.web_search == "With Web Search":
            filtered_models = {name: model for name, model in filtered_models.items() 
                             if model.web_search}
        elif criteria.web_search == "Without Web Search":
            filtered_models = {name: model for name, model in filtered_models.items() 
                             if not model.web_search}
        
        # Filtre par provider
        if criteria.provider and criteria.provider != "All":
            filtered_models = {name: model for name, model in filtered_models.items() 
                             if model.provider_to_display == criteria.provider}
        
        return filtered_models
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """
        Récupère un modèle par son nom
        
        Args:
            name (str): Nom du modèle
            
        Returns:
            Optional[ModelInfo]: Le modèle s'il existe, None sinon
        """
        models_dict = self.get_models_dict()
        return models_dict.get(name)
    
    def is_model_available(self, name: str) -> bool:
        """
        Vérifie si un modèle est disponible
        
        Args:
            name (str): Nom du modèle
            
        Returns:
            bool: True si le modèle est disponible
        """
        model = self.get_model_by_name(name)
        return model is not None and model.available
    
    def get_selection_stats(self, selected_models: List[str]) -> dict:
        """
        Génère les statistiques d'une sélection de modèles
        
        Args:
            selected_models (List[str]): Liste des noms de modèles sélectionnés
            
        Returns:
            dict: Statistiques de la sélection
        """
        if not selected_models:
            return {
                'count': 0,
                'free_count': 0,
                'paid_count': 0,
                'web_search_count': 0,
                'short_names': []
            }
        
        models_dict = self.get_models_dict()
        
        free_count = 0
        paid_count = 0
        web_search_count = 0
        short_names = []
        
        for name in selected_models:
            model = models_dict.get(name)
            if model:
                short_name = name.split(':')[0].strip()
                short_names.append(short_name)
                
                if model.free:
                    free_count += 1
                else:
                    paid_count += 1
                    
                if model.web_search:
                    web_search_count += 1
        
        return {
            'count': len(selected_models),
            'free_count': free_count,
            'paid_count': paid_count,
            'web_search_count': web_search_count,
            'short_names': short_names
        }
    
    def reload_config(self) -> Dict[str, ModelInfo]:
        """
        Recharge la configuration depuis le fichier
        
        Returns:
            Dict[str, ModelInfo]: Nouvelle configuration
        """
        self._models_dict = None
        self._raw_config = None
        return self.load_models_config()


# Instance globale du service (singleton pattern)
_model_config_service = None


def get_model_config_service() -> ModelConfigService:
    """
    Retourne l'instance singleton du ModelConfigService
    
    Returns:
        ModelConfigService: Instance du service
    """
    global _model_config_service
    if _model_config_service is None:
        _model_config_service = ModelConfigService()
    return _model_config_service