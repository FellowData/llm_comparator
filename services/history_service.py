"""
Service de gestion de la persistance de l'historique des prompts et évaluations
Centralise toute la logique de sauvegarde/chargement (JSON local + Supabase)
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

HISTORY_FILE = "data/prompt_history.json"
SUPBASE_TABLE_NAME = "prompt_history"

@dataclass
class HistoryEntry:
    """Représente une entrée d'historique standardisée"""
    timestamp: str
    prompt: str
    model_name: str
    model_id: str
    response: str
    evaluation_score: float
    readability_score: float
    structure_score: float
    sources_score: float
    completeness_score: float
    relevance_score: float
    
    @classmethod
    def from_evaluation(cls, prompt: str, model_name: str, model_id: str, 
                       response: str, evaluation) -> 'HistoryEntry':
        """Crée une HistoryEntry depuis une évaluation"""
        return cls(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            model_name=model_name,
            model_id=model_id,
            response=response,
            evaluation_score=evaluation.overall_score,
            readability_score=evaluation.readability_score,
            structure_score=evaluation.structure_score,
            sources_score=evaluation.sources_score,
            completeness_score=evaluation.completeness_score,
            relevance_score=evaluation.relevance_score
        )
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour sauvegarde"""
        return asdict(self)


class HistoryService:
    """Service de gestion de la persistance de l'historique"""
    
    def __init__(self, history_file: str = HISTORY_FILE):
        """
        Initialise le service d'historique
        
        Args:
            history_file (str): Chemin vers le fichier JSON local
        """
        self.history_file = Path(history_file)
        self.table_name = SUPBASE_TABLE_NAME
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """S'assure que le répertoire de données existe"""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
    
    # === MÉTHODES DE SAUVEGARDE ===
    
    def save_evaluation_entry(self, prompt: str, model_name: str, model_id: str, 
                            response: str, evaluation) -> bool:
        """
        Sauvegarde une entrée avec évaluation (JSON + Supabase)
        
        Args:
            prompt (str): Le prompt utilisé
            model_name (str): Nom du modèle
            model_id (str): ID du modèle
            response (str): Réponse du modèle
            evaluation: Objet d'évaluation avec les scores
            
        Returns:
            bool: True si au moins une sauvegarde a réussi
        """
        try:
            # Créer l'entrée standardisée
            entry = HistoryEntry.from_evaluation(prompt, model_name, model_id, response, evaluation)
            entry_dict = entry.to_dict()
            
            # Tentatives de sauvegarde
            json_success = self._save_to_json(entry_dict)
            supabase_success = self._save_to_supabase(entry_dict)
            
            # Retourner True si au moins une sauvegarde a réussi
            return json_success or supabase_success
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la sauvegarde de l'entrée: {e}")
            return False
    
    def _save_to_json(self, entry: dict) -> bool:
        """
        Sauvegarde dans le fichier JSON local
        
        Args:
            entry (dict): Entrée à sauvegarder
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            # Charger les données existantes
            if self.history_file.exists():
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
            else:
                history_data = []
            
            # Ajouter la nouvelle entrée
            history_data.append(entry)
            
            # Sauvegarder
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Échec de sauvegarde JSON locale: {e}")
            return False
    
    def _save_to_supabase(self, entry: dict) -> bool:
        """
        Sauvegarde dans Supabase
        
        Args:
            entry (dict): Entrée à sauvegarder
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        try:
            from supabase_client import insert_llm_result
            insert_llm_result(self.table_name, entry)
            return True
            
        except ImportError:
            st.warning("⚠️ Module Supabase non disponible")
            return False
        except Exception as e:
            st.warning(f"⚠️ Échec de sauvegarde Supabase: {e}")
            return False
    
    # === MÉTHODES DE CHARGEMENT ===
    
    def load_history(self) -> List[dict]:
        """
        Charge l'historique (Supabase puis fallback JSON)
        
        Returns:
            List[dict]: Liste des entrées d'historique
        """
        history_data, _ = self.load_history_with_source()
        return history_data
    
    def load_history_with_source(self) -> Tuple[List[dict], str]:
        """
        Charge l'historique et retourne la source de données
        
        Returns:
            Tuple[List[dict], str]: (données, source)
        """
        # Tentative de chargement depuis Supabase
        supabase_data, supabase_success = self._load_from_supabase()
        
        if supabase_success:
            # Synchroniser le backup JSON
            sync_success = self._sync_json_backup(supabase_data)
            
            if sync_success:
                st.success(f"✅ Données chargées depuis Supabase ({len(supabase_data)} entrées) • Backup local synchronisé")
            else:
                st.success(f"✅ Données chargées depuis Supabase ({len(supabase_data)} entrées)")
                st.warning("⚠️ Échec de synchronisation du backup local")
            
            return supabase_data, "Supabase"
        
        # Fallback vers JSON local
        st.warning("⚠️ Échec du chargement Supabase")
        st.info("🔄 Basculement vers le fichier JSON local...")
        
        json_data, json_success = self._load_from_json()
        
        if json_success:
            st.success(f"✅ Données chargées depuis JSON local ({len(json_data)} entrées)")
            return json_data, "JSON Local"
        else:
            st.info("📝 Aucun fichier d'historique local trouvé")
            return [], "Aucune source"
    
    def _load_from_supabase(self) -> Tuple[List[dict], bool]:
        """
        Charge depuis Supabase
        
        Returns:
            Tuple[List[dict], bool]: (données, succès)
        """
        try:
            from supabase_client import get_llm_history
            
            with st.spinner("🔄 Chargement depuis Supabase..."):
                history_data = get_llm_history(self.table_name)
            
            return history_data, True
            
        except ImportError:
            return [], False
        except Exception as e:
            st.warning(f"⚠️ Échec du chargement Supabase: {e}")
            return [], False
    
    def _load_from_json(self) -> Tuple[List[dict], bool]:
        """
        Charge depuis JSON local
        
        Returns:
            Tuple[List[dict], bool]: (données, succès)
        """
        try:
            if not self.history_file.exists():
                return [], False
            
            with open(self.history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            
            return history_data, True
            
        except Exception as e:
            st.error(f"❌ Échec du chargement du fichier JSON: {e}")
            return [], False
    
    def _sync_json_backup(self, data: List[dict]) -> bool:
        """
        Synchronise le backup JSON local avec les données Supabase
        
        Args:
            data (List[dict]): Données à synchroniser
            
        Returns:
            bool: True si la synchronisation a réussi
        """
        try:
            self._ensure_directory_exists()
            
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Échec de synchronisation du backup JSON: {e}")
            return False
    
    # === MÉTHODES D'EXPORT ===
    
    def export_to_excel(self, data: List[dict]) -> Optional[bytes]:
        """
        Exporte les données vers Excel et retourne les bytes
        
        Args:
            data (List[dict]): Données à exporter
            
        Returns:
            Optional[bytes]: Données Excel ou None si échec
        """
        try:
            if not data:
                st.warning("⚠️ Aucune donnée à exporter")
                return None
            
            # Préparer le DataFrame
            df = self.prepare_dataframe(data)
            
            # Générer le fichier Excel
            excel_file = f"data/prompt_history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            excel_path = Path(excel_file)
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_excel(excel_path, index=False)
            
            # Lire et retourner les bytes
            with open(excel_path, "rb") as f:
                excel_bytes = f.read()
            
            # Nettoyer le fichier temporaire
            excel_path.unlink()
            
            return excel_bytes
            
        except Exception as e:
            st.error(f"❌ Échec de l'export Excel: {e}")
            return None
    
    def prepare_dataframe(self, data: List[dict]) -> pd.DataFrame:
        """
        Prépare un DataFrame pandas avec colonnes formatées
        
        Args:
            data (List[dict]): Données brutes
            
        Returns:
            pd.DataFrame: DataFrame formaté
        """
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Traitement des sources et réponses principales
        if 'response' in df.columns:
            df["sources"] = df["response"].apply(
                lambda r: r.split("=== SOURCES ===")[1].strip() 
                if isinstance(r, str) and "=== SOURCES ===" in r 
                else ""
            )
            df["main_response"] = df["response"].apply(
                lambda r: r.split("=== SOURCES ===")[0].strip() 
                if isinstance(r, str) and "=== SOURCES ===" in r 
                else str(r) if r else ""
            )
        
        # Formater les colonnes de scores
        score_columns = [
            'evaluation_score', 'readability_score', 'structure_score', 
            'sources_score', 'completeness_score', 'relevance_score'
        ]
        
        for col in score_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(1)
        
        # Formater la colonne timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Formater la colonne timestamp (CORRECTION TIMEZONE)
        if 'timestamp' in df.columns:
            # Convertir en datetime puis supprimer timezone info pour Excel
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Convertir en timezone naive (supprimer timezone info)
            if hasattr(df['timestamp'].dtype, 'tz') and df['timestamp'].dtype.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert(None)

            
        return df
    
    # === MÉTHODES DE MAINTENANCE ===
    
    def cleanup_old_entries(self, days: int = 90) -> int:
        """
        Nettoie les entrées anciennes du fichier JSON local
        
        Args:
            days (int): Nombre de jours à conserver
            
        Returns:
            int: Nombre d'entrées supprimées
        """
        try:
            data, success = self._load_from_json()
            
            if not success or not data:
                return 0
            
            # Calculer la date limite
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            # Filtrer les entrées récentes
            initial_count = len(data)
            filtered_data = []
            
            for entry in data:
                try:
                    entry_date = datetime.fromisoformat(entry['timestamp']).timestamp()
                    if entry_date >= cutoff_date:
                        filtered_data.append(entry)
                except (KeyError, ValueError):
                    # Garder les entrées avec timestamp invalide
                    filtered_data.append(entry)
            
            # Sauvegarder les données filtrées
            if len(filtered_data) < initial_count:
                with open(self.history_file, "w", encoding="utf-8") as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                
                removed_count = initial_count - len(filtered_data)
                st.info(f"🧹 Nettoyage terminé: {removed_count} entrée(s) supprimée(s)")
                return removed_count
            
            return 0
            
        except Exception as e:
            st.error(f"❌ Erreur lors du nettoyage: {e}")
            return 0
    
    def validate_data_integrity(self) -> bool:
        """
        Valide l'intégrité des données JSON locales
        
        Returns:
            bool: True si les données sont valides
        """
        try:
            data, success = self._load_from_json()
            
            if not success:
                return True  # Pas de fichier = pas d'erreur
            
            # Vérifier la structure de chaque entrée
            required_fields = [
                'timestamp', 'prompt', 'model_name', 'model_id', 'response'
            ]
            
            issues = []
            
            for i, entry in enumerate(data):
                if not isinstance(entry, dict):
                    issues.append(f"Entrée {i}: n'est pas un dictionnaire")
                    continue
                
                for field in required_fields:
                    if field not in entry:
                        issues.append(f"Entrée {i}: champ '{field}' manquant")
                
                # Vérifier le format du timestamp
                try:
                    datetime.fromisoformat(entry.get('timestamp', ''))
                except ValueError:
                    issues.append(f"Entrée {i}: timestamp invalide")
            
            if issues:
                st.warning(f"⚠️ Problèmes d'intégrité détectés:")
                for issue in issues[:10]:  # Limiter l'affichage
                    st.write(f"• {issue}")
                if len(issues) > 10:
                    st.write(f"• ... et {len(issues) - 10} autres problèmes")
                return False
            
            st.success(f"✅ Intégrité des données validée ({len(data)} entrées)")
            return True
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la validation: {e}")
            return False
    
    # === MÉTHODES UTILITAIRES ===
    
    def get_file_path(self) -> Path:
        """Retourne le chemin du fichier d'historique"""
        return self.history_file
    
    def get_file_size(self) -> str:
        """Retourne la taille du fichier d'historique formatée"""
        try:
            if self.history_file.exists():
                size_bytes = self.history_file.stat().st_size
                
                if size_bytes < 1024:
                    return f"{size_bytes} B"
                elif size_bytes < 1024**2:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024**2):.1f} MB"
            return "0 B"
            
        except Exception:
            return "Taille inconnue"


# Instance globale du service (singleton pattern)
_history_service = None


def get_history_service() -> HistoryService:
    """
    Retourne l'instance singleton du HistoryService
    
    Returns:
        HistoryService: Instance du service
    """
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service