"""
Service d'orchestration des évaluations LLM
Centralise la logique d'évaluation, comparaison et cache des résultats
"""

import hashlib
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from response_evaluator import ResponseEvaluator, EvaluationResult, display_evaluation_results, display_comparative_dashboard


@dataclass
class EvaluationRequest:
    """Représente une demande d'évaluation"""
    prompt: str
    response: str
    model_name: str
    
    def get_cache_key(self) -> str:
        """Génère une clé de cache unique pour cette demande"""
        content = f"{self.prompt}|{self.response}|{self.model_name}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()


@dataclass
class EvaluationSession:
    """Représente une session d'évaluation comparative"""
    session_id: str
    evaluations: Dict[str, EvaluationResult]
    requests: Dict[str, EvaluationRequest]
    created_at: datetime
    
    def add_evaluation(self, model_name: str, request: EvaluationRequest, evaluation: EvaluationResult):
        """Ajoute une évaluation à la session"""
        self.evaluations[model_name] = evaluation
        self.requests[model_name] = request
    
    def get_model_count(self) -> int:
        """Retourne le nombre de modèles évalués"""
        return len(self.evaluations)
    
    def is_ready_for_comparison(self) -> bool:
        """Vérifie si la session est prête pour comparaison"""
        return len(self.evaluations) >= 2


class EvaluationService:
    """Service d'orchestration des évaluations avec cache et sessions"""
    
    def __init__(self, response_evaluator: Optional[ResponseEvaluator] = None):
        """
        Initialise le service d'évaluation
        
        Args:
            response_evaluator: Instance du ResponseEvaluator (créée si None)
        """
        self.evaluator = response_evaluator or ResponseEvaluator()
        self._cache: Dict[str, EvaluationResult] = {}
        self._sessions: Dict[str, EvaluationSession] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    # === ÉVALUATIONS INDIVIDUELLES ===
    
    def evaluate_single_response(self, request: EvaluationRequest, use_cache: bool = True) -> EvaluationResult:
        """
        Évalue une seule réponse avec système de cache
        
        Args:
            request: Demande d'évaluation
            use_cache: Utiliser le cache si disponible
            
        Returns:
            EvaluationResult: Résultat de l'évaluation
        """
        cache_key = request.get_cache_key()
        
        # Vérifier le cache si activé
        if use_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        # Évaluer avec le ResponseEvaluator
        evaluation = self.evaluator.evaluate_response(
            request.prompt, 
            request.response, 
            request.model_name
        )
        
        # Stocker en cache
        if use_cache:
            self._cache[cache_key] = evaluation
        
        self._cache_misses += 1
        return evaluation
    
    def display_single_evaluation(self, evaluation: EvaluationResult, model_name: str):
        """
        Affiche une évaluation individuelle dans Streamlit
        
        Args:
            evaluation: Résultat d'évaluation
            model_name: Nom du modèle (version courte)
        """
        display_evaluation_results(evaluation, model_name)
    
    # === SESSIONS D'ÉVALUATION COMPARATIVE ===
    
    def create_evaluation_session(self, session_id: Optional[str] = None) -> str:
        """
        Crée une nouvelle session d'évaluation comparative
        
        Args:
            session_id: ID personnalisé (généré automatiquement si None)
            
        Returns:
            str: ID de la session créée
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self._sessions[session_id] = EvaluationSession(
            session_id=session_id,
            evaluations={},
            requests={},
            created_at=datetime.now()
        )
        
        return session_id
    
    def add_to_session(self, session_id: str, request: EvaluationRequest, 
                      display_individual: bool = True) -> bool:
        """
        Ajoute une évaluation à une session existante
        
        Args:
            session_id: ID de la session
            request: Demande d'évaluation
            display_individual: Afficher l'évaluation individuelle
            
        Returns:
            bool: True si ajouté avec succès
        """
        if session_id not in self._sessions:
            st.error(f"❌ Session {session_id} non trouvée")
            return False
        
        session = self._sessions[session_id]
        
        # Évaluer la réponse
        evaluation = self.evaluate_single_response(request)
        
        # Ajouter à la session
        session.add_evaluation(request.model_name, request, evaluation)
        
        # Afficher l'évaluation individuelle si demandé
        if display_individual:
            model_short = request.model_name.split(':')[0].strip()
            self.display_single_evaluation(evaluation, model_short)
        
        return True
    
    def get_session(self, session_id: str) -> Optional[EvaluationSession]:
        """
        Récupère une session par son ID
        
        Args:
            session_id: ID de la session
            
        Returns:
            EvaluationSession ou None si non trouvée
        """
        return self._sessions.get(session_id)
    
    def get_session_summary(self, session_id: str) -> Dict[str, any]:
        """
        Génère un résumé de session
        
        Args:
            session_id: ID de la session
            
        Returns:
            Dict avec les statistiques de session
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        
        evaluations = list(session.evaluations.values())
        
        return {
            'session_id': session_id,
            'model_count': session.get_model_count(),
            'created_at': session.created_at.isoformat(),
            'average_score': sum(e.overall_score for e in evaluations) / len(evaluations) if evaluations else 0,
            'best_model': max(session.evaluations.items(), key=lambda x: x[1].overall_score)[0] if evaluations else None,
            'ready_for_comparison': session.is_ready_for_comparison()
        }
    
    # === COMPARAISONS ET RAPPORTS ===
    
    def generate_comparison_report(self, session_id: str) -> Optional[Dict[str, any]]:
        """
        Génère un rapport comparatif pour une session
        
        Args:
            session_id: ID de la session
            
        Returns:
            Dict avec le rapport comparatif ou None si erreur
        """
        session = self.get_session(session_id)
        if not session:
            st.error(f"❌ Session {session_id} non trouvée")
            return None
        
        if not session.is_ready_for_comparison():
            st.warning("⚠️ Au moins 2 évaluations nécessaires pour la comparaison")
            return None
        
        # Préparer les données pour le ResponseEvaluator
        evaluations_by_model = {
            model_name: [evaluation] 
            for model_name, evaluation in session.evaluations.items()
        }
        
        # Générer le rapport via ResponseEvaluator
        return self.evaluator.generate_comparative_report(evaluations_by_model)
    
    def display_comparison_dashboard(self, session_id: str):
        """
        Affiche le dashboard comparatif pour une session
        
        Args:
            session_id: ID de la session
        """
        report = self.generate_comparison_report(session_id)
        if report:
            display_comparative_dashboard(report)
    
    def display_session_comparison_analysis(self, session_id: str):
        """
        Affiche l'analyse comparative complète d'une session
        
        Args:
            session_id: ID de la session
        """
        session = self.get_session(session_id)
        if not session or not session.is_ready_for_comparison():
            return
        
        st.markdown("---")
        st.markdown("## 🏆 Analyse Comparative Avancée")
        
        # Tableau comparatif
        comparison_data = []
        for model_name, evaluation in session.evaluations.items():
            comparison_data.append({
                'Modèle': model_name.split(':')[0].strip(),
                'Score Global': evaluation.overall_score,
                '📖 Lisibilité': evaluation.readability_score,
                '🏗️ Structure': evaluation.structure_score,
                '🔗 Sources': evaluation.sources_score,
                '📋 Complétude': evaluation.completeness_score,
                '🎯 Pertinence': evaluation.relevance_score,
                '📊 Mots': evaluation.details.get('word_count', 0)
            })
        
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Score Global', ascending=False)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Graphique radar si pas trop de modèles
        if len(comparison_data) <= 4:
            self._display_radar_chart(session.evaluations)
        
        # Analyse du meilleur modèle
        self._display_best_model_analysis(session.evaluations)
        
        # Dashboard comparatif complet
        self.display_comparison_dashboard(session_id)
    
    def _display_radar_chart(self, evaluations: Dict[str, EvaluationResult]):
        """Affiche le graphique radar comparatif"""
        import plotly.graph_objects as go
        
        categories = ['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        fig = go.Figure()
        
        for i, (model_name, evaluation) in enumerate(evaluations.items()):
            values = [
                evaluation.readability_score,
                evaluation.structure_score,
                evaluation.sources_score,
                evaluation.completeness_score,
                evaluation.relevance_score
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model_name.split(':')[0].strip(),
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            title="Comparaison Radar des Performances",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_best_model_analysis(self, evaluations: Dict[str, EvaluationResult]):
        """Affiche l'analyse du meilleur modèle"""
        best_model_name, best_evaluation = max(evaluations.items(), key=lambda x: x[1].overall_score)
        
        st.markdown(f"### 🏆 Meilleur Modèle: {best_model_name.split(':')[0].strip()}")
        st.markdown(f"**Score: {best_evaluation.overall_score}/10**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Points forts:**")
            strengths = []
            if best_evaluation.readability_score >= 8:
                strengths.append("📖 Excellente lisibilité")
            if best_evaluation.structure_score >= 8:
                strengths.append("🏗️ Très bien structuré")
            if best_evaluation.sources_score >= 8:
                strengths.append("🔗 Sources de qualité")
            if best_evaluation.completeness_score >= 8:
                strengths.append("📋 Réponse complète")
            if best_evaluation.relevance_score >= 8:
                strengths.append("🎯 Très pertinent")
            
            for strength in strengths:
                st.write(f"• {strength}")
            
            if not strengths:
                st.write("• Scores corrects sans excellence particulière")
        
        with col2:
            st.markdown("**⚠️ Points d'amélioration:**")
            weaknesses = []
            if best_evaluation.readability_score < 6:
                weaknesses.append("📖 Lisibilité à améliorer")
            if best_evaluation.structure_score < 6:
                weaknesses.append("🏗️ Structure à revoir")
            if best_evaluation.sources_score < 6:
                weaknesses.append("🔗 Manque de sources")
            if best_evaluation.completeness_score < 6:
                weaknesses.append("📋 Réponse incomplète")
            if best_evaluation.relevance_score < 6:
                weaknesses.append("🎯 Pertinence à améliorer")
            
            if not weaknesses:
                st.write("✅ Aucun point faible majeur identifié")
            else:
                for weakness in weaknesses:
                    st.write(f"• {weakness}")
    
    # === RECONSTRUCTION DEPUIS HISTORIQUE ===
    
    def reconstruct_evaluations_from_history(self, history_data: List[dict]) -> Dict[str, List[EvaluationResult]]:
        """
        Reconstruit les évaluations depuis les données d'historique
        
        Args:
            history_data: Données d'historique avec scores
            
        Returns:
            Dict[str, List[EvaluationResult]]: Évaluations par modèle
        """
        evaluations_by_model = {}
        
        for entry in history_data:
            model_name = entry.get('model_name', 'Unknown')
            
            # Reconstruire l'EvaluationResult depuis les données stockées
            eval_result = EvaluationResult(
                overall_score=entry.get('evaluation_score', 0),
                readability_score=entry.get('readability_score', 0),
                structure_score=entry.get('structure_score', 0),
                sources_score=entry.get('sources_score', 0),
                completeness_score=entry.get('completeness_score', 0),
                relevance_score=entry.get('relevance_score', 0),
                details={'word_count': len(entry.get('response', '').split())},
                recommendations=[]
            )
            
            if model_name not in evaluations_by_model:
                evaluations_by_model[model_name] = []
            
            evaluations_by_model[model_name].append(eval_result)
        
        return evaluations_by_model
    
    def generate_historical_report(self, history_data: List[dict]) -> Dict[str, any]:
        """
        Génère un rapport comparatif depuis l'historique
        
        Args:
            history_data: Données d'historique
            
        Returns:
            Dict avec le rapport historique
        """
        if not history_data:
            return {}
        
        evaluations_by_model = self.reconstruct_evaluations_from_history(history_data)
        return self.evaluator.generate_comparative_report(evaluations_by_model)
    
    def display_historical_dashboard(self, history_data: List[dict]):
        """
        Affiche le dashboard pour données historiques
        
        Args:
            history_data: Données d'historique
        """
        report = self.generate_historical_report(history_data)
        if report:
            display_comparative_dashboard(report)
    
    # === CACHE ET STATISTIQUES ===
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Retourne les statistiques du cache
        
        Returns:
            Dict avec les stats de cache
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 1),
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Vide le cache des évaluations"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_sessions(self):
        """Vide toutes les sessions d'évaluation"""
        self._sessions.clear()
    
    def display_cache_info(self):
        """Affiche les informations de cache dans Streamlit"""
        stats = self.get_cache_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🗄️ Cache Size", stats['cache_size'])
        with col2:
            st.metric("✅ Cache Hits", stats['cache_hits'])
        with col3:
            st.metric("❌ Cache Misses", stats['cache_misses'])
        with col4:
            st.metric("📊 Hit Rate", f"{stats['hit_rate_percent']}%")


# Instance globale du service (singleton pattern)
_evaluation_service = None


def get_evaluation_service() -> EvaluationService:
    """
    Retourne l'instance singleton du EvaluationService
    
    Returns:
        EvaluationService: Instance du service
    """
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service