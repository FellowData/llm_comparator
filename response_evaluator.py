"""
Service d'évaluation automatique des réponses LLM
Évalue la qualité des réponses selon plusieurs critères objectifs
Version complète avec fonctionnalités avancées
"""

import re
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import streamlit as st
import json

# Remplacement de textstat par une implémentation simple
def simple_flesch_reading_ease(text: str) -> float:
    """Calcul simplifié du score de lisibilité Flesch"""
    if not text.strip():
        return 0
    
    # Compter les phrases
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        sentences = 1
    
    # Compter les mots
    words = len(text.split())
    if words == 0:
        return 0
    
    # Compter les syllabes (approximation)
    syllables = 0
    for word in text.split():
        syllables += count_syllables(word)
    
    # Formule Flesch
    if syllables == 0:
        syllables = words  # Fallback
    
    score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
    return max(0, min(100, score))

def count_syllables(word: str) -> int:
    """Compte approximatif des syllabes dans un mot"""
    word = word.lower().strip('.,!?;:"()[]{}')
    if not word:
        return 1
    
    vowels = 'aeiouAEIOUÀÁÂÃÄÅàáâãäåÈÉÊËèéêëÌÍÎÏìíîïÒÓÔÕÖòóôõöÙÚÛÜùúûüÿ'
    syllables = 0
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            syllables += 1
        previous_was_vowel = is_vowel
    
    # Les mots se terminant par 'e' muet
    if word.endswith('e') and syllables > 1:
        syllables -= 1
    
    return max(1, syllables)

def simple_flesch_kincaid_grade(text: str) -> float:
    """Calcul simplifié du grade Flesch-Kincaid"""
    if not text.strip():
        return 0
    
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        sentences = 1
    
    words = len(text.split())
    if words == 0:
        return 0
    
    syllables = sum(count_syllables(word) for word in text.split())
    
    grade = (0.39 * words / sentences) + (11.8 * syllables / words) - 15.59
    return max(0, grade)

@dataclass
class EvaluationResult:
    """Résultat d'évaluation d'une réponse"""
    overall_score: float
    readability_score: float
    structure_score: float
    sources_score: float
    completeness_score: float
    relevance_score: float
    details: Dict[str, any]
    recommendations: List[str]

class ResponseEvaluator:
    """Évaluateur automatique de réponses LLM avec fonctionnalités avancées"""
    
    def __init__(self):
        self.weights = {
            'readability': 0.2,
            'structure': 0.2,
            'sources': 0.25,
            'completeness': 0.15,
            'relevance': 0.2
        }
    
    def evaluate_response(self, prompt: str, response: str, model_name: str = "") -> EvaluationResult:
        """Évalue une réponse selon tous les critères"""
        
        # Séparer la réponse principale des sources
        main_response, sources_text = self._extract_main_and_sources(response)
        
        # Calcul des scores individuels
        readability = self._evaluate_readability(main_response)
        structure = self._evaluate_structure(main_response)
        sources = self._evaluate_sources(sources_text)
        completeness = self._evaluate_completeness(prompt, main_response)
        relevance = self._evaluate_relevance(prompt, main_response)
        
        # Score global pondéré
        overall_score = (
            readability['score'] * self.weights['readability'] +
            structure['score'] * self.weights['structure'] +
            sources['score'] * self.weights['sources'] +
            completeness['score'] * self.weights['completeness'] +
            relevance['score'] * self.weights['relevance']
        )
        
        # Génération des recommandations
        recommendations = self._generate_recommendations(
            readability, structure, sources, completeness, relevance
        )
        
        return EvaluationResult(
            overall_score=round(overall_score, 1),
            readability_score=round(readability['score'], 1),
            structure_score=round(structure['score'], 1),
            sources_score=round(sources['score'], 1),
            completeness_score=round(completeness['score'], 1),
            relevance_score=round(relevance['score'], 1),
            details={
                'readability': readability,
                'structure': structure,
                'sources': sources,
                'completeness': completeness,
                'relevance': relevance,
                'word_count': len(main_response.split()),
                'char_count': len(main_response),
                'model_name': model_name
            },
            recommendations=recommendations
        )
    
    def evaluate_batch_responses(self, prompts_responses: List[Tuple[str, str, str]]) -> Dict[str, List[EvaluationResult]]:
        """Évalue plusieurs réponses en lot pour comparaison"""
        results = {}
        
        for prompt, response, model_name in prompts_responses:
            if model_name not in results:
                results[model_name] = []
            
            evaluation = self.evaluate_response(prompt, response, model_name)
            results[model_name].append(evaluation)
        
        return results

    def generate_comparative_report(self, evaluations: Dict[str, List[EvaluationResult]]) -> Dict[str, any]:
        """Génère un rapport comparatif détaillé"""
        report = {
            'summary': {},
            'rankings': {},
            'insights': [],
            'recommendations': []
        }
        
        # Calcul des moyennes par modèle
        for model_name, evals in evaluations.items():
            if not evals:
                continue
                
            avg_scores = {
                'overall': sum(e.overall_score for e in evals) / len(evals),
                'readability': sum(e.readability_score for e in evals) / len(evals),
                'structure': sum(e.structure_score for e in evals) / len(evals),
                'sources': sum(e.sources_score for e in evals) / len(evals),
                'completeness': sum(e.completeness_score for e in evals) / len(evals),
                'relevance': sum(e.relevance_score for e in evals) / len(evals)
            }
            
            report['summary'][model_name] = {
                'averages': avg_scores,
                'count': len(evals),
                'consistency': self._calculate_consistency(evals)
            }
        
        # Classement par critère
        for criterion in ['overall', 'readability', 'structure', 'sources', 'completeness', 'relevance']:
            ranked = sorted(
                report['summary'].items(),
                key=lambda x: x[1]['averages'][criterion],
                reverse=True
            )
            report['rankings'][criterion] = [model for model, _ in ranked]
        
        # Génération d'insights
        report['insights'] = self._generate_insights(report['summary'])
        report['recommendations'] = self._generate_comparative_recommendations(report['summary'])
        
        return report

    def _calculate_consistency(self, evaluations: List[EvaluationResult]) -> float:
        """Calcule la consistance des scores (inverse de l'écart-type)"""
        if len(evaluations) < 2:
            return 1.0
        
        scores = [e.overall_score for e in evaluations]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Normaliser la consistance sur 0-1 (1 = très consistant)
        return max(0, 1 - (std_dev / 5))

    def _generate_insights(self, summary: Dict[str, Dict]) -> List[str]:
        """Génère des insights automatiques"""
        insights = []
        
        if not summary:
            return insights
        
        # Trouver le meilleur modèle global
        best_model = max(summary.items(), key=lambda x: x[1]['averages']['overall'])
        insights.append(f"🏆 Meilleur modèle global: {best_model[0]} ({best_model[1]['averages']['overall']:.1f}/10)")
        
        # Identifier les spécialistes
        specialists = {}
        for criterion in ['readability', 'structure', 'sources', 'completeness', 'relevance']:
            best_for_criterion = max(summary.items(), key=lambda x: x[1]['averages'][criterion])
            specialists[criterion] = best_for_criterion[0]
        
        # Modèles spécialisés
        for criterion, model in specialists.items():
            if model != best_model[0]:
                score = summary[model]['averages'][criterion]
                insights.append(f"🎯 Spécialiste {criterion}: {model} ({score:.1f}/10)")
        
        # Consistance
        most_consistent = max(summary.items(), key=lambda x: x[1]['consistency'])
        if most_consistent[1]['consistency'] > 0.8:
            insights.append(f"⚖️ Plus consistant: {most_consistent[0]} (consistance: {most_consistent[1]['consistency']:.2f})")
        
        return insights

    def _generate_comparative_recommendations(self, summary: Dict[str, Dict]) -> List[str]:
        """Génère des recommandations basées sur la comparaison"""
        recommendations = []
        
        if len(summary) < 2:
            return recommendations
        
        # Recommandations par usage
        best_overall = max(summary.items(), key=lambda x: x[1]['averages']['overall'])
        best_sources = max(summary.items(), key=lambda x: x[1]['averages']['sources'])
        best_readability = max(summary.items(), key=lambda x: x[1]['averages']['readability'])
        
        recommendations.append(f"📋 Usage général: Privilégier {best_overall[0]}")
        
        if best_sources[0] != best_overall[0]:
            recommendations.append(f"🔗 Recherche documentée: Utiliser {best_sources[0]}")
        
        if best_readability[0] != best_overall[0]:
            recommendations.append(f"📖 Communication grand public: Choisir {best_readability[0]}")
        
        # Détection des modèles à éviter
        worst_model = min(summary.items(), key=lambda x: x[1]['averages']['overall'])
        if worst_model[1]['averages']['overall'] < 6:
            recommendations.append(f"⚠️ À éviter pour des tâches critiques: {worst_model[0]}")
        
        return recommendations
    
    def _extract_main_and_sources(self, response: str) -> Tuple[str, str]:
        """Sépare la réponse principale des sources"""
        if "=== SOURCES ===" in response:
            parts = response.split("=== SOURCES ===", 1)
            return parts[0].strip(), parts[1].strip()
        return response.strip(), ""
    
    def _evaluate_readability(self, text: str) -> Dict[str, any]:
        """Évalue la lisibilité du texte"""
        if not text.strip():
            return {'score': 0, 'level': 'Aucun texte', 'details': {}}
        
        try:
            # Score Flesch (0-100, plus haut = plus lisible)
            flesch_score = simple_flesch_reading_ease(text)
            # Grade Flesch-Kincaid (niveau scolaire)
            fk_grade = simple_flesch_kincaid_grade(text)
            
            # Normalisation du score Flesch sur 10
            if flesch_score >= 90:
                normalized_score = 10
                level = "Très facile"
            elif flesch_score >= 80:
                normalized_score = 8.5
                level = "Facile"
            elif flesch_score >= 70:
                normalized_score = 7.5
                level = "Assez facile"
            elif flesch_score >= 60:
                normalized_score = 6.5
                level = "Standard"
            elif flesch_score >= 50:
                normalized_score = 5.5
                level = "Assez difficile"
            elif flesch_score >= 30:
                normalized_score = 4
                level = "Difficile"
            else:
                normalized_score = 2
                level = "Très difficile"
            
            return {
                'score': normalized_score,
                'level': level,
                'details': {
                    'flesch_score': round(flesch_score, 1),
                    'grade_level': round(fk_grade, 1),
                    'avg_sentence_length': self._avg_sentence_length(text),
                    'avg_word_length': self._avg_word_length(text)
                }
            }
        except Exception as e:
            # Fallback simple si le calcul échoue
            return self._simple_readability(text)
    
    def _simple_readability(self, text: str) -> Dict[str, any]:
        """Calcul de lisibilité simplifié en cas d'échec"""
        sentences = len(re.findall(r'[.!?]+', text))
        words = len(text.split())
        avg_sentence_length = words / max(sentences, 1)
        avg_word_length = sum(len(word) for word in text.split()) / max(words, 1)
        
        # Score simple basé sur les moyennes
        if avg_sentence_length < 15 and avg_word_length < 5:
            score = 8
            level = "Facile"
        elif avg_sentence_length < 20 and avg_word_length < 6:
            score = 6
            level = "Standard"
        else:
            score = 4
            level = "Difficile"
        
        return {
            'score': score,
            'level': level,
            'details': {
                'avg_sentence_length': round(avg_sentence_length, 1),
                'avg_word_length': round(avg_word_length, 1)
            }
        }
    
    def _evaluate_structure(self, text: str) -> Dict[str, any]:
        """Évalue la structure et l'organisation du texte"""
        if not text.strip():
            return {'score': 0, 'issues': ['Aucun texte']}
        
        score = 5  # Score de base
        issues = []
        bonuses = []
        
        # Présence de paragraphes
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            score += 1
            bonuses.append("Texte structuré en paragraphes")
        else:
            issues.append("Manque de paragraphes")
        
        # Présence de listes ou énumérations
        if re.search(r'^\s*[-•*]\s', text, re.MULTILINE) or re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
            score += 1
            bonuses.append("Utilise des listes pour clarifier")
        
        # Présence de titres ou sous-sections
        if re.search(r'^#{1,6}\s', text, re.MULTILINE) or re.search(r'\*\*[^*]+\*\*', text):
            score += 1
            bonuses.append("Contient des titres/sous-sections")
        
        # Longueur appropriée
        word_count = len(text.split())
        if 50 <= word_count <= 500:
            score += 1
            bonuses.append("Longueur appropriée")
        elif word_count < 20:
            score -= 1
            issues.append("Réponse trop courte")
        elif word_count > 800:
            score -= 0.5
            issues.append("Réponse peut-être trop longue")
        
        # Présence d'introduction/conclusion
        if text.lower().startswith(('en ', 'pour ', 'dans ', 'il ', 'la ', 'le ', 'cette ', 'ce ')):
            score += 0.5
            bonuses.append("Introduction claire")
        
        if any(word in text.lower()[-100:] for word in ['conclusion', 'résumé', 'finalement', 'en résumé', 'pour conclure']):
            score += 0.5
            bonuses.append("Conclusion présente")
        
        return {
            'score': min(10, max(0, score)),
            'issues': issues,
            'bonuses': bonuses,
            'details': {
                'paragraph_count': len(paragraphs),
                'word_count': word_count,
                'has_lists': bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE)),
                'has_headers': bool(re.search(r'^#{1,6}\s', text, re.MULTILINE))
            }
        }
    
    def _evaluate_sources(self, sources_text: str) -> Dict[str, any]:
        """Évalue la qualité et pertinence des sources"""
        if not sources_text or sources_text == "*Aucune source identifiable fournie.*":
            return {
                'score': 0,
                'url_count': 0,
                'valid_urls': 0,
                'domain_diversity': 0,
                'issues': ['Aucune source fournie']
            }
        
        # Extraction des URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', sources_text)
        
        if not urls:
            return {
                'score': 1,
                'url_count': 0,
                'valid_urls': 0,
                'domain_diversity': 0,
                'issues': ['Sources mentionnées mais pas d\'URLs valides']
            }
        
        # Validation des URLs
        valid_urls = []
        domains = set()
        reliable_domains = {'wikipedia.org', 'gov', 'edu', 'nature.com', 'sciencedirect.com', 'pubmed.ncbi.nlm.nih.gov'}
        
        for url in urls:
            try:
                parsed = urlparse(url)
                if parsed.netloc:
                    valid_urls.append(url)
                    domains.add(parsed.netloc.lower())
            except:
                pass
        
        # Calcul du score
        score = 0
        issues = []
        bonuses = []
        
        # Nombre de sources
        if len(valid_urls) >= 3:
            score += 3
            bonuses.append(f"{len(valid_urls)} sources fournies")
        elif len(valid_urls) >= 1:
            score += len(valid_urls)
            bonuses.append(f"{len(valid_urls)} source(s) fournie(s)")
        
        # Diversité des domaines
        domain_diversity = len(domains) / max(len(valid_urls), 1)
        if domain_diversity > 0.7:
            score += 2
            bonuses.append("Bonne diversité des sources")
        elif domain_diversity > 0.5:
            score += 1
        else:
            issues.append("Sources trop concentrées sur peu de domaines")
        
        # Fiabilité des domaines
        reliable_count = sum(1 for domain in domains if any(rel in domain for rel in reliable_domains))
        if reliable_count > 0:
            score += min(2, reliable_count)
            bonuses.append(f"{reliable_count} source(s) fiable(s) identifiée(s)")
        
        return {
            'score': min(10, score),
            'url_count': len(urls),
            'valid_urls': len(valid_urls),
            'domain_diversity': round(domain_diversity, 2),
            'domains': list(domains),
            'issues': issues,
            'bonuses': bonuses
        }
    
    def _evaluate_completeness(self, prompt: str, response: str) -> Dict[str, any]:
        """Évalue si la réponse répond complètement à la question"""
        if not response.strip():
            return {'score': 0, 'coverage': 0, 'issues': ['Aucune réponse']}
        
        # Analyse simple basée sur les mots-clés du prompt
        prompt_keywords = set(re.findall(r'\b\w+\b', prompt.lower()))
        prompt_keywords = {word for word in prompt_keywords if len(word) > 3}  # Mots significatifs
        
        response_lower = response.lower()
        covered_keywords = sum(1 for keyword in prompt_keywords if keyword in response_lower)
        coverage = covered_keywords / max(len(prompt_keywords), 1)
        
        # Détection des questions dans le prompt
        question_indicators = ['comment', 'pourquoi', 'quoi', 'qui', 'quand', 'où', 'combien', '?']
        has_questions = any(indicator in prompt.lower() for indicator in question_indicators)
        
        score = 5  # Score de base
        issues = []
        bonuses = []
        
        # Score basé sur la couverture des mots-clés
        if coverage >= 0.7:
            score += 3
            bonuses.append("Couvre bien les éléments du prompt")
        elif coverage >= 0.5:
            score += 2
        elif coverage >= 0.3:
            score += 1
        else:
            issues.append("Ne couvre pas assez d'éléments du prompt")
        
        # Longueur proportionnelle à la complexité du prompt
        response_length = len(response.split())
        prompt_length = len(prompt.split())
        
        if prompt_length > 20 and response_length < 50:
            score -= 1
            issues.append("Réponse courte pour un prompt complexe")
        elif prompt_length < 10 and response_length > 200:
            score -= 0.5
            issues.append("Réponse peut-être trop détaillée")
        else:
            bonuses.append("Longueur appropriée au prompt")
        
        return {
            'score': min(10, max(0, score)),
            'coverage': round(coverage, 2),
            'covered_keywords': covered_keywords,
            'total_keywords': len(prompt_keywords),
            'issues': issues,
            'bonuses': bonuses
        }
    
    def _evaluate_relevance(self, prompt: str, response: str) -> Dict[str, any]:
        """Évalue la pertinence de la réponse par rapport au prompt"""
        if not response.strip():
            return {'score': 0, 'relevance_indicators': [], 'issues': ['Aucune réponse']}
        
        score = 5  # Score de base
        issues = []
        bonuses = []
        relevance_indicators = []
        
        # Détection du type de demande
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Demande d'explication
        if any(word in prompt_lower for word in ['expliquer', 'comment', 'pourquoi', 'qu\'est-ce']):
            if any(word in response_lower for word in ['parce que', 'car', 'en effet', 'cela signifie']):
                score += 1
                bonuses.append("Fournit des explications")
                relevance_indicators.append("explanatory")
        
        # Demande de liste/énumération
        if any(word in prompt_lower for word in ['liste', 'énumère', 'quels sont', 'exemples']):
            if re.search(r'^\s*[-•*]\s', response, re.MULTILINE) or re.search(r'^\s*\d+\.\s', response, re.MULTILINE):
                score += 1
                bonuses.append("Répond avec une structure de liste")
                relevance_indicators.append("list_format")
        
        # Demande d'analyse/comparaison
        if any(word in prompt_lower for word in ['analyser', 'comparer', 'différence', 'avantages', 'inconvénients']):
            if any(word in response_lower for word in ['d\'une part', 'd\'autre part', 'tandis que', 'en revanche', 'contrairement']):
                score += 1
                bonuses.append("Structure comparative appropriée")
                relevance_indicators.append("comparative")
        
        # Cohérence thématique simple (mots-clés communs)
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
        response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
        common_words = prompt_words.intersection(response_words)
        
        if len(common_words) >= 3:
            score += 1
            bonuses.append("Bonne cohérence thématique")
            relevance_indicators.append("thematic_coherence")
        elif len(common_words) < 1:
            score -= 1
            issues.append("Manque de cohérence thématique")
        
        # Détection de hors-sujet évident
        if len(response.split()) > 50 and len(common_words) == 0:
            score -= 2
            issues.append("Possible hors-sujet")
        
        return {
            'score': min(10, max(0, score)),
            'relevance_indicators': relevance_indicators,
            'common_keywords': len(common_words),
            'issues': issues,
            'bonuses': bonuses
        }
    
    def _generate_recommendations(self, readability: Dict, structure: Dict, 
                                sources: Dict, completeness: Dict, relevance: Dict) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        if readability['score'] < 6:
            recommendations.append("💡 Améliorer la lisibilité : utiliser des phrases plus courtes et un vocabulaire plus simple")
        
        if structure['score'] < 6:
            if 'Manque de paragraphes' in structure.get('issues', []):
                recommendations.append("📝 Structurer en paragraphes pour une meilleure organisation")
            recommendations.append("🔤 Ajouter des titres ou listes pour clarifier la structure")
        
        if sources['score'] < 5:
            recommendations.append("🔗 Fournir plus de sources fiables et diversifiées")
        elif sources['score'] < 8:
            recommendations.append("🎯 Diversifier les sources (éviter de se concentrer sur un seul domaine)")
        
        if completeness['score'] < 6:
            recommendations.append("📋 Réponse incomplète : couvrir plus d'aspects de la question")
        
        if relevance['score'] < 6:
            recommendations.append("🎯 Améliorer la pertinence : mieux répondre aux éléments spécifiques du prompt")
        
        if not recommendations:
            recommendations.append("✅ Excellente réponse ! Maintenir cette qualité")
        
        return recommendations
    
    def _avg_sentence_length(self, text: str) -> float:
        """Calcule la longueur moyenne des phrases"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        total_words = sum(len(sentence.split()) for sentence in sentences)
        return total_words / len(sentences)
    
    def _avg_word_length(self, text: str) -> float:
        """Calcule la longueur moyenne des mots"""
        words = text.split()
        if not words:
            return 0
        return sum(len(word.strip('.,!?;:')) for word in words) / len(words)

# Fonction d'aide pour l'interface Streamlit
def display_evaluation_results(evaluation: EvaluationResult, model_name: str = ""):
    """Affiche les résultats d'évaluation dans Streamlit"""
    
    # Header avec score global
    score_emoji = "⭐" * int(evaluation.overall_score // 2)
    st.markdown(f"### 📊 Évaluation {model_name}")
    st.markdown(f"**Score Global: {evaluation.overall_score}/10** {score_emoji}")
    
    # Graphique en barres des scores
    import plotly.graph_objects as go
    
    categories = ['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
    scores = [
        evaluation.readability_score,
        evaluation.structure_score,
        evaluation.sources_score,
        evaluation.completeness_score,
        evaluation.relevance_score
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=scores,
            marker_color=['#FF6B6B' if s < 5 else '#4ECDC4' if s < 8 else '#45B7D1' for s in scores],
            text=[f'{s}/10' for s in scores],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Détail des Scores - {model_name}",
        yaxis_title="Score (/10)",
        yaxis=dict(range=[0, 10]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Détails par catégorie
    with st.expander("📋 Détails de l'évaluation"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📖 Lisibilité:**")
            st.write(f"- Niveau: {evaluation.details['readability']['level']}")
            if 'flesch_score' in evaluation.details['readability']['details']:
                st.write(f"- Score Flesch: {evaluation.details['readability']['details']['flesch_score']}")
            
            st.markdown("**🏗️ Structure:**")
            structure_details = evaluation.details['structure']
            if structure_details.get('bonuses'):
                for bonus in structure_details['bonuses']:
                    st.write(f"✅ {bonus}")
            if structure_details.get('issues'):
                for issue in structure_details['issues']:
                    st.write(f"⚠️ {issue}")
        
        with col2:
            st.markdown("**🔗 Sources:**")
            sources_details = evaluation.details['sources']
            st.write(f"- URLs trouvées: {sources_details['valid_urls']}")
            st.write(f"- Diversité domaines: {sources_details['domain_diversity']}")
            
            st.markdown("**📊 Statistiques:**")
            st.write(f"- Mots: {evaluation.details['word_count']}")
            st.write(f"- Caractères: {evaluation.details['char_count']}")
    
    # Recommandations
    if evaluation.recommendations:
        st.markdown("### 💡 Recommandations d'amélioration:")
        for rec in evaluation.recommendations:
            st.write(f"• {rec}")

# Fonction pour l'affichage du dashboard comparatif
def display_comparative_dashboard(report: Dict[str, any]):
    """Affiche le dashboard comparatif dans Streamlit"""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    
    st.markdown("## 🏆 Rapport Comparatif des Modèles")
    
    # Métriques principales
    if report['summary']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_model = report['rankings']['overall'][0] if report['rankings']['overall'] else "N/A"
            best_score = report['summary'][best_model]['averages']['overall'] if best_model != "N/A" else 0
            st.metric("🥇 Meilleur Modèle", best_model, f"{best_score:.1f}/10")
        
        with col2:
            total_evaluations = sum(data['count'] for data in report['summary'].values())
            st.metric("📊 Total Évaluations", total_evaluations)
        
        with col3:
            avg_global = sum(data['averages']['overall'] for data in report['summary'].values()) / len(report['summary'])
            st.metric("📈 Score Moyen Global", f"{avg_global:.1f}/10")
    
    # Graphiques comparatifs
    tab1, tab2, tab3 = st.tabs(["📊 Scores Moyens", "🏅 Classements", "💡 Insights"])
    
    with tab1:
        if report['summary']:
            # Graphique en barres groupées
            data_for_chart = []
            for model, data in report['summary'].items():
                for criterion, score in data['averages'].items():
                    if criterion != 'overall':  # Exclure le score global pour éviter la redondance
                        data_for_chart.append({
                            'Modèle': model.split(':')[0] if ':' in model else model,
                            'Critère': criterion.title(),
                            'Score': score
                        })
            
            df_chart = pd.DataFrame(data_for_chart)
            
            fig = px.bar(
                df_chart,
                x='Modèle',
                y='Score',
                color='Critère',
                title='Scores Moyens par Critère et Modèle',
                barmode='group'
            )
            fig.update_layout(yaxis=dict(range=[0, 10]))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Classement Global")
            for i, model in enumerate(report['rankings']['overall'][:5], 1):
                score = report['summary'][model]['averages']['overall']
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1] if i <= 5 else f"{i}."
                model_short = model.split(':')[0] if ':' in model else model
                st.write(f"{medal} {model_short}: {score:.1f}/10")
        
        with col2:
            st.subheader("🎯 Spécialistes par Critère")
            criteria_labels = {
                'readability': '📖 Lisibilité',
                'structure': '🏗️ Structure',
                'sources': '🔗 Sources',
                'completeness': '📋 Complétude',
                'relevance': '🎯 Pertinence'
            }
            
            for criterion, label in criteria_labels.items():
                if criterion in report['rankings']:
                    best = report['rankings'][criterion][0]
                    score = report['summary'][best]['averages'][criterion]
                    best_short = best.split(':')[0] if ':' in best else best
                    st.write(f"{label}: **{best_short}** ({score:.1f}/10)")
    
    with tab3:
        st.subheader("💡 Insights Automatiques")
        for insight in report['insights']:
            st.write(f"• {insight}")
        
        st.subheader("🎯 Recommandations")
        for recommendation in report['recommendations']:
            st.write(f"• {recommendation}")
        
        # Graphique radar des top 3
        if len(report['summary']) >= 2:
            st.subheader("📡 Comparaison Radar - Top 3")
            
            top_3_models = report['rankings']['overall'][:3]
            
            fig = go.Figure()
            categories = ['Lisibilité', 'Structure', 'Sources', 'Complétude', 'Pertinence']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for i, model in enumerate(top_3_models):
                if model in report['summary']:
                    values = [
                        report['summary'][model]['averages']['readability'],
                        report['summary'][model]['averages']['structure'],
                        report['summary'][model]['averages']['sources'],
                        report['summary'][model]['averages']['completeness'],
                        report['summary'][model]['averages']['relevance']
                    ]
                    
                    model_short = model.split(':')[0] if ':' in model else model
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=model_short,
                        line_color=colors[i % len(colors)]
                    ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Fonction pour sauvegarder les rapports
def save_evaluation_report(report: Dict[str, any], filename: str):
    """Sauvegarde le rapport d'évaluation en JSON"""
    
    # S'assurer que le dossier existe
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    report_with_metadata = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'version': '1.0',
            'total_models': len(report['summary']),
            'total_evaluations': sum(data['count'] for data in report['summary'].values())
        },
        'report': report
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_with_metadata, f, ensure_ascii=False, indent=2)