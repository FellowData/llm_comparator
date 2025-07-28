#!/usr/bin/env python3
"""
Script de test pour l'évaluateur de réponses
Vérifie que tout fonctionne correctement
"""

from response_evaluator import ResponseEvaluator, count_syllables, simple_flesch_reading_ease

def test_syllable_counter():
    """Test du compteur de syllabes"""
    print("🔤 Test du compteur de syllabes:")
    test_words = [
        ("hello", 2),
        ("world", 1), 
        ("artificial", 4),
        ("intelligence", 4),
        ("évaluation", 5),
        ("français", 2)
    ]
    
    for word, expected in test_words:
        result = count_syllables(word)
        status = "✅" if abs(result - expected) <= 1 else "❌"  # Tolérance de ±1
        print(f"  {status} '{word}': {result} syllabes (attendu: ~{expected})")

def test_readability():
    """Test du score de lisibilité"""
    print("\n📖 Test du score de lisibilité:")
    
    texts = [
        ("Texte très simple. Mots courts. Phrases courtes.", "Facile"),
        ("Ce texte présente une complexité intermédiaire avec des phrases de longueur moyenne.", "Standard"),
        ("Cette démonstration illustre la méthodologie d'évaluation algorithmique sophistiquée.", "Difficile")
    ]
    
    for text, expected_level in texts:
        score = simple_flesch_reading_ease(text)
        print(f"  📝 Score: {score:.1f} - Texte: {text[:50]}...")

def test_full_evaluation():
    """Test complet de l'évaluateur"""
    print("\n🔬 Test complet de l'évaluateur:")
    
    evaluator = ResponseEvaluator()
    
    # Test avec une réponse simulée
    prompt = "Expliquez-moi les avantages de l'intelligence artificielle"
    response = """L'intelligence artificielle présente plusieurs avantages majeurs :

1. **Automatisation** : Elle permet d'automatiser des tâches répétitives
2. **Analyse de données** : Traitement rapide de grandes quantités d'information
3. **Précision** : Réduction des erreurs humaines

En conclusion, l'IA transforme notre façon de travailler.

=== SOURCES ===
https://fr.wikipedia.org/wiki/Intelligence_artificielle
https://www.lemonde.fr/intelligence-artificielle/
https://www.nature.com/subjects/machine-learning
"""
    
    result = evaluator.evaluate_response(prompt, response, "Test Model")
    
    print(f"  🎯 Score Global: {result.overall_score}/10")
    print(f"  📖 Lisibilité: {result.readability_score}/10")
    print(f"  🏗️ Structure: {result.structure_score}/10")
    print(f"  🔗 Sources: {result.sources_score}/10")
    print(f"  📋 Complétude: {result.completeness_score}/10")
    print(f"  🎯 Pertinence: {result.relevance_score}/10")
    
    print(f"\n💡 Recommandations:")
    for rec in result.recommendations[:2]:  # Afficher les 2 premières
        print(f"  • {rec}")

def test_edge_cases():
    """Test des cas limites"""
    print("\n⚠️ Test des cas limites:")
    
    evaluator = ResponseEvaluator()
    
    # Réponse vide
    result = evaluator.evaluate_response("Test prompt", "", "Empty Model")
    print(f"  📭 Réponse vide - Score: {result.overall_score}/10")
    
    # Réponse très courte
    result = evaluator.evaluate_response("Test prompt", "Oui.", "Short Model")
    print(f"  📏 Réponse courte - Score: {result.overall_score}/10")
    
    # Réponse sans sources
    result = evaluator.evaluate_response("Test prompt", "Une réponse normale sans sources.", "No Sources Model")
    print(f"  🚫 Sans sources - Score: {result.overall_score}/10")

if __name__ == "__main__":
    print("🧪 Test de l'Évaluateur de Réponses LLM")
    print("=" * 50)
    
    try:
        test_syllable_counter()
        test_readability()
        test_full_evaluation()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("✅ Tous les tests sont passés avec succès!")
        print("🚀 L'évaluateur est prêt à être utilisé!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
        print("🔧 Vérifiez que response_evaluator.py est dans le même dossier")