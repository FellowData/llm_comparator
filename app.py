#!/usr/bin/env python3
"""
Point d'entrée principal de l'application LLM Comparator
Ce fichier remplace l'ancien app.py monolithique
"""

import sys
from ui.main_ui import main

if __name__ == "__main__":
    # Point d'entrée de l'application
    # Lance l'interface utilisateur principale

    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Application fermée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        print("🆘 Contactez le support technique")
        sys.exit(1)