"""
Script simple pour démontrer le déploiement v1→v2→rollback
"""
import time
import os

def deploy_version(version):
    """Simule le déploiement d'une version"""
    print(f"\n{'='*50}")
    print(f"🚀 Déploiement de la version {version}")
    print(f"{'='*50}")
    
    os.environ['MODEL_VERSION'] = f'v{version}'
    print(f"✅ Variable MODEL_VERSION définie: v{version}")
    
    model_path = f"models/model_v{version}.pkl"
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"✅ Fichier modèle trouvé: {model_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ Fichier modèle introuvable: {model_path}")
        return False

def main():
    print("="*70)
    print("🎭 DÉMONSTRATION: Déploiement v1 → v2 → Rollback → v2")
    print("="*70)
    
    # Test 1: Deploy v1 (baseline)
    print("\n📦 Étape 1: Déploiement version 1 (Baseline - 93.33%)")
    if deploy_version(1):
        print("✅ Version 1 déployée avec succès")
    time.sleep(1)
    
    # Test 2: Deploy v2 (optimized)
    print("\n📦 Étape 2: Déploiement version 2 (Optimisée - 100%)")
    if deploy_version(2):
        print("✅ Version 2 déployée avec succès")
    time.sleep(1)
    
    # Test 3: Rollback to v1
    print("\n⏪ Étape 3: ROLLBACK vers version 1")
    if deploy_version(1):
        print("✅ Rollback réussi vers v1")
    time.sleep(1)
    
    # Test 4: Re-deploy v2
    print("\n🔄 Étape 4: Re-déploiement version 2")
    if deploy_version(2):
        print("✅ Version 2 re-déployée")
    
    print("\n" + "="*70)
    print("✅ DÉMONSTRATION TERMINÉE AVEC SUCCÈS")
    print("="*70)
    print("\n📋 Versions disponibles:")
    for v in [1, 2, 3]:
        path = f"models/model_v{v}.pkl"
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✅ v{v}: {path} ({size_kb:.1f} KB)")
    
    print("\n💡 Pour déploiement Docker réel:")
    print("   export MODEL_VERSION=v2")
    print("   docker-compose up -d")

if __name__ == "__main__":
    main()