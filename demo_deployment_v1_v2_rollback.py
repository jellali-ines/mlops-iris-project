"""
Simple script to demonstrate v1→v2→rollback deployment
"""
import time
import os

def deploy_version(version):
    """Simulates a version deployment"""
    print(f"\n{'='*50}")
    print(f"🚀 Déploiement de la version {version}")
    print(f"{'='*50}")
    
    os.environ['MODEL_VERSION'] = f'v{version}'
    print(f"✅ Variable MODEL_VERSION set: v{version}")
    
    model_path = f"models/model_v{version}.pkl"
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"✅ Fichier modèle trouvé: {model_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        return False

def main():
    print("="*70)
    print("🎭 DEMONSTRATION: v1 → v2 → Rollback → v2 Deployment")
    print("="*70)
    
    # Test 1: Deploy v1 (baseline)
    print("\n📦 Step 1: Deploying version 1 (Baseline - Stable)")
    if deploy_version("1"):
        print("✅ Version 1 deployed successfully")
    time.sleep(1)
    
    # Test 2: Deploy optuna_best (The real upgrade)
    print("\n📦 Step 2: Deploying OPTIMIZED model (Optuna - 96.7%)")
    os.environ['MODEL_VERSION'] = 'optuna_best'
    model_path = "models/model_optuna_best.pkl"
    if os.path.exists(model_path):
        print(f"✅ Version OPTUNA_BEST deployed successfully")
    time.sleep(1)
    
    # Test 3: Rollback to v1
    print("\n⏪ Step 3: Emergency ROLLBACK to v1")
    if deploy_version("1"):
        print("✅ Rollback successful to v1 (Baseline)")
    time.sleep(1)
    
    # Test 4: Re-deploy optuna_best
    print("\n🔄 Step 4: Re-deploying Final Version (Optuna)")
    os.environ['MODEL_VERSION'] = 'optuna_best'
    print("✅ Final Version (Optuna) re-deployed")
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\n📋 Available Versions:")
    for v in [1, 2, 3]:
        path = f"models/model_v{v}.pkl"
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✅ v{v}: {path} ({size_kb:.1f} KB)")
    
    print("\n💡 For real Docker deployment:")
    print("   export MODEL_VERSION=v2")
    print("   docker-compose up -d")

if __name__ == "__main__":
    main()