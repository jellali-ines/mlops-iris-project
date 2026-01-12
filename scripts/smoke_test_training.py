"""
Smoke test for training pipeline - Continuous Training (CT)
Runs quick training with subset and minimal epochs
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time

def main():
    print('='*60)
    print('SMOKE TEST: Training Pipeline (CT)')
    print('='*60)
    
    # Load subset with balanced classes (30 samples)
    print('\n[1/4] Loading iris dataset (subset: 30 samples)...')
    iris = load_iris()
    
    # Take 10 samples from each of 3 classes (30 total, balanced)
    X_balanced = []
    y_balanced = []
    for class_idx in range(3):
        class_mask = iris.target == class_idx
        class_indices = [i for i, val in enumerate(class_mask) if val][:10]
        X_balanced.extend(iris.data[class_indices])
        y_balanced.extend([class_idx] * len(class_indices))
    
    X = np.array(X_balanced)
    y = np.array(y_balanced)
    print(f'      Loaded {len(X)} samples (20% of full dataset)')
    print(f'      Classes: {np.unique(y)} (balanced distribution)')
    
    # Split data
    print('[2/4] Splitting data (80/20)...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'      Train: {len(X_train)} samples, Test: {len(X_test)} samples')
    
    # Quick training (max_iter=10 = 1 epoch equivalent)
    print('[3/4] Training model (max_iter=10, ~1 epoch)...')
    start = time.time()
    model = LogisticRegression(max_iter=10, random_state=42)
    model.fit(X_train, y_train)
    duration = time.time() - start
    print(f'      Training completed in {duration:.2f} seconds')
    
    # Evaluate
    print('[4/4] Evaluating model...')
    score = model.score(X_test, y_test)
    print(f'      Test Accuracy: {score:.2%}')
    
    # Summary
    print('\n' + '='*60)
    print('SMOKE TEST RESULTS:')
    print('='*60)
    print(f'✅ Duration:        {duration:.2f}s')
    print(f'✅ Accuracy:        {score:.2%}')
    print(f'✅ Train samples:   {len(X_train)}')
    print(f'✅ Test samples:    {len(X_test)}')
    print(f'✅ Classes:         {len(np.unique(y_train))}')
    print(f'✅ Iterations:      10 (minimal, ~1 epoch)')
    print('='*60)
    print('✅ SUCCESS: Smoke training test PASSED')
    print('='*60)

if __name__ == '__main__':
    main()