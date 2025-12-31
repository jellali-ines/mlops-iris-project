"""
ZenML Pipeline لتدريب نموذج Iris
End-to-end ML pipeline with ZenML
"""
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zenml import pipeline, step


# ============================================
# STEPS
# ============================================

@step
def data_loader(csv_path: str = 'data/raw/iris.csv') -> Tuple[np.ndarray, np.ndarray]:
    """
    تحميل بيانات Iris
    Load Iris dataset from CSV
    
    Returns:
        Features (X) and labels (y)
    """
    print("📊 تحميل البيانات من CSV...")
    
    df = pd.read_csv(csv_path)
    X = df.drop(['target', 'species'], axis=1).values
    y = df['target'].values
    
    print(f"✅ تم تحميل {X.shape[0]} عينة، {X.shape[1]} مميزات، {len(np.unique(y))} أصناف")
    
    return X, y


@step
def data_splitter(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    تقسيم البيانات إلى train و test
    Split data into train and test sets
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"🔪 تقسيم البيانات (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✅ Train: {X_train.shape[0]} عينة، Test: {X_test.shape[0]} عينة")
    
    return X_train, X_test, y_train, y_test


@step
def data_preprocessor(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    تطبيع البيانات باستخدام StandardScaler
    Preprocess data with standardization
    
    Returns:
        Scaled training features, scaled test features, fitted scaler
    """
    print("⚙️ تطبيع البيانات...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ تم تطبيع البيانات")
    
    return X_train_scaled, X_test_scaled, scaler


@step
def model_trainer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 10.0,
    max_iter: int = 200
) -> LogisticRegression:
    """
    تدريب نموذج Logistic Regression
    Train ML model
    
    Returns:
        Trained model
    """
    print(f"🤖 تدريب النموذج (C={C}, max_iter={max_iter})...")
    
    model = LogisticRegression(
        C=C,
        solver='saga',
        max_iter=max_iter,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("✅ تم تدريب النموذج")
    
    return model


@step
def model_evaluator(
    model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    تقييم النموذج
    Evaluate trained model
    
    Returns:
        Dictionary of metrics
    """
    print("📈 تقييم النموذج...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1)
    }
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    
    return metrics


@step
def model_exporter(
    model: LogisticRegression,
    scaler: StandardScaler,
    metrics: dict,
    output_dir: str = 'models',
    version: str = 'zenml_v1'
) -> Tuple[str, str]:
    """
    حفظ النموذج والـ scaler
    Export trained model and scaler
    
    Returns:
        Paths to saved model and scaler
    """
    print(f"💾 حفظ النموذج (version: {version})...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"model_{version}.pkl"
    scaler_path = output_dir / f"scaler_{version}.pkl"
    
    # حفظ النموذج
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # حفظ الـ scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ تم حفظ النموذج في: {model_path}")
    print(f"✅ تم حفظ الـ scaler في: {scaler_path}")
    print(f"📊 Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return str(model_path), str(scaler_path)


# ============================================
# PIPELINE
# ============================================

@pipeline(enable_cache=False)
def iris_training_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
    C: float = 10.0,
    max_iter: int = 200,
    version: str = 'zenml_v1'
):
    """
    Pipeline كامل لتدريب نموذج Iris
    End-to-end training pipeline for Iris classification
    
    Steps:
    1. تحميل البيانات (Load data)
    2. تقسيم البيانات (Split data)
    3. تطبيع البيانات (Preprocess)
    4. تدريب النموذج (Train model)
    5. تقييم النموذج (Evaluate)
    6. حفظ النموذج (Export)
    """
    # تحميل البيانات
    X, y = data_loader()
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = data_splitter(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state
    )
    
    # تطبيع البيانات
    X_train_scaled, X_test_scaled, scaler = data_preprocessor(
        X_train=X_train,
        X_test=X_test
    )
    
    # تدريب النموذج
    model = model_trainer(
        X_train=X_train_scaled,
        y_train=y_train,
        C=C,
        max_iter=max_iter
    )
    
    # تقييم النموذج
    metrics = model_evaluator(
        model=model,
        X_test=X_test_scaled,
        y_test=y_test
    )
    
    # حفظ النموذج
    model_path, scaler_path = model_exporter(
        model=model,
        scaler=scaler,
        metrics=metrics,
        version=version
    )


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='تشغيل ZenML training pipeline')
    parser.add_argument('--test-size', type=float, default=0.2, help='حجم بيانات الاختبار')
    parser.add_argument('--C', type=float, default=10.0, help='معامل Regularization')
    parser.add_argument('--max-iter', type=int, default=200, help='عدد التكرارات')
    parser.add_argument('--version', type=str, default='zenml_v1', help='إصدار النموذج')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 ZenML Training Pipeline - MLOps Iris Classification")
    print("=" * 70)
    print(f"\n⚙️ المعاملات:")
    print(f"   - Test Size: {args.test_size}")
    print(f"   - C: {args.C}")
    print(f"   - Max Iter: {args.max_iter}")
    print(f"   - Version: {args.version}")
    print("\n" + "=" * 70)
    
    # تشغيل الـ pipeline
    iris_training_pipeline(
        test_size=args.test_size,
        C=args.C,
        max_iter=args.max_iter,
        version=args.version
    )
    
    print("\n" + "=" * 70)
    print("✅ اكتمل Pipeline بنجاح!")
    print("=" * 70)
    print("\n📋 لعرض runs:")
    print("   zenml pipeline runs list")
    print("\n🌐 لفتح Dashboard:")
    print("   zenml up")
    print("\n" + "=" * 70)