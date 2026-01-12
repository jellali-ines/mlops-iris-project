"""
Training script with MLflow tracking
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(csv_path='data/raw/iris.csv', test_size=0.2, random_state=42):
    """
    Load and prepare data
    """
    print("\n📊 Loading data...")
    
    # Read data from CSV
    df = pd.read_csv(csv_path)
    
    # Separate Features and Target
    X = df.drop(['target', 'species'], axis=1).values
    y = df['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Train size: {X_train.shape}")
    print(f"✅ Test size: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(config, X_train, y_train):
    """Train the model"""
    model_type = config['model']['type']
    params = config['model']['params']
    
    print(f"\n🤖 Training model: {model_type}")
    
    if model_type == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_type == 'svm':
        model = SVC(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    print("\n📈 Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred
    }


def save_model(model, scaler, output_dir, version):
    """Save model artifacts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"model_{version}.pkl"
    scaler_path = output_dir / f"scaler_{version}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n💾 Model artifacts saved:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")
    
    return str(model_path), str(scaler_path)


def main(args):
    print("=" * 70)
    print("🚀 Starting Training - MLOps Iris Classification")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n📋 Configuration: {args.config}")
    print(f"📋 Experiment Name: {config['experiment_name']}")
    print(f"📋 Run Name: {config.get('run_name', 'training-run')}")
    
    # Setup MLflow
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config['experiment_name'])
    
    print(f"\n📊 MLflow URI: {mlflow_uri}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=config.get('run_name', 'training-run')):
        
        # Log Parameters
        mlflow.log_params(config['model']['params'])
        mlflow.log_param('model_type', config['model']['type'])
        mlflow.log_param('test_size', config['data']['test_size'])
        mlflow.log_param('random_state', config['data']['random_state'])
        
        # Training
        model, training_time = train_model(config, X_train, y_train)
        
        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log Metrics
        mlflow.log_metric('accuracy', metrics['accuracy'])
        mlflow.log_metric('f1_score', metrics['f1_score'])
        mlflow.log_metric('training_time', training_time)
        
        # Log Artifacts
        mlflow.log_text(str(metrics['confusion_matrix']), 'confusion_matrix.txt')
        mlflow.log_text(metrics['classification_report'], 'classification_report.txt')
        
        # Save model
        version = args.version if args.version else config.get('version', 'v1')
        model_path, scaler_path = save_model(model, scaler, 'models', version)
        
        # Log model artifacts (skipping log_model to avoid errors)
        # mlflow.sklearn.log_model(model, "model")  # temporary disable
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        
        # Tags
        mlflow.set_tags({
            'version': version,
            'framework': 'scikit-learn',
            'dataset': 'iris',
            'model_type': config['model']['type']
        })
        
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print(f"\n📊 Results:")
        print(f"   - Accuracy: {metrics['accuracy']:.4f}")
        print(f"   - F1-Score: {metrics['f1_score']:.4f}")
        print(f"   - Training Time: {training_time:.4f}s")
        print(f"\n💾 Model saved at: {model_path}")
        print(f"\n📈 MLflow:")
        print(f"   - Tracking URI: {mlflow_uri}")
        print(f"   - Run ID: {run_id}")
        print(f"\n💡 To view results, open: {mlflow_uri}")
        print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Iris Model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='Path to config file')
    parser.add_argument('--version', type=str, help='Model version')
    
    args = parser.parse_args()
    main(args)