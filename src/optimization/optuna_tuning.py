"""
تحسين Hyperparameters باستخدام Optuna
Hyperparameter optimization using Optuna with MLflow
"""
import argparse
import os
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def prepare_data(csv_path='data/raw/iris.csv', test_size=0.2, random_state=42):
    """تحضير البيانات"""
    df = pd.read_csv(csv_path)
    X = df.drop(['target', 'species'], axis=1).values
    y = df['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


class Objective:
    """دالة الهدف لـ Optuna"""
    
    def __init__(self, X_train, y_train, X_test, y_test, model_type='logistic_regression'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_type = model_type
    
    def __call__(self, trial):
        """تقييم trial واحد"""
        
        # اقتراح Hyperparameters
        if self.model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'saga']),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'random_state': 42
            }
            model = LogisticRegression(**params)
        
        elif self.model_type == 'svm':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'random_state': 42
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            
            model = SVC(**params, probability=True)
        
        else:
            raise ValueError(f"نوع نموذج غير معروف: {self.model_type}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # التدريب على كل البيانات
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='macro')
        
        # تسجيل في MLflow
        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_metric('cv_accuracy_mean', cv_mean)
            mlflow.log_metric('cv_accuracy_std', cv_std)
            mlflow.log_metric('test_accuracy', test_accuracy)
            mlflow.log_metric('test_f1_score', test_f1)
            mlflow.log_metric('trial_number', trial.number)
            mlflow.set_tag('optuna_study', 'iris_optimization')
        
        print(f"Trial {trial.number}: CV Accuracy = {cv_mean:.4f} (+/- {cv_std:.4f}), "
              f"Test Accuracy = {test_accuracy:.4f}")
        
        return cv_mean


def save_best_config(study, model_type, output_path='configs/best_optuna.yaml'):
    """حفظ أفضل تكوين"""
    best_params = study.best_params
    
    config = {
        'experiment_name': 'iris-classification',
        'run_name': 'optuna-best',
        'version': 'optuna_best',
        'data': {
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'type': model_type,
            'params': best_params
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n💾 تم حفظ أفضل تكوين في: {output_path}")
    return output_path


def main(args):
    print("=" * 70)
    print("⚡ تحسين Hyperparameters مع Optuna")
    print("=" * 70)
    
    # إعداد MLflow
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment('iris-optimization')
    
    # تحضير البيانات
    print("\n📊 تحضير البيانات...")
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    # إنشاء Optuna study
    study_name = f"iris_{args.model_type}_study"
    
    print(f"\n⚙️ بدء تحسين Optuna:")
    print(f"   - النموذج: {args.model_type}")
    print(f"   - عدد التجارب: {args.n_trials}")
    print(f"   - اسم الدراسة: {study_name}")
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        load_if_exists=True
    )
    
    # بدء تحسين مع MLflow
    with mlflow.start_run(run_name=f"optuna_study_{args.model_type}"):
        objective = Objective(X_train, y_train, X_test, y_test, args.model_type)
        
        # التحسين
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
        
        # تسجيل النتائج
        mlflow.log_param('n_trials', args.n_trials)
        mlflow.log_param('model_type', args.model_type)
        mlflow.log_metric('best_cv_accuracy', study.best_value)
        
        for key, value in study.best_params.items():
            mlflow.log_param(f'best_{key}', value)
        
        mlflow.log_metric('n_completed_trials', len(study.trials))
        
        print("\n" + "=" * 70)
        print("✅ اكتمل التحسين!")
        print("=" * 70)
        print(f"\n🏆 أفضل تجربة: Trial #{study.best_trial.number}")
        print(f"📊 أفضل CV Accuracy: {study.best_value:.4f}")
        print(f"\n⚙️ أفضل Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")
        print("=" * 70)
        
        # حفظ التكوين
        config_path = save_best_config(study, args.model_type)
        mlflow.log_artifact(str(config_path))
        
        print(f"\n💡 لتدريب النموذج بأفضل تكوين:")
        print(f"   python src/models/train.py --config {config_path}")
        print("\n" + "=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='تحسين Hyperparameters مع Optuna')
    parser.add_argument(
        '--model-type',
        type=str,
        default='logistic_regression',
        choices=['logistic_regression', 'svm'],
        help='نوع النموذج'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='عدد تجارب Optuna'
    )
    
    args = parser.parse_args()
    main(args)
    