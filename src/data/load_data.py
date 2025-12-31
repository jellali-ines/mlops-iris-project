"""
تحميل بيانات Iris من sklearn وحفظها في CSV
Load Iris dataset from sklearn and save to CSV
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def load_and_save_iris(output_path='data/raw/iris.csv'):
    """
    تحميل بيانات Iris وحفظها
    Load Iris dataset and save to CSV
    
    Args:
        output_path: مسار حفظ الملف
    """
    print("=" * 60)
    print("🌸 تحميل بيانات Iris من sklearn...")
    print("=" * 60)
    
    # تحميل البيانات
    iris = load_iris()
    
    # إنشاء DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    
    # إضافة العمود الهدف
    df['target'] = iris.target
    
    # إضافة أسماء الأنواع للوضوح
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(target_names)
    
    # التأكد من وجود المجلد
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # حفظ إلى CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ تم حفظ البيانات في: {output_path}")
    print(f"📊 الحجم: {df.shape}")
    print(f"📝 الأعمدة: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print("أول 5 صفوف:")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("توزيع الأنواع:")
    print("=" * 60)
    print(df['species'].value_counts())
    
    print("\n" + "=" * 60)
    print("إحصائيات البيانات:")
    print("=" * 60)
    print(df.describe())
    
    return df


def main():
    """الدالة الرئيسية"""
    df = load_and_save_iris()
    
    print("\n" + "=" * 60)
    print("✅ تم إعداد البيانات بنجاح!")
    print("=" * 60)
    print("\n📋 الخطوات التالية:")
    print("  1. تتبع البيانات مع DVC: dvc add data/raw/iris.csv")
    print("  2. تدريب النموذج: python src/models/train.py")
    print("  3. تشغيل MLflow: mlflow ui")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
    