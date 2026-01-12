"""
Load Iris dataset from sklearn and save to CSV
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def load_and_save_iris(output_path='data/raw/iris.csv'):
    """
    Load Iris dataset and save to CSV

    Args:
        output_path: Path to save the CSV file
    """
    print("=" * 60)
    print("🌸 Loading Iris dataset from sklearn...")
    print("=" * 60)
    
    # Load data
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    
    # Add target column
    df['target'] = iris.target
    
    # Add species names for clarity
    target_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(target_names)
    
    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data saved at: {output_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print("First 5 rows:")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Species distribution:")
    print("=" * 60)
    print(df['species'].value_counts())
    
    print("\n" + "=" * 60)
    print("Data statistics:")
    print("=" * 60)
    print(df.describe())
    
    return df


def main():
    """Main function"""
    df = load_and_save_iris()
    
    print("\n" + "=" * 60)
    print("✅ Data prepared successfully!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("  1. Track data with DVC: dvc add data/raw/iris.csv")
    print("  2. Train the model: python src/models/train.py")
    print("  3. Run MLflow: mlflow ui")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
