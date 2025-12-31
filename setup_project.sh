#!/bin/bash

# Setup script for MLOps Iris Project
# هذا السكريبت سينشئ كل المجلدات والملفات الأساسية

set -e

echo "================================================"
echo "🚀 إعداد مشروع MLOps Iris"
echo "================================================"

# 1. إنشاء المجلدات الرئيسية
echo ""
echo "📁 إنشاء هيكل المجلدات..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p mlruns
mkdir -p mlartifacts
mkdir -p configs
mkdir -p src/data
mkdir -p src/models
mkdir -p src/pipelines
mkdir -p src/optimization
mkdir -p src/serving
mkdir -p tests
mkdir -p docker
mkdir -p scripts
mkdir -p docs/screenshots
mkdir -p monitoring/grafana-dashboards
mkdir -p reports

echo "✅ تم إنشاء المجلدات بنجاح"

# 2. إنشاء ملفات __init__.py
echo ""
echo "📝 إنشاء ملفات __init__.py..."

touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/pipelines/__init__.py
touch src/optimization/__init__.py
touch src/serving/__init__.py
touch tests/__init__.py

echo "✅ تم إنشاء ملفات __init__.py"

# 3. إنشاء ملف .gitignore
echo ""
echo "📝 إنشاء ملف .gitignore..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Data
data/raw/*.csv
data/processed/
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.joblib
models/*.h5
!models/.gitkeep

# MLflow
mlruns/
mlartifacts/

# DVC
.dvc/cache/
.dvc/tmp/

# Optuna
*.db

# Logs
*.log

# Environment
.env
EOF

echo "✅ تم إنشاء .gitignore"

# 4. إنشاء ملفات .gitkeep للمجلدات الفارغة
echo ""
echo "📝 إنشاء ملفات .gitkeep..."

touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep
touch reports/.gitkeep

echo "✅ تم إنشاء ملفات .gitkeep"

# 5. عرض الهيكل النهائي
echo ""
echo "================================================"
echo "✅ تم إنشاء هيكل المشروع بنجاح!"
echo "================================================"
echo ""
echo "📊 هيكل المشروع:"
echo ""

# عرض الهيكل (إذا كان tree متوفراً)
if command -v tree &> /dev/null; then
    tree -L 2 -a -I '__pycache__|*.pyc'
else
    find . -maxdepth 2 -type d | grep -v __pycache__ | sort
fi

echo ""
echo "================================================"
echo "📋 الخطوات التالية:"
echo "================================================"
echo ""
echo "1. تهيئة Git:"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial project structure'"
echo ""
echo "2. إنشاء بيئة افتراضية وتثبيت المكتبات:"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # Linux/Mac"
echo "   # أو: .\\venv\\Scripts\\activate  # Windows"
echo "   pip install -r requirements.txt"
echo ""
echo "3. تهيئة DVC:"
echo "   dvc init"
echo ""
echo "4. تهيئة ZenML:"
echo "   zenml init"
echo ""
echo "================================================"