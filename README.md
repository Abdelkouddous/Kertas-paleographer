# Machine Learning Classification System

**Master's thesis — 2022.** This GitHub repository was created in October 2025 when the thesis code was published online. The research itself is a 2022 master's thesis project.

A comprehensive machine learning classification system for comparing different ML algorithms across multiple feature extraction methods.

## 📋 Overview

This project implements and compares multiple classification algorithms:

- **Support Vector Machine (SVM)**
- **Random Forest**
- **Gradient Boosting Trees**
- **AdaBoost**

Applied to two different feature extraction methods:

- **ChainCode Global Features**
- **Polygon Features**

## 🚀 Quick Start

### Option 1: Docker (Recommended for Production) 🐳

**Fastest way to get started:**

```bash
# Using the quick start script
./docker-start.sh          # macOS/Linux
# or
docker-start.bat           # Windows

# Or directly with Docker Compose
docker-compose up -d
```

Access the web interface at: **http://localhost:8501**

**Benefits:**

- ✅ No Python installation required
- ✅ No dependency conflicts
- ✅ Production-ready environment
- ✅ Identical setup across all systems

📚 **See [DOCKER.md](DOCKER.md) for complete Docker documentation**

### Option 2: Local Python Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

The application uses **relative paths** automatically! No configuration needed.

Your data files are already in the correct location:

```
pythonML/
└── KERTASpaleographer/
    ├── training/
    │   ├── features_training_ChainCodeGlobalFE.csv
    │   ├── features_training_PolygonFE.csv
    │   └── label_training.csv
    └── testing/
        ├── features_testing_chainCodeGlobalFE.csv
        ├── features_testing_PolygonFE.csv
        └── label_testing.csv
```

✅ **The paths are automatically detected - just run the application!**

### Running the Application

**Option 1: Web UI (Recommended)** 🎨

```bash
streamlit run app.py
```

Beautiful web interface with interactive visualizations!

**Option 2: Command Line** 💻

```bash
python main.py
```

Traditional terminal interface for advanced users.

## 🎨 User Interfaces

### Web UI (Recommended) 🌐

- **Modern, interactive web interface** built with Streamlit
- Real-time training progress with visual feedback
- Interactive confusion matrices and charts
- Side-by-side model comparisons
- Beautiful, responsive design
- No coding required - just click and train!

**Quick Start:**

```bash
streamlit run app.py
```

See [UI_GUIDE.md](UI_GUIDE.md) for complete documentation.

### Command Line Interface 💻

- Traditional terminal-based interface
- Menu-driven workflow
- Full feature access for advanced users

**Quick Start:**

```bash
python main.py
```

## 📊 Features

### 1. Single Model Training

Train a specific model on selected feature type with optional hyperparameter tuning via Grid Search.

### 2. Model Comparison

Compare all four algorithms on the same dataset and visualize results.

### 3. Feature Type Analysis

Compare performance across different feature extraction methods.

### 4. Visualization

- Confusion matrices with heatmaps
- Accuracy comparison bar charts
- Classification reports

### 5. Hyperparameter Tuning

Optional Grid Search for optimal parameters:

**SVM Parameters:**

- C: [500, 1000, 5000, 10000, 50000]
- gamma: [50000, 5000, 500, 50, 5]
- kernel: ['rbf']

**Random Forest Parameters:**

- n_estimators: [5, 10, 20, 100]
- max_depth: [1, 5, 10, 20]

**Gradient Boosting Parameters:**

- learning_rate: [0.01, 0.05, 0.1]
- n_estimators: [100, 200, 300]

## 📁 Project Structure

```
pythonML/
├── app.py                            # 🎨 Web UI (Streamlit)
├── main.py                           # 💻 Command-line interface
├── test_app.py                       # 🧪 Testing
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── SETUP.md                          # Setup guide
├── TESTING_GUIDE.md                  # Testing guide
├── .gitignore                        # Git ignore file


└── KERTASpaleographer/               # Data and legacy scripts
    ├── training/                     # Training data
    │   ├── features_training_ChainCodeGlobalFE.csv
    │   ├── features_training_PolygonFE.csv
    │   ├── label_training.csv
    │   └── *.m (MATLAB files)
    ├── testing/                      # Testing data
    │   ├── features_testing_chainCodeGlobalFE.csv
    │   ├── features_testing_PolygonFE.csv
    │   ├── label_testing.csv
    │   └── *.m (MATLAB files)
    └── *.py (legacy model scripts)
```

## 🎯 Usage Examples

### Interactive Mode

The application provides an interactive menu:

```
1. Train Single Model         - Train one specific model
2. Compare All Models          - Compare all algorithms
3. Train on ChainCode Features - Full comparison with ChainCode
4. Train on Polygon Features   - Full comparison with Polygon
5. Compare Both Feature Types  - Comprehensive comparison
6. Exit
```

### Example Workflow

1. **Quick Model Test:**

   - Select option 1
   - Choose model and feature type
   - Decide on Grid Search
   - View results and confusion matrix

2. **Full Comparison:**
   - Select option 5
   - Enable Grid Search for best results
   - Compare all models on both feature types
   - Identify best performing combination

## 📈 Output

The system provides:

- **Accuracy scores** for each model
- **Confusion matrices** (normalized)
- **Classification reports** (precision, recall, f1-score)
- **Visual comparisons** via matplotlib/seaborn
- **Best model recommendations**

## 🔧 Classes and Components

### Config

Configuration class containing all paths and hyperparameter grids.

### DataLoader

Handles loading and validation of training/testing datasets.

### ModelTrainer (Base Class)

- `train()`: Train the model
- `evaluate()`: Evaluate and print metrics
- `plot_confusion_matrix()`: Visualize confusion matrix

### Specific Trainers

- `SVMTrainer`
- `RandomForestTrainer`
- `GradientBoostingTrainer`
- `AdaBoostTrainer`

### ModelComparison

Runs all models and generates comparison reports.

## 📊 Sample Output

```
================================================================
MODEL COMPARISON - CHAINCODE FEATURES
================================================================

✓ Data loaded successfully
  Training samples: 1000, Features: 50
  Testing samples: 200, Features: 50

Training SVM Model...
✓ SVM training completed
Model Accuracy: 0.9234 (92.34%)

SUMMARY
================================================================
Model                     Accuracy        Percentage
-------------------------------------------------------
SVM                      0.9234          92.34%
Random Forest            0.9012          90.12%
Gradient Boosting        0.8956          89.56%
AdaBoost                 0.8734          87.34%

🏆 Best Model: SVM with 92.34% accuracy
================================================================
```

## 🛠️ Customization

### Adding New Models

1. Create a new trainer class inheriting from `ModelTrainer`
2. Implement the `train()` method
3. Add to the models dictionary in `ModelComparison.run_comparison()`

### Modifying Hyperparameters

Update the parameter grids in the `Config` class:

```python
class Config:
    YOUR_MODEL_PARAM_GRID = {
        'param1': [values],
        'param2': [values]
    }
```

## 📝 Notes

- Grid Search can be time-consuming for large datasets
- Adjust `cv` (cross-validation folds) for faster training
- Update class names in `Config.CLASS_NAMES` if different
- All paths use `os.path.join()` for cross-platform compatibility

## 🤝 Contributing

Feel free to fork and improve this project. Suggestions for enhancements:

- Add deep learning models
- Implement feature importance analysis
- Add ROC/AUC curves
- Export results to CSV/Excel
- Command-line arguments support

## 📄 License

This project is part of a PFE (Projet de Fin d'Études).

## 👤 Author

**Aymen**

- Project: Machine Learning Classification System
- Year: 2022

## 🐳 Docker Deployment

### Why Docker?

This project includes **production-ready Docker configuration** for:

- 🚀 **Easy Deployment** - One command to start the entire application
- 🔒 **Reproducibility** - Identical environments across dev/staging/production
- ☁️ **Cloud-Ready** - Deploy to AWS, GCP, Azure with minimal changes
- 📦 **No Dependencies** - Everything packaged in the container
- 🔄 **MLDevOps** - Modern ML infrastructure best practices

### Quick Start with Docker

```bash
# Start the application
docker-compose up -d

# Access at http://localhost:8501

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### Features

- ✅ Multi-stage Docker builds (optimized size)
- ✅ Non-root user for security
- ✅ Health checks for reliability
- ✅ Resource limits and monitoring
- ✅ Ready for Kubernetes deployment
- ✅ CI/CD pipeline integration

### Documentation

- 📚 **[DOCKER.md](DOCKER.md)** - Complete Docker deployment guide
- 🚀 **[GETTING_STARTED_DOCKER.md](GETTING_STARTED_DOCKER.md)** - Docker quick start guide
- 🎯 **[QUICK_START.md](QUICK_START.md)** - Application quick start guide
- 🧪 **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Testing guide

### Cloud Deployment

Deploy to your favorite cloud platform:

**AWS ECS:**

```bash
docker push your-registry/kertas-paleographer:latest
# Deploy via ECS console or CLI
```

**Google Cloud Run:**

```bash
gcloud run deploy kertas-paleographer \
  --image gcr.io/PROJECT-ID/kertas-paleographer \
  --platform managed
```

**Azure Container Instances:**

```bash
az container create \
  --resource-group myResourceGroup \
  --name kertas-app \
  --image your-registry/kertas-paleographer
```

See [DOCKER.md](DOCKER.md) for detailed cloud deployment instructions.

---

## 🐛 Troubleshooting

**Issue: FileNotFoundError**

- Solution: Update `BASE_DATA_PATH` in the `Config` class

**Issue: Low accuracy**

- Solution: Enable Grid Search for hyperparameter optimization
- Consider feature engineering or data preprocessing

**Issue: Out of Memory**

- Solution: Reduce Grid Search parameter ranges
- Use smaller dataset for initial testing

**Issue: Docker port already in use**

- Solution: Change port mapping in `docker-compose.yml` or stop conflicting service

**Issue: Docker container exits immediately**

- Solution: Check logs with `docker-compose logs app`

---

## 📄 Project Files

```
Kertas-paleographer/
├── app.py                  # 🎨 Streamlit Web UI
├── main.py                 # 💻 Command-line interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # 🐳 Docker build instructions
├── docker-compose.yml      # 🎼 Multi-container orchestration
├── .dockerignore           # Docker build exclusions
├── docker-start.sh         # 🚀 Quick start script (Linux/Mac)
├── docker-start.bat        # 🚀 Quick start script (Windows)
├── README.md               # This file
├── DOCKER.md               # 📚 Complete Docker guide
├── GETTING_STARTED_DOCKER.md # Docker quick start
├── DOCKER_SETUP_COMPLETE.md  # Docker setup summary
├── QUICK_START.md          # Application quick start guide
├── SETUP.md                # Setup instructions
├── TESTING_GUIDE.md        # Testing guide
└── KERTASpaleographer/     # Data and MATLAB scripts
```

---

For questions or issues, please refer to the documentation or contact the author.

**🐳 Ready for Production | ☁️ Cloud-Native | 🚀 MLDevOps-Ready**
