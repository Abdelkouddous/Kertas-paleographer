# Setup Guide

## 🔧 Installation Steps

### 1. Clone the Repository (if not done already)

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/Kertas-paleographer.git

# Navigate to project directory
cd Kertas-paleographer
```

> **Note:** Replace `YOUR_USERNAME` with your actual GitHub username.

### 2. Install Python Dependencies

```bash
# Make sure you're in the project directory
cd Kertas-paleographer

# Install requirements
pip install -r requirements.txt

# Or if using pip3
pip3 install -r requirements.txt
```

### 3. Configure Data Paths

✅ **No configuration needed!** The application automatically uses relative paths.

Your data should be in the correct location:

```
Kertas-paleographer/
└── KERTASpaleographer/
    ├── training/
    │   └── (CSV files)
    └── testing/
        └── (CSV files)
```

The application will automatically find your data files.

### 4. Verify Data Files

Your data directory should have this structure:

```
Kertas-paleographer/
└── KERTASpaleographer/
    ├── training/
    │   ├── features_training_ChainCodeGlobalFE.csv  ✓
    │   ├── features_training_PolygonFE.csv          ✓
    │   └── label_training.csv                       ✓
    └── testing/
        ├── features_testing_chainCodeGlobalFE.csv   ✓
        ├── features_testing_PolygonFE.csv           ✓
        └── label_testing.csv                        ✓
```

All required CSV files should be present and ready to use!

### 5. Run the Application

Make sure you're in the project root directory:

```bash
# Verify you're in the right place
pwd  # Should show: .../Kertas-paleographer

# List files to confirm
ls -la  # Should see: app.py, main.py, requirements.txt, etc.
```

Then run:

```bash
python main.py
```

Or:

```bash
python3 main.py
```

## 🚀 Quick Test

To quickly test if everything is working:

1. Run the application: `python main.py`
2. Select option **1** (Train Single Model)
3. Choose **SVM** (option 1)
4. Choose **ChainCode** features (option 1)
5. Choose **n** for Grid Search (faster testing)
6. Check if the model trains and shows results

## ⚙️ Optional: Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 📊 Expected Output

When you run `python main.py`, you should see:

```
    ╔═══════════════════════════════════════════════════════════════════╗
    ║     MACHINE LEARNING CLASSIFICATION APPLICATION                  ║
    ║     Author: Aymen                                                ║
    ║     Version: 1.0                                                 ║
    ╚═══════════════════════════════════════════════════════════════════╝

======================================================================
  MACHINE LEARNING CLASSIFICATION SYSTEM
======================================================================

1. Train Single Model
2. Compare All Models
3. Train on ChainCode Features
4. Train on Polygon Features
5. Compare Both Feature Types
6. Exit

----------------------------------------------------------------------
Enter your choice (1-6):
```

## 🐛 Troubleshooting

### Import Errors

If you see import errors like:

```
ImportError: No module named 'sklearn'
```

**Solution:**

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

### File Not Found

If you see:

```
FileNotFoundError: [Errno 2] No such file or directory: ...
```

**Solution:**

1. Make sure you're running from the project root directory (Kertas-paleographer/)
2. Verify your CSV files exist in the correct locations (KERTASpaleographer/training/ and KERTASpaleographer/testing/)
3. Note: Testing file is named `chainCodeGlobalFE` (lowercase 'c') while training file is `ChainCodeGlobalFE` (uppercase 'C')

### Permission Issues

If you get permission errors:

```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use pip3
pip3 install --user -r requirements.txt
```

### Memory Issues

If Grid Search causes memory issues:

1. Reduce the parameter grid ranges in `Config` class
2. Reduce `cv` parameter (cross-validation folds)
3. Use a smaller subset of data for testing

## 📝 Additional Notes

- First run might take time if using Grid Search
- Matplotlib plots may require GUI backend on some systems
- For headless servers, save plots instead of showing them

## ✅ Verification Checklist

- [ ] Python 3.7+ installed
- [ ] All packages from requirements.txt installed
- [ ] Data path configured correctly
- [ ] All 6 CSV files present
- [ ] Application runs without import errors
- [ ] At least one model trains successfully

## 🆘 Still Having Issues?

1. Check Python version: `python --version` (should be 3.7+)
2. Verify pip installation: `pip --version`
3. List installed packages: `pip list | grep -E "numpy|pandas|sklearn"`
4. Check file permissions on data directory
5. Try running one of the individual scripts in `main/` folder first

---

For more information, see [README.md](README.md)
