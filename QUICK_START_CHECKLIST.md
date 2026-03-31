# Quick Start Implementation Checklist
## 24-Hour Hackathon Execution Plan

---

## ✅ Pre-Hackathon (Before You Start)

- [ ] Fork/clone the project repo
- [ ] Set up Git with large file support: `git lfs install`
- [ ] Join a team (ideally: 1 ML engineer, 1 backend engineer, 1 frontend engineer)
- [ ] Download datasets in advance:
  - [ ] Kaggle Wheat Leaf Disease Dataset (~1GB)
  - [ ] PlantVillage subset (if available)
  - [ ] Store in `data/raw/` locally
- [ ] Install Cuda if GPU available (or plan for CPU-only)
- [ ] Test PyTorch GPU: Run `python -c "import torch; print(torch.cuda.is_available())"`

---

## 🚀 PHASE 1: Setup & Data (Hours 0-6)

### Hour 0-1: Environment Setup

```bash
# Clone repo
git clone <your-repo>
cd wheat-disease-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU (if available)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

- [ ] Virtual environment active
- [ ] All dependencies installed (no errors in `pip install`)
- [ ] GPU detected (or accept CPU path)

### Hour 1-2: Dataset Download & Organization

```bash
# Create data structure
mkdir -p data/{raw,processed,splits}

# Download datasets
# Option 1: Kaggle CLI (if setup)
kaggle datasets download -d <dataset-id> -p data/raw/

# Option 2: Manual download from Kaggle/PlantVillage
# - Extract into data/raw/
# - Should have: .jpg/.png images in subdirectories labeled by disease

# Verify structure
ls -la data/raw/
# Expected: folders like "Rust", "Leaf_Blight", "Healthy", etc.
```

- [ ] Downloaded datasets locally
- [ ] Organized into disease folders
- [ ] Total images: 2000+ (target)

### Hour 2-4: Data Pipeline

```python
# scripts/prepare_data.py
# Implement and run this to:
# 1. Find all images recursively
# 2. Create train/val/test splits (60/20/20)
# 3. Save split indices

python scripts/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --seed 42
```

Output should create:
- `data/processed/train/` - symlinks or copies
- `data/processed/val/` - symlinks or copies  
- `data/processed/test/` - symlinks or copies

- [ ] Data splits created (verified no overlap)
- [ ] No class is <100 images (handle imbalance)
- [ ] Can load batch in PyTorch without errors

### Hour 4-6: Sanity Checks

```python
# Quick test to ensure everything works before training

from src.data.dataset import WheatDataset
from src.data.augmentation import WheatDiseaseAugmentation
import torch

# Load dataset
dataset = WheatDataset(
    root='data/processed/train',
    transform=WheatDiseaseAugmentation()
)

# Test batch loading
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
batch = next(iter(loader))
print(f"Batch shape: {batch[0].shape}")  # Should be [32, 3, 224, 224]
print(f"Labels: {batch[1]}")  # Should be [0, 1, 2, 3] mix

# Visualize augmentation
for i in range(3):
    img, label = dataset[i]
    print(f"Image {i}: shape {img.shape}, label {label}")
```

- [ ] DataLoader works without errors
- [ ] Batch shapes correct: (N, 3, 224, 224)
- [ ] Augmentation produces different outputs for same image
- [ ] No NaN or inf values in batches

---

## 🎓 PHASE 2: Model Training & Backend API (Hours 6-12)

### Hour 6-9: Train Stage 1 (Healthy vs Diseased)

```bash
# Start training
python scripts/train.py --stage 1 --epochs 20 --batch-size 32

# Monitor progress:
# - Watch terminal output for loss/acc improvement
# - After epoch 1: loss should decrease
# - After epoch 10: > 85% val accuracy
# - After epoch 20: > 92% val accuracy
```

**Important:** While training, move to parallel task (API setup)

- [ ] Training started (check GPU memory: `nvidia-smi`)
- [ ] First epoch completes in <30 min
- [ ] Loss is decreasing (not NaN)
- [ ] Checkpoints saved at `models/checkpoints/stage1_epoch*.pth`

### Hour 9-10: Stage 1 Evaluation

```bash
# Once stage 1 training done
python scripts/evaluate.py --stage 1

# Output: confusion_matrix.png, metrics.txt
# Expected:
# - Accuracy: >92%
# - False negative rate: <5% (don't miss sick plants!)
```

- [ ] Evaluation metrics saved
- [ ] Accuracy > 85% (minimum for hackathon)
- [ ] No class is missed (recall > 80%)

### Hour 10-12: Train Stage 2 + API Setup (Parallel)

**While Stage 2 trains, implement FastAPI backend:**

```python
# src/api/main.py (from IMPLEMENTATION_TEMPLATES.md)
# Implement core endpoints:
# 1. Load models on startup
# 2. /predict - takes image, returns disease
# 3. /health - shows status
# 4. Error handling + quality checks

python -m uvicorn src.api.main:app --reload
```

**Test API locally:**

```bash
# In new terminal
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"

# Should return JSON:
# {
#   "disease": "Rust",
#   "confidence": 0.87,
#   "stage": 2,
#   ...
# }
```

- [ ] Stage 2 training started (20+ epochs)
- [ ] FastAPI server running on localhost:8000
- [ ] /health endpoint works
- [ ] /predict endpoint accepts image file
- [ ] Database initialized (schema.sql executed)

---

## 🎨 PHASE 3: Frontend & Integration (Hours 12-18)

### Hour 12-14: React Scaffold + Image Upload

```bash
# Create React app
npx create-react-app frontend
# OR (faster)
npm create vite@latest frontend -- --template react
cd frontend

# Install dependencies
npm install axios leaflet react-leaflet

# Start dev server
npm start
```

Build components (from IMPLEMENTATION_TEMPLATES.md):
1. `ImageUploader.jsx` - file upload + preview
2. Hook up to `http://localhost:8000/predict`

- [ ] React dev server running (localhost:3000)
- [ ] Image upload component renders
- [ ] Can select file and see preview
- [ ] API call triggers on upload

### Hour 14-16: Results Display

Build `Results.jsx` component:
- Disease badge with color coding
- Confidence bar
- Treatment recommendations (from API response)
- Prevention tips

```jsx
// Test with mock data first
const mockPrediction = {
  disease: 'Rust',
  confidence: 0.87,
  recommendations: { ... }
};

return <Results prediction={mockPrediction} />;
```

- [ ] Results component renders
- [ ] Disease badge shows correct color
- [ ] Recommendations display properly
- [ ] Styling with Tailwind CSS (not too fancy)

### Hour 16-18: Mapping + Polish

Build `Map.jsx` component using Leaflet:
- Display heatmap of disease spread
- Center on India
- Color gradient: green (healthy) → red (critical)

```jsx
import L from 'leaflet';

useEffect(() => {
  const map = L.map('map').setView([20.5937, 78.9629], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { ... }).addTo(map);
}, []);
```

- [ ] Map loads (no console errors)
- [ ] Can drag/zoom
- [ ] Integration works end-to-end: upload → predict → map

---

## 🧪 PHASE 4: Testing & Deployment (Hours 18-24)

### Hour 18-20: Integration Testing

```bash
# End-to-end test
# 1. Upload test image via frontend
# 2. Verify API processes it
# 3. Check database stores prediction
# 4. Verify map updates

# Test edge cases:
# - Very blurry image → should ask for retake
# - Very dark image → should ask for retake
# - Wrong file type (PDF, etc.) → should reject with clear error
# - Concurrent uploads → should not crash

# Quick stress test
for i in {1..5}; do
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@test_image.jpg" &
done
wait

# Check no GPU OOM or crashes
```

- [ ] Can upload image start-to-finish without errors
- [ ] Predictions consistent (same image → same result)
- [ ] Quality checks work (blur/brightness detection)
- [ ] No SQL errors or database corruption
- [ ] Handles 5+ concurrent requests

### Hour 20-22: Documentation & Packaging

```bash
# Create comprehensive README.md with:
# 1. Quick start (2 minute setup)
# 2. Architecture diagram (ASCII art)
# 3. Model accuracy metrics
# 4. Known limitations
# 5. Troubleshooting section

# Clean up code
# - Remove debug print statements
# - Add docstrings to functions
# - Organize imports

# Export models for submission
python scripts/prepare_submission.py
# Outputs: models.tar.gz with both stage1 & stage2 weights

# Initialize database for fresh deployment
python scripts/init_db.py
```

- [ ] README is clear and complete
- [ ] Model weights exported
- [ ] Database schema documented
- [ ] No large files in Git (use .gitignore)
- [ ] Code is clean (no TODO comments)
- [ ] Final commit with message: "Hackathon submission - Wheat Disease Detection"

### Hour 22-24: Demo & Presentation

**Prepare for live demo or recording:**

```bash
# Get demo images (20-50 real leaf photos)
# From: https://github.com/spMohanty/PlantVillage-Dataset
# Or: Take photos with phone

# Test demo flow:
# 1. Open http://localhost:3000
# 2. Upload disease image
# 3. Show result (disease + confidence)
# 4. Show recommendation
# 5. Show map (if location enabled)
# 6. Repeat for 2-3 different diseases
# 7. Timing: < 3 min total

# Create 2-min presentation:
# - Problem: Farmers lose crops to disease
# - Solution: Mobile app for early detection
# - Key innovation: Two-stage classifier + minimal preprocessing
# - Results: 92% accuracy on test set
# - Future: Offline mode, multilingual UI
# - Live demo (or pre-recorded video)

# Save demo video (optional but impressive):
# FFmpeg command to screen record
ffmpeg -f gdigrab -framerate 30 -i desktop -f dshow -i audio="Microphone" output.mp4
```

- [ ] Demo flow practiced 2-3 times
- [ ] Demo images ready
- [ ] Timing: <3 minutes
- [ ] Presentation deck ready (5 slides max)
- [ ] Can answer: "Why two stages?", "How does it handle low light?", "Accuracy?"
- [ ] Final code pushed to GitHub

---

## 🔧 Troubleshooting Guide

### Problem: "CUDA out of memory" during training

**Solution:**
```bash
# Reduce batch size
python scripts/train.py --stage 1 --batch-size 16  # instead of 32

# Or: Use CPU (slower but works)
# Edit config.py: device = 'cpu'
```

### Problem: Model training but validation loss not improving

**Solution:**
```python
# Check learning rate was too high
# Edit config: learning_rate = 0.0005  # Lower

# Enable early stopping
early_stopping = EarlyStopping(patience=5)

# Check for data leakage (train/val overlap)
python scripts/check_leakage.py
```

### Problem: API returns "Image too blurry" for clear images

**Solution:**
```python
# Adjust blur threshold (in src/api/main.py)
# Current: laplacian_var > 100
# Try: laplacian_var > 80  # Lower threshold = more lenient
```

### Problem: Frontend won't connect to backend API

**Solution:**
```bash
# Check CORS settings
# src/api/main.py should have:
# CORSMiddleware(allow_origins=["*"])

# Check port is correct in frontend
# API: http://localhost:8000
# Frontend: http://localhost:3000

# Test API directly
curl http://localhost:8000/health
# Should return 200 with JSON
```

### Problem: Database errors (UNIQUE constraint failed)

**Solution:**
```bash
# Delete corrupted database and reinitialize
rm database/wheat_disease.db

# Reinitialize with fresh schema
python scripts/init_db.py
```

### Problem: Model weights file too large (>200MB)

**Solution:**
```bash
# Quantize model to reduce size 4x
python scripts/quantize_model.py \
  --input models/stage1_final.pth \
  --output models/stage1_quantized.pth

# Use quantized model instead
# In src/api/main.py: load_model('models/stage1_quantized.pth')
```

### Problem: Very slow inference (>5 seconds per image)

**Solution:**
```bash
# Use GPU if available (automatic in code)
# Or: Use quantized model (see above)
# Or: Use smaller model (MobileNetV2 instead of EfficientNetB0)
```

### Problem: Prediction confidence always <65% (keeps rejecting)

**Solution:**
```python
# Check model was trained properly
python scripts/evaluate.py --stage 2
# If accuracy <70%, model needs more training

# Or: Lower confidence threshold (risky!)
# In src/api/main.py: CONFIDENCE_THRESHOLD = 0.55
```

---

## 📊 Success Criteria

By end of 24 hours, you should have:

**✓ Model:**
- Stage 1 accuracy: > 90% (healthy vs diseased)
- Stage 2 accuracy: > 80% (specific disease)
- Inference time: < 1 second per image
- Handles: different lighting, camera noise, slight blur

**✓ Backend API:**
- /predict endpoint works
- Quality checks implemented
- Database stores predictions
- Error messages are clear

**✓ Frontend:**
- Image upload works
- Results display shows disease + confidence
- Map visualizes disease spread (basic version OK)
- Mobile-responsive design

**✓ Deployment:**
- Works on: GPU and CPU
- Can run locally with: `python scripts/train.py`, `uvicorn src.api.main:app --reload`, `npm start`
- Database initializes automatically
- No hardcoded paths (uses relative paths)

**✓ Documentation:**
- README with quick start
- Architecture diagram
- Model accuracy metrics
- Known limitations listed

**✓ Demo:**
- 2-3 minute live demo showing:
  - Upload image
  - Get disease prediction
  - See recommendations
  - View map
- Presentation: < 5 minutes
- Can answer judges' questions

---

## 🏆 Bonus Features (If Time Allows)

- [ ] Offline model deployment (ONNX export)
- [ ] Multilingual support (Telugu/Hindi)
- [ ] Voice-based input ("wheat disease check")
- [ ] Farmer profile + disease history
- [ ] Share results on WhatsApp
- [ ] Regional statistics dashboard
- [ ] Mobile app (React Native)
- [ ] SMS-based interface for low-bandwidth areas

---

## 📝 Important Notes

1. **Don't perfectionism:** A working MVP beats a beautiful but incomplete system
2. **Prioritize:** MVP > Features > Optimization > UI polish
3. **Test early:** Each component should work before moving to next
4. **Document as you go:** Future you and judges will thank you
5. **Sleep strategically:** Don't code the last 2 hours exhausted
6. **Team coordination:** Clear divide of labor prevents conflicts

---

## 🎯 Final Checklist Before Submission

- [ ] Code committed to GitHub (no uncommitted changes)
- [ ] README updated with all required info
- [ ] Model weights saved and tracked (git-lfs if necessary)
- [ ] Database schema documented
- [ ] Installation instructions tested on fresh clone
- [ ] Demo works start-to-finish
- [ ] Presentation prepared (< 5 min)
- [ ] Team knows what to say and do during demo
- [ ] No hardcoded passwords/API keys in code
- [ ] .gitignore configured properly
- [ ] Final push done with clear commit message

---

**Good luck! 🚀**

