# AI-Driven Wheat Disease Detection System
## Complete Hackathon Implementation Blueprint

**Status:** Production-Ready Architecture  
**Timeline:** 24-Hour Hackathon  
**Tech Stack:** PyTorch + FastAPI + React + SQLite  

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Model Design & Training](#model-design--training)
4. [Data Pipeline](#data-pipeline)
5. [Backend API](#backend-api)
6. [Frontend UI](#frontend-ui)
7. [Inference Pipeline](#inference-pipeline)
8. [Decision Engine](#decision-engine)
9. [Disease Mapping](#disease-mapping)
10. [Project Structure](#project-structure)
11. [24-Hour Timeline](#24-hour-timeline)
12. [Common Pitfalls](#common-pitfalls)
13. [Deployment Guide](#deployment-guide)

---

## Executive Summary

**Problem:** Farmers need quick, affordable disease detection for wheat crops to minimize losses.

**Solution:** A mobile-friendly web app where farmers upload leaf images → system identifies disease → provides actionable treatment recommendations → visualizes regional disease trends.

**Key Innovation:** 
- Two-stage architecture for robustness (Healthy vs Diseased → Specific Disease)
- Heavy training augmentation + minimal inference preprocessing
- Offline-capable model deployment
- Farmer-friendly interface with confidence thresholds

**Realistic Scope:** 24 hours → 85-92% accuracy, MVP feature set, deployable locally or cloud.

---

## System Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FARMER MOBILE/WEB                         │
│  [Upload Image] → [View Results] → [See Recommendations]    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Route: /predict                                      │   │
│  │ - Image validation & quality check                   │   │
│  │ - Inference orchestration                            │   │
│  │ - Response formatting                                │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────┬──────────────────────┬────────────────────┘
                 │                      │
        ┌────────▼─────────┐  ┌────────▼──────────┐
        │   Model Service  │  │  Decision Engine  │
        │                  │  │                   │
        │ Stage 1: Health  │  │ Disease → Treat   │
        │ Stage 2: Disease │  │ Prevention Tips   │
        │ Classifier       │  │                   │
        └────────┬─────────┘  └────────┬──────────┘
                 │                      │
                 └──────────────┬───────┘
                                │
                         ┌──────▼──────────┐
                         │  Database       │
                         │  (SQLite)       │
                         │  - Predictions  │
                         │  - Locations    │
                         │  - Users        │
                         └────────┬────────┘
                                  │
                         ┌────────▼──────────┐
                         │  Mapping Engine   │
                         │  - Heatmap        │
                         │  - Regional Data  │
                         │  - Trends         │
                         └───────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Tech |
|-----------|---|---|
| **Frontend** | Image upload, results display, map | React/HTML5 |
| **API Gateway** | Route requests, validation, auth | FastAPI |
| **Model Service** | Inference (two-stage) | PyTorch |
| **Decision Engine** | Rule-based recommendations | Python |
| **Database** | Store predictions & locations | SQLite |
| **Map Service** | Visualize disease spread | Leaflet.js |

---

## Model Design & Training

### Architecture: Two-Stage Classifier

**Why two stages?**
- Stage 1 filters out healthy plants (high confidence)
- Stage 2 focuses only on disease classification
- Better accuracy + interpretability
- Faster inference (skip Stage 2 if healthy)

### Stage 1: Healthy vs Diseased

```
Input Image (224×224×3)
        │
        ▼
    ┌─────────────────────┐
    │ EfficientNetB0      │ ← Pre-trained on ImageNet
    │ (remove final layer)│
    └──────────┬──────────┘
               │
        ┌──────▼──────┐
        │ FC Layer 1  │   (512 neurons)
        │ ReLU + Drop │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ FC Layer 2  │   (256 neurons)
        │ ReLU + Drop │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ Output: 2   │  [Healthy, Diseased]
        │ Softmax     │
        └─────────────┘

Output: P(Healthy), P(Diseased)
```

### Stage 2: Disease Classification (Only if Diseased)

```
Input: Diseased Leaf Image (224×224×3)
        │
        ▼
    ┌─────────────────────┐
    │ EfficientNetB0      │ ← Different pre-trained weights
    │ (remove final layer)│
    └──────────┬──────────┘
               │
        ┌──────▼──────┐
        │ FC Layer 1  │   (512 neurons)
        │ ReLU + Drop │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │ FC Layer 2  │   (256 neurons)
        │ ReLU + Drop │
        └──────┬──────┘
               │
        ┌──────▼─────────────────────────┐
        │ Output: 4 Classes               │
        │ [Rust, Leaf_Blight, Powdery,   │
        │  Spot_Blotch]                   │
        │ Softmax                         │
        └────────────────────────────────┘

Output: P(Rust), P(Leaf_Blight), P(Powdery), P(Spot_Blotch)
```

### Why EfficientNetB0?

| Criterion | Why EfficientNetB0 |
|-----------|---|
| **Speed** | 53ms inference on CPU (perfect for 24h) |
| **Accuracy** | 87% ImageNet top-1 (strong baseline) |
| **Parameters** | 5.3M (lightweight, fast training) |
| **Transfer Learning** | Excellent pre-trained weights |
| **Hackathon Fit** | Trains in 4-6 hours on GPU |

**Alternative:** MobileNetV2 (even faster, 3.5M params)

---

## Data Pipeline

### 1. Dataset Sources (Combination Strategy)

```
Available Open-Source Datasets:
├── Wheat Leaf Disease Dataset (Kaggle)
│   └── ~1000 images, 4 diseases
├── PlantVillage (partial wheat subset)
│   └── ~500 images after filtering
├── Synthetic Data (Augmentation)
│   └── Generate 2-3x training set during training
└── User-Generated (During hackathon)
    └── Collect 20-50 real farm images for testing

Target: 2000-3000 training images minimum
```

### 2. Class Imbalance Handling

```python
# Strategy: Weighted sampling + augmentation
class_weights = {
    'Healthy': 1.0,
    'Rust': 1.5,              # Less common, boost sampling
    'Leaf_Blight': 1.3,
    'Powdery': 1.8,           # Rarest, max weight
    'Spot_Blotch': 1.2
}

# Implement: WeightedRandomSampler in DataLoader
# Effect: Minorities appear more frequently per epoch
```

### 3. Data Splits

```
Total Images: 2500
├── Training: 60% (1500) - used for backprop
├── Validation: 20% (500) - used for hyperparameter tuning
└── Test: 20% (500) - held-out for final evaluation

IMPORTANT: No data leakage!
- Split BEFORE augmentation
- No patient/location overlap between splits
- Stratified split to maintain class balance
```

### 4. Preprocessing (Training Phase - MINIMAL)

```python
# ONLY these operations - no destructive filtering!
transforms_train = torchvision.transforms.Compose([
    # Resize to model input size
    torchvision.transforms.Resize((224, 224)),
    
    # Normalize with ImageNet statistics
    # (R: 0.485±0.229, G: 0.456±0.224, B: 0.406±0.225)
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# NO Gaussian Blur
# NO Equalization
# NO Heavy Filtering
# These destroy disease texture patterns!
```

### 5. Augmentation (Training - VERY AGGRESSIVE)

```python
class WheatDiseaseAugmentation:
    """
    Simulate real-world conditions:
    - Different lighting (field varies throughout day)
    - Camera noise (smartphone vs professional)
    - Slight blur (motion, focus issues)
    - Shadows on leaves
    - Rotation (leaf orientation varies)
    """
    
    def __init__(self):
        self.augmentations = A.Compose([
            # Geometric
            A.Rotate(limit=40, p=0.7),              # Field images at angles
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            
            # Lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),                  # Real-world shadows
            A.RandomSunFlare(p=0.3),                # Outdoor conditions
            
            # Noise (smartphone simulation)
            A.GaussNoise(p=0.4),
            A.ISONoise(p=0.3),
            
            # Blur (focus issues)
            A.MotionBlur(blur_limit=3, p=0.3),     # NOT heavy blur!
            A.MedianBlur(blur_limit=3, p=0.2),
            
            # Color (different sensors)
            A.RandomRain(p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3
            ),
            
            # Cutout (robustness)
            A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.2),
        ])
    
    def __call__(self, image):
        return self.augmentations(image=image)['image']
```

### 6. Class Imbalance - Focal Loss

```python
# Standard CrossEntropyLoss fails on imbalanced data
# Use Focal Loss instead

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Focal Loss: Focuses on hard examples
        L = -α(1-p_t)^γ log(p_t)
        
        Where:
        - α: weighting factor (0.25 good default)
        - γ: focusing parameter (2.0 standard)
        - p_t: model's predicted probability for true class
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## Training Pipeline

### Hyperparameters (24-Hour Optimized)

```
Stage 1 (Healthy vs Diseased):
├── Model: EfficientNetB0
├── Batch Size: 32
├── Epochs: 20
├── Learning Rate: 0.001 (Adam optimizer)
├── LR Scheduler: ReduceLROnPlateau (patience=3)
├── Loss: FocalLoss (α=0.25, γ=2.0)
├── Dropout: 0.3
└── Training Time: ~2 hours on V100 GPU

Stage 2 (Disease Classification):
├── Model: EfficientNetB0
├── Batch Size: 32
├── Epochs: 25
├── Learning Rate: 0.0005 (lower, fine-tuning)
├── LR Scheduler: ReduceLROnPlateau (patience=4)
├── Loss: FocalLoss (α=0.25, γ=2.0)
├── Dropout: 0.4
└── Training Time: ~3 hours on V100 GPU
```

### Training Loop Pseudocode

```python
def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
    
    accuracy = 100 * correct / len(train_loader.dataset)
    avg_loss = total_loss / len(train_loader)
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, all_preds, all_labels
```

### Key Training Strategies

| Strategy | Implementation | Benefit |
|----------|---|---|
| **Early Stopping** | Stop if val_loss doesn't improve for 5 epochs | Prevent overfitting |
| **LR Scheduling** | Reduce LR by 0.5x if plateau for 3 epochs | Better convergence |
| **Gradient Clipping** | Clip gradients to norm=1.0 | Stable training |
| **Weight Decay** | L2 regularization = 1e-4 | Generalization |
| **Warmup** | 1 epoch at LR/10 | Stable initialization |

---

## Backend API

### Tech Stack Choice: FastAPI

**Why FastAPI?**
- Built-in async support (handle multiple uploads simultaneously)
- Auto-generates API documentation (Swagger)
- Fast JSON serialization (Pydantic)
- Native support for file uploads
- Production-ready with Uvicorn

### Core API Endpoints

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Wheat Disease Detection API")

# ============= Data Models =============

class PredictionResponse(BaseModel):
    """API Response Schema"""
    disease: str                          # "Rust" | "Leaf_Blight" | ...
    stage: str                            # "Healthy" | "Diseased"
    confidence: float                     # 0.0-1.0
    image_quality: dict                   # blur_score, brightness_score
    recommendations: dict                 # treatment, prevention
    request_id: str
    timestamp: str

class LocationData(BaseModel):
    latitude: float
    longitude: float
    farmer_name: Optional[str] = None
    contact: Optional[str] = None

# ============= Model Loading =============

# Global variables (load once at startup)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stage1_model = None
stage2_model = None

@app.on_event("startup")
async def load_models():
    """Load models once at server startup"""
    global stage1_model, stage2_model
    
    # Load Stage 1: Healthy vs Diseased
    stage1_model = EfficientNetB0(num_classes=2)
    stage1_model.load_state_dict(
        torch.load('models/stage1_weights.pth', map_location=DEVICE)
    )
    stage1_model.to(DEVICE)
    stage1_model.eval()
    
    # Load Stage 2: Disease Classification
    stage2_model = EfficientNetB0(num_classes=4)
    stage2_model.load_state_dict(
        torch.load('models/stage2_weights.pth', map_location=DEVICE)
    )
    stage2_model.to(DEVICE)
    stage2_model.eval()

# ============= Utility Functions =============

def check_image_quality(image: np.ndarray) -> dict:
    """
    Detect blurry or dark images
    Returns quality metrics
    """
    # Blur detection using Laplacian variance
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Brightness check
    brightness = np.mean(gray)
    
    return {
        'blur_score': laplacian_var,           # >100 = clear
        'brightness_score': brightness,         # 40-200 = good
        'is_clear': laplacian_var > 100,
        'is_bright_enough': brightness > 40
    }

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Minimal preprocessing (resize + normalize only)
    NO destructive filtering
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize
    image = image.resize((224, 224), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_array = np.array(image) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to torch tensor
    tensor = torch.from_numpy(image_array).float()
    tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
    
    return tensor.unsqueeze(0)  # Add batch dimension

def run_inference(image_tensor: torch.Tensor) -> dict:
    """
    Two-stage inference pipeline
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        
        # Stage 1: Healthy vs Diseased
        stage1_logits = stage1_model(image_tensor)
        stage1_probs = torch.softmax(stage1_logits, dim=1)
        
        is_healthy = stage1_probs[0, 0] > 0.7  # Threshold
        healthy_conf = stage1_probs[0, 0].item()
        
        if is_healthy:
            return {
                'disease': 'Healthy',
                'confidence': healthy_conf,
                'stage': 'Stage1'
            }
        
        # Stage 2: Disease Classification (only if diseased)
        stage2_logits = stage2_model(image_tensor)
        stage2_probs = torch.softmax(stage2_logits, dim=1)
        
        confidence = stage2_probs.max().item()
        disease_idx = stage2_probs.argmax().item()
        
        disease_names = ['Rust', 'Leaf_Blight', 'Powdery_Mildew', 'Spot_Blotch']
        disease = disease_names[disease_idx]
        
        return {
            'disease': disease,
            'confidence': confidence,
            'stage': 'Stage2',
            'probabilities': {
                disease_names[i]: stage2_probs[0, i].item()
                for i in range(4)
            }
        }

# ============= API Endpoints =============

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    location: Optional[LocationData] = None
):
    """
    Main prediction endpoint
    
    Input:
        - file: Image file (JPG/PNG)
        - location: Optional GPS data
    
    Output:
        - PredictionResponse with disease, confidence, recommendations
    """
    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(400, "Only JPEG/PNG images supported")
        
        # Read image
        image_bytes = await file.read()
        image_array = np.array(Image.open(io.BytesIO(image_bytes)))
        
        # Check quality
        quality = check_image_quality(image_array)
        
        # ⚠️ IMPORTANT: If quality is poor, ask for retake
        if not quality['is_clear']:
            raise HTTPException(400, {
                'message': 'Image is too blurry. Please retake.',
                'blur_score': quality['blur_score'],
                'threshold': 100
            })
        
        if not quality['is_bright_enough']:
            raise HTTPException(400, {
                'message': 'Image is too dark. Try better lighting.',
                'brightness': quality['brightness_score'],
                'threshold': 40
            })
        
        # Preprocess
        tensor = preprocess_image(image_bytes)
        
        # Inference
        result = run_inference(tensor)
        
        # Get recommendations
        recommendations = get_recommendations(result['disease'])
        
        # Store in database
        request_id = store_prediction(
            disease=result['disease'],
            confidence=result['confidence'],
            location=location,
            quality=quality
        )
        
        return PredictionResponse(
            disease=result['disease'],
            stage=result['stage'],
            confidence=result['confidence'],
            image_quality=quality,
            recommendations=recommendations,
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'models_loaded': stage1_model is not None
    }

@app.get("/stats")
async def get_statistics():
    """Get regional disease statistics"""
    # Aggregate predictions from database
    stats = get_disease_statistics_from_db()
    return stats
```

### Error Handling Strategy

```python
class WheatDiseaseAPIException(Exception):
    """Custom exception for API errors"""
    
    def __init__(self, code: int, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}

# Usage in endpoints:
@app.exception_handler(WheatDiseaseAPIException)
async def exception_handler(request, exc):
    return {
        "error": exc.message,
        "code": exc.code,
        "details": exc.details,
        "timestamp": datetime.now().isoformat()
    }
```

---

## Frontend UI

### Architecture: React + Tailwind CSS (Hackathon Optimized)

**Why React?**
- Component reusability
- State management with hooks
- Fast development
- Perfect for real-time uploads

### Key Components

```jsx
// App.js - Main Application Shell
import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import Results from './components/Results';
import Map from './components/Map';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <header className="bg-white shadow">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-green-700">
            🌾 Wheat Disease Detection
          </h1>
          <p className="text-gray-600 mt-2">
            Upload a leaf image for instant disease diagnosis
          </p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div>
            <ImageUploader 
              onUpload={handleUpload} 
              loading={loading}
            />
          </div>

          {/* Results Section */}
          <div>
            {loading && <LoadingSpinner />}
            {error && <ErrorAlert message={error} />}
            {prediction && <Results prediction={prediction} />}
          </div>
        </div>

        {/* Map Section - Full Width */}
        {prediction?.location && (
          <div className="mt-12">
            <Map locations={[prediction.location]} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
```

```jsx
// components/ImageUploader.jsx
import React, { useState, useRef } from 'react';

function ImageUploader({ onUpload, loading }) {
  const [preview, setPreview] = useState(null);
  const [location, setLocation] = useState(null);
  const fileInput = useRef(null);

  const handleFileSelect = async (file) => {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);

    // Get location if available
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((pos) => {
        setLocation({
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude
        });
      });
    }

    // Upload
    const formData = new FormData();
    formData.append('file', file);
    if (location) {
      formData.append('location', JSON.stringify(location));
    }

    onUpload(formData);
  };

  return (
    <div className="border-2 border-dashed border-green-300 rounded-lg p-8 text-center">
      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        onChange={(e) => handleFileSelect(e.target.files[0])}
        className="hidden"
      />

      {preview ? (
        <div>
          <img src={preview} alt="Preview" className="w-full rounded" />
          <button
            onClick={() => fileInput.current.click()}
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded"
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Change Image'}
          </button>
        </div>
      ) : (
        <div
          onClick={() => fileInput.current.click()}
          className="cursor-pointer"
        >
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <p className="mt-4 text-lg font-medium">Click to upload wheat leaf image</p>
          <p className="text-sm text-gray-500">or drag and drop</p>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
```

```jsx
// components/Results.jsx
import React from 'react';
import { ConfidenceBar, RecommendationCard } from './SubComponents';

function Results({ prediction }) {
  const getDiseaseColor = (disease) => {
    const colors = {
      'Healthy': 'bg-green-100 text-green-800 border-green-300',
      'Rust': 'bg-orange-100 text-orange-800 border-orange-300',
      'Leaf_Blight': 'bg-red-100 text-red-800 border-red-300',
      'Powdery_Mildew': 'bg-purple-100 text-purple-800 border-purple-300',
      'Spot_Blotch': 'bg-yellow-100 text-yellow-800 border-yellow-300'
    };
    return colors[disease] || 'bg-gray-100';
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Disease Badge */}
      <div className={`inline-block px-4 py-2 rounded-full font-bold border-2 ${getDiseaseColor(prediction.disease)}`}>
        {prediction.disease === 'Healthy' ? '✓' : '⚠'} {prediction.disease}
      </div>

      {/* Confidence */}
      <div className="mt-6">
        <label className="block text-sm font-medium text-gray-700">Detection Confidence</label>
        <ConfidenceBar value={prediction.confidence} />
        <p className="text-sm text-gray-600 mt-2">
          {(prediction.confidence * 100).toFixed(1)}% confident this is {prediction.disease}
        </p>
      </div>

      {/* Quality Check */}
      <div className="mt-6 p-4 bg-blue-50 rounded">
        <h3 className="font-semibold mb-2">Image Quality</h3>
        <p className="text-sm">
          ✓ Clear: {prediction.image_quality.is_clear ? 'Yes' : 'Blurry - retake'}
        </p>
        <p className="text-sm">
          ✓ Brightness: {prediction.image_quality.brightness_score.toFixed(0)}/255
        </p>
      </div>

      {/* Recommendations */}
      {prediction.disease !== 'Healthy' && (
        <div className="mt-6">
          <h3 className="font-semibold mb-4">Treatment & Prevention</h3>
          <div className="space-y-3">
            {prediction.recommendations.treatments.map((t, i) => (
              <RecommendationCard key={i} type="treatment" text={t} />
            ))}
            {prediction.recommendations.prevention.map((p, i) => (
              <RecommendationCard key={i} type="prevention" text={p} />
            ))}
          </div>
        </div>
      )}

      {/* Share/Export */}
      <div className="mt-6 flex gap-2">
        <button className="flex-1 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          📍 Share Location
        </button>
        <button className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400">
          📥 Download Report
        </button>
      </div>
    </div>
  );
}

export default Results;
```

```jsx
// components/Map.jsx
import React, { useEffect } from 'react';
import L from 'leaflet';

function Map({ locations }) {
  useEffect(() => {
    const map = L.map('map').setView([20.5937, 78.9629], 5); // India center

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap',
      maxZoom: 19
    }).addTo(map);

    // Add heatmap layer
    const diseaseHeatData = locations.map(loc => [
      loc.latitude,
      loc.longitude,
      getHeatmapIntensity(loc.disease)
    ]);

    const heat = L.heatLayer(diseaseHeatData, {
      radius: 25,
      blur: 15,
      maxZoom: 17,
      gradient: {
        0.0: 'green',
        0.25: 'lime',
        0.5: 'yellow',
        0.75: 'orange',
        1.0: 'red'
      }
    }).addTo(map);

    return () => map.remove();
  }, [locations]);

  return <div id="map" className="w-full h-96 rounded-lg shadow" />;
}

function getHeatmapIntensity(disease) {
  const intensities = {
    'Healthy': 0.1,
    'Spot_Blotch': 0.4,
    'Leaf_Blight': 0.6,
    'Rust': 0.8,
    'Powdery_Mildew': 0.9
  };
  return intensities[disease] || 0.5;
}

export default Map;
```

### Styling: Tailwind Configuration

```js
// tailwind.config.js
module.exports = {
  content: ['./src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        wheat: {
          50: '#FEF9E7',
          500: '#D4A574',
          900: '#6B4C2E'
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite'
      }
    }
  },
  plugins: []
};
```

---

## Inference Pipeline

### Real-Time Workflow

```python
"""
Complete inference pipeline for production
Handles edge cases, quality checks, confidence thresholds
"""

class WheatDiseaseInferencePipeline:
    
    def __init__(self, stage1_model_path, stage2_model_path, device='cpu'):
        self.device = torch.device(device)
        self.stage1_model = self._load_model(stage1_model_path, num_classes=2)
        self.stage2_model = self._load_model(stage2_model_path, num_classes=4)
        
        self.disease_names = ['Rust', 'Leaf_Blight', 'Powdery_Mildew', 'Spot_Blotch']
        self.confidence_threshold = 0.65  # Reject if <65%
        
    def _load_model(self, path, num_classes):
        model = EfficientNetB0(num_classes=num_classes)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Minimal preprocessing - only resize + normalize
        """
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224), Image.BILINEAR)
        
        image_array = np.array(image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
        return tensor.unsqueeze(0).to(self.device)
    
    def check_blur(self, image_array: np.ndarray, threshold=100) -> bool:
        """
        Laplacian variance test
        High variance = clear image
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance > threshold, variance
    
    def check_brightness(self, image_array: np.ndarray) -> bool:
        """
        Check if image is too dark
        """
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        return brightness > 40, brightness
    
    @torch.no_grad()
    def predict(self, image_array: np.ndarray) -> dict:
        """
        Main prediction method
        
        Returns:
        {
            'disease': str,
            'confidence': float,
            'is_healthy': bool,
            'stage': int,
            'quality_issues': list,
            'probabilities': dict,
            'recommendation_level': str  # 'high', 'medium', 'low'
        }
        """
        
        # Quality checks
        quality_issues = []
        
        is_clear, blur_score = self.check_blur(image_array)
        if not is_clear:
            quality_issues.append(f'Image blur: {blur_score:.1f} (threshold: 100)')
        
        is_bright, brightness = self.check_brightness(image_array)
        if not is_bright:
            quality_issues.append(f'Image too dark: {brightness:.1f} (threshold: 40)')
        
        # If severe quality issues, return early
        if quality_issues and blur_score < 50:
            return {
                'error': 'Image quality too poor',
                'quality_issues': quality_issues,
                'blur_score': blur_score,
                'can_retry': True
            }
        
        # Preprocess
        tensor = self.preprocess_from_array(image_array)
        
        # Stage 1: Healthy vs Diseased
        stage1_logits = self.stage1_model(tensor)
        stage1_probs = torch.softmax(stage1_logits, dim=1)[0]
        
        is_healthy = stage1_probs[0] > 0.7
        healthy_confidence = stage1_probs[0].item()
        
        if is_healthy:
            return {
                'disease': 'Healthy',
                'confidence': healthy_confidence,
                'is_healthy': True,
                'stage': 1,
                'quality_issues': quality_issues,
                'recommendation_level': 'low',
                'probabilities': {'Healthy': healthy_confidence}
            }
        
        # Stage 2: Specific disease classification
        stage2_logits = self.stage2_model(tensor)
        stage2_probs = torch.softmax(stage2_logits, dim=1)[0]
        
        confidence = stage2_probs.max().item()
        disease_idx = stage2_probs.argmax().item()
        disease = self.disease_names[disease_idx]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                'disease': disease,
                'confidence': confidence,
                'is_uncertain': True,
                'message': f'Low confidence ({confidence:.1%}). Please retake image.',
                'stage': 2,
                'quality_issues': quality_issues,
                'recommendation_level': 'low',
                'probabilities': {self.disease_names[i]: stage2_probs[i].item() for i in range(4)}
            }
        
        # High confidence - return full prediction
        return {
            'disease': disease,
            'confidence': confidence,
            'is_healthy': False,
            'stage': 2,
            'quality_issues': quality_issues,
            'recommendation_level': 'high' if confidence > 0.85 else 'medium',
            'probabilities': {
                self.disease_names[i]: stage2_probs[i].item() 
                for i in range(4)
            }
        }
```

---

## Decision Engine

### Rule-Based Recommendation System

```python
class DecisionEngine:
    """
    Maps disease → treatment/prevention recommendations
    Keeps implementation simple for hackathon
    """
    
    DISEASE_KNOWLEDGE_BASE = {
        'Healthy': {
            'explanation': 'Plant is healthy. Continue normal care.',
            'treatments': [
                'No treatment needed'
            ],
            'prevention': [
                'Maintain regular watering schedule',
                'Monitor for early signs of disease',
                'Ensure good field hygiene'
            ],
            'risk_level': 'Low',
            'urgent': False
        },
        
        'Rust': {
            'explanation': 'Fungal disease causing rust-colored pustules on leaves.',
            'treatments': [
                'Apply sulfur-based fungicide (1% solution) - spray both leaf surfaces',
                'Recommended products: Sulfur dust, Wettable sulfur',
                'Spray interval: 10-14 days during growing season',
                'Cost estimate: ₹200-400 per hectare'
            ],
            'prevention': [
                'Remove infected leaves immediately',
                'Ensure good air circulation - don\'t overcrowd plants',
                'Avoid overhead watering (increases humidity)',
                'Use disease-resistant wheat varieties',
                'Plow under infected crop residue after harvest'
            ],
            'fertilizer': {
                'high_nitrogen_caution': True,
                'recommendation': 'Reduce nitrogen - use potassium-based fertilizer (K:N = 1:1)'
            },
            'water_management': 'Reduce moisture - allow soil to dry between watering',
            'risk_level': 'Medium',
            'urgent': False,
            'days_to_damage': '10-14 days if untreated'
        },
        
        'Leaf_Blight': {
            'explanation': 'Fungal blight causes water-soaked lesions, rapid leaf death.',
            'treatments': [
                'Apply copper-based fungicide (0.5-1% Bordeaux mixture)',
                'Alternative: Tricyclazole (75% WP) at 1g/L water',
                'Spray immediately on detection',
                'Repeat spray every 7-10 days until disease stops spreading'
            ],
            'prevention': [
                'Use certified disease-resistant seed',
                'Practice crop rotation (avoid planting wheat after susceptible host)',
                'Remove and burn infected crop residue',
                'Drain water logging immediately',
                'Space plants for air circulation'
            ],
            'fertilizer': {
                'high_nitrogen_caution': True,
                'recommendation': 'Avoid excess nitrogen - causes tender growth (more susceptible)'
            },
            'risk_level': 'High',
            'urgent': True,
            'days_to_damage': '3-5 days if untreated - FAST SPREADING'
        },
        
        'Powdery_Mildew': {
            'explanation': 'White powder coating on leaves - fungal infection.',
            'treatments': [
                'Spray sulfur dust (wettable sulfur 80% WP) - safest option',
                'Apply potassium bicarbonate - organic certified',
                'Use neem oil spray every 5-7 days',
                'Cost estimate: ₹150-300 per hectare'
            ],
            'prevention': [
                'Avoid dense planting - ensure air gaps',
                'Morning irrigation at soil level (not foliage)',
                'Remove infected leaves promptly',
                'Use mulch to prevent soil splash',
                'Destroy infected plant debris'
            ],
            'environment': 'Prefers cool nights + warm days (15-25°C). Worse in spring/fall.',
            'water_management': 'Water early morning - allow foliage to dry quickly',
            'risk_level': 'Medium',
            'urgent': False,
            'days_to_damage': '14-21 days'
        },
        
        'Spot_Blotch': {
            'explanation': 'Dark, irregular spots with concentric rings on leaves.',
            'treatments': [
                'Apply Propiconazole (25% EC) at 0.1% concentration',
                'Alternative: Hexaconazole 5% EC at 0.05% concentration',
                'First spray at Z23-Z25 growth stage (ear emergence)',
                'Follow-up spray after 2 weeks if needed'
            ],
            'prevention': [
                'Use tolerant varieties (UAS 328, PBW 175)',
                'Avoid overhead irrigation - water at base only',
                'Crop rotation (minimum 2 years)',
                'Remove volunteer wheat plants',
                'Sanitize equipment between fields'
            ],
            'fertilizer': {
                'recommendation': 'Balanced NPK (recommended ratio 1:0.5:0.5 of N)',
                'zinc_deficiency_note': 'Zinc deficiency may increase susceptibility'
            },
            'climate': 'Favored by warm, wet conditions. More common in irrigated areas.',
            'risk_level': 'Medium',
            'urgent': False,
            'days_to_damage': '7-10 days'
        }
    }
    
    def get_recommendation(self, disease: str, confidence: float) -> dict:
        """
        Get treatment/prevention recommendations
        Adjusted based on confidence level
        """
        
        if disease not in self.DISEASE_KNOWLEDGE_BASE:
            return {'error': f'Unknown disease: {disease}'}
        
        knowledge = self.DISEASE_KNOWLEDGE_BASE[disease]
        
        # Adjust certainty messaging based on confidence
        certainty = 'high' if confidence > 0.85 else 'medium' if confidence > 0.7 else 'low'
        
        return {
            'disease': disease,
            'explanation': knowledge['explanation'],
            'certainty': certainty,
            'treatments': knowledge.get('treatments', []),
            'prevention': knowledge.get('prevention', []),
            'risk_level': knowledge.get('risk_level', 'Unknown'),
            'urgent': knowledge.get('urgent', False),
            'fertilizer': knowledge.get('fertilizer', {}),
            'water_management': knowledge.get('water_management', ''),
            'days_to_damage': knowledge.get('days_to_damage', 'Unknown'),
            'next_action': self._get_next_action(disease, confidence)
        }
    
    def _get_next_action(self, disease: str, confidence: float) -> str:
        """
        Provide actionable next steps
        """
        if disease == 'Healthy':
            return 'Continue monitoring. Check again in 2 weeks.'
        
        if confidence < 0.7:
            return f'Confidence is low. Consult agricultural expert or retake image.'
        
        risk_level = self.DISEASE_KNOWLEDGE_BASE[disease].get('risk_level', 'Medium')
        
        if risk_level == 'High':
            return 'URGENT: Apply fungicide within 24 hours. Contact local agricultural officer.'
        elif risk_level == 'Medium':
            return 'Apply recommended fungicide within 3-5 days.'
        else:
            return 'Monitor closely. Implement prevention measures.'
```

### Database Schema for Recommendations

```sql
-- Store all predictions for analytics
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    image_path TEXT,
    disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    stage INTEGER,  -- 1 or 2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    user_id TEXT
);

CREATE TABLE locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accuracy REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT PRIMARY KEY,
    treatments TEXT,  -- JSON array
    prevention TEXT,  -- JSON array
    fertilizer_config TEXT,  -- JSON
    last_updated TIMESTAMP
);

-- Indices for fast queries
CREATE INDEX idx_disease ON predictions(disease);
CREATE INDEX idx_created_at ON predictions(created_at);
CREATE INDEX idx_location ON locations(latitude, longitude);
```

---

## Disease Mapping

### Visualization Architecture

```python
"""
Heatmap generation and regional analysis
"""

class DiseaseMapper:
    
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
    
    def get_heatmap_data(self, bounding_box=None, time_range=None):
        """
        Get all predictions with locations for heatmap
        
        Returns list of [lat, lon, intensity] for Leaflet heatmap
        Intensity: 0.0 (Healthy) to 1.0 (Severe disease)
        """
        
        query = """
        SELECT l.latitude, l.longitude, p.disease, p.confidence
        FROM locations l
        JOIN predictions p ON l.prediction_id = p.prediction_id
        WHERE p.created_at >= datetime('now', '-7 days')
        AND l.latitude IS NOT NULL AND l.longitude IS NOT NULL
        """
        
        cursor = self.db.execute(query)
        data = []
        
        for lat, lon, disease, confidence in cursor:
            # Map disease to intensity
            intensity = self._disease_to_intensity(disease, confidence)
            data.append([lat, lon, intensity])
        
        return data
    
    def _disease_to_intensity(self, disease: str, confidence: float) -> float:
        """
        Convert disease type + confidence to heatmap intensity
        Range: 0.0 (safe) to 1.0 (critical)
        """
        
        disease_severity = {
            'Healthy': 0.0,
            'Spot_Blotch': 0.3,
            'Rust': 0.5,
            'Leaf_Blight': 0.8,
            'Powdery_Mildew': 0.4
        }
        
        base_severity = disease_severity.get(disease, 0.5)
        # Multiply by confidence (more confident = higher intensity)
        intensity = base_severity * confidence
        
        return min(intensity, 1.0)  # Cap at 1.0
    
    def get_regional_stats(self, region_bounds: dict) -> dict:
        """
        Get disease statistics for a region
        
        region_bounds: {'north': lat, 'south': lat, 'east': lon, 'west': lon}
        """
        
        query = """
        SELECT p.disease, COUNT(*) as count, AVG(p.confidence) as avg_confidence
        FROM predictions p
        JOIN locations l ON p.prediction_id = l.prediction_id
        WHERE l.latitude BETWEEN ? AND ?
        AND l.longitude BETWEEN ? AND ?
        AND p.created_at >= datetime('now', '-30 days')
        GROUP BY p.disease
        ORDER BY count DESC
        """
        
        cursor = self.db.execute(query, (
            region_bounds['south'], region_bounds['north'],
            region_bounds['west'], region_bounds['east']
        ))
        
        stats = {
            'total_predictions': 0,
            'disease_distribution': {}
        }
        
        for disease, count, avg_conf in cursor:
            stats['disease_distribution'][disease] = {
                'count': count,
                'avg_confidence': avg_conf
            }
            stats['total_predictions'] += count
        
        return stats
    
    def get_trend_data(self, disease: str, days: int = 30):
        """
        Get temporal trend for specific disease
        """
        
        query = """
        SELECT DATE(p.created_at) as date, COUNT(*) as count
        FROM predictions p
        WHERE p.disease = ? AND p.created_at >= datetime('now', ? || ' days')
        GROUP BY DATE(p.created_at)
        ORDER BY date ASC
        """
        
        cursor = self.db.execute(query, (disease, -days))
        
        return [
            {'date': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]
```

### Frontend Mapping Component

```jsx
// components/DiseaseMap.jsx
import React, { useEffect, useState } from 'react';
import L from 'leaflet';
import 'leaflet-heatmap';

function DiseaseMap() {
  const [map, setMap] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    // Initialize map centered on India
    const newMap = L.map('map-container').setView([20.5937, 78.9629], 5);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap',
      maxZoom: 19
    }).addTo(newMap);

    setMap(newMap);

    // Fetch heatmap data from backend
    fetch('/api/heatmap-data')
      .then(res => res.json())
      .then(data => {
        // Add heatmap layer
        L.heatLayer(data, {
          radius: 30,
          blur: 20,
          maxZoom: 17,
          gradient: {
            0.0: '#06B6D4',    // Healthy (cyan)
            0.25: '#10B981',   // Minor (green)
            0.5: '#EAB308',    // Moderate (yellow)
            0.75: '#F97316',   // Severe (orange)
            1.0: '#DC2626'     // Critical (red)
          }
        }).addTo(newMap);
      });

    // Fetch regional statistics
    fetch('/api/regional-stats')
      .then(res => res.json())
      .then(data => setStats(data));

    return () => newMap.remove();
  }, []);

  return (
    <div className="w-full">
      <div id="map-container" className="h-96 rounded-lg shadow-lg" />
      
      {stats && (
        <div className="mt-6 grid grid-cols-2 gap-4 p-4 bg-white rounded-lg">
          <div>
            <h3 className="font-semibold">Disease Distribution</h3>
            {Object.entries(stats.disease_distribution).map(([disease, data]) => (
              <div key={disease} className="text-sm mt-2">
                <span className="font-medium">{disease}:</span>
                <span className="ml-2">{data.count} ({(data.avg_confidence * 100).toFixed(0)}%)</span>
              </div>
            ))}
          </div>
          
          <div>
            <h3 className="font-semibold">Statistics</h3>
            <p className="text-sm mt-2">Total Predictions: {stats.total_predictions}</p>
            <p className="text-sm">Highest Risk Area: {stats.highest_risk}</p>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="mt-4 p-4 bg-white rounded-lg">
        <h3 className="font-semibold mb-3">Heatmap Legend</h3>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-cyan-500"></div>
            <span>Healthy (0.0)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-green-500"></div>
            <span>Minor Disease (0.25)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-yellow-500"></div>
            <span>Moderate (0.5)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-orange-500"></div>
            <span>Severe (0.75)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-red-500"></div>
            <span>Critical (1.0)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DiseaseMap;
```

---

## Project Structure

```
wheat-disease-detection/
│
├── README.md
├── requirements.txt
├── setup.py
│
├── data/                           # Data pipeline
│   ├── raw/                        # Raw images from datasets
│   ├── processed/                  # Cleaned, normalized data
│   ├── splits/                     # Train/val/test splits
│   └── config.yaml                 # Dataset configuration
│
├── models/                         # Pre-trained models
│   ├── stage1_health_classifier.pth
│   ├── stage2_disease_classifier.pth
│   └── model_config.json           # Hyperparameters
│
├── src/                            # Source code
│   │
│   ├── data/                       # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── augmentation.py         # Albumentations config
│   │   ├── loading.py              # DataLoader setup
│   │   └── combine_sources.py      # Merge multiple datasets
│   │
│   ├── models/                     # Model architecture
│   │   ├── __init__.py
│   │   ├── efficientnet.py         # EfficientNetB0 wrapper
│   │   ├── two_stage_classifier.py # Two-stage architecture
│   │   └── utils.py                # Model utilities
│   │
│   ├── training/                   # Training loop
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training loop
│   │   ├── losses.py               # Focal loss, custom losses
│   │   ├── metrics.py              # Accuracy, precision, recall, F1
│   │   ├── callbacks.py            # Early stopping, LR scheduler
│   │   └── config.py               # Training hyperparameters
│   │
│   ├── inference/                  # Runtime inference
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Inference pipeline class
│   │   ├── quality_checks.py       # Image validation
│   │   └── preprocessor.py         # Minimal preprocessing
│   │
│   ├── decision_engine/            # Recommendations
│   │   ├── __init__.py
│   │   ├── engine.py               # Rule-based recommendations
│   │   ├── knowledge_base.py       # Disease knowledge
│   │   └── formatter.py            # Output formatting
│   │
│   ├── mapping/                    # Geographic visualization
│   │   ├── __init__.py
│   │   ├── mapper.py               # Heatmap generation
│   │   ├── analytics.py            # Regional analysis
│   │   └── database.py             # DB operations
│   │
│   ├── api/                        # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes.py               # API endpoints
│   │   ├── schemas.py              # Pydantic models
│   │   ├── middleware.py           # CORS, auth
│   │   └── handlers.py             # Error handling
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── logger.py               # Logging setup
│       ├── config.py               # Configuration management
│       └── helpers.py              # Helper functions
│
├── frontend/                       # React app
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── Results.jsx
│   │   │   ├── Map.jsx
│   │   │   └── ...
│   │   ├── pages/
│   │   ├── services/
│   │   │   └── api.js              # API client
│   │   └── styles/
│   ├── package.json
│   ├── tailwind.config.js
│   └── .env.example
│
├── database/
│   ├── schema.sql                  # Database schema
│   └── wheat_disease.db            # SQLite database
│
├── scripts/                        # Utility scripts
│   ├── download_datasets.py        # Fetch data from sources
│   ├── prepare_data.py             # Preprocessing
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Model evaluation
│   ├── deploy_model.py             # Model export
│   └── test_inference.py           # Quick inference test
│
├── tests/                          # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_inference.py
│   └── test_api.py
│
├── notebooks/                      # Jupyter notebooks (optional)
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_training.ipynb           # Training experiments
│   └── 03_evaluation.ipynb         # Evaluation metrics
│
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_DOCS.md
│   ├── DEPLOYMENT.md
│   └── TROUBLESHOOTING.md
│
└── .gitignore
```

---

## 24-Hour Timeline

### Phase 1: Setup & Data (0-6 Hours)

**Goal:** Prepare data, download pre-trained models, set up training environment

```
Hour 0-1:
├── Clone repo / Create directory structure
├── Set up virtual environment (venv)
├── Install dependencies (PyTorch, FastAPI, React, etc.)
└── Download pre-trained EfficientNetB0 weights

Hour 1-2:
├── Download datasets from Kaggle:
│   ├── "Wheat Leaf Disease Dataset" (~1000 images)
│   └── "PlantVillage" - extract wheat subset (~500 images)
├── Extract and organize into data/raw/
└── Verify file structure matches config

Hour 2-4:
├── Implement data augmentation pipeline (augmentation.py)
├── Create train/val/test splits (60/20/20)
├── Verify no data leakage or duplicate images
├── Load sample batch and visualize augmented images
└── Sanity check: batch shapes, labels, normalization ranges

Hour 4-6:
├── Set up training infrastructure:
│   ├── Implement Focal Loss (no standard CrossEntropy)
│   ├── Implement metrics: Accuracy, Precision, Recall, F1
│   ├── Create learning rate scheduler + early stopping
│   └── Create checkpoint saving system
├── Test training loop on 1 epoch (verify no crashes)
└── Set up TensorBoard logging
```

**Deliverables:**
- Clean data folder with 2000+ images
- Working DataLoader with augmentation
- Training loop can run without errors
- First epoch completes in <30 min on GPU

---

### Phase 2: Model Training & API (6-12 Hours)

**Goal:** Train both model stages, build FastAPI backend

```
Hour 6-9: Train Stage 1 (Healthy vs Diseased)
├── Start training Stage 1 on GPU
│   ├── Target: 92%+ accuracy
│   ├── Expected time: ~2-3 hours on V100
│   └── Save checkpoint every epoch
├── Monitor on TensorBoard
├── While training, implement FastAPI structure:
│   ├── Set up main FastAPI app
│   ├── Implement image quality checks (blur, brightness)
│   ├── Implement minimal preprocessing
│   └── Create /health endpoint
└── Test API locally (unit tests)

Hour 9-10: Evaluate Stage 1
├── Run on test set
├── Generate confusion matrix, precision-recall curves
├── Verify confidence distribution
├── If accuracy <85%, train Stage 2 anyway (time pressure)

Hour 10-12: Train Stage 2 (Disease Classification) + API Integration
├── Start Stage 2 training
│   ├── Target: 88%+ accuracy (harder task)
│   ├── Expected time: ~2-3 hours on V100
│   └── Use Stage 1 weights as initialization (transfer learning)
├── Simultaneously, implement:
│   ├── Two-stage inference pipeline
│   ├── Confidence threshold logic
│   ├── /predict endpoint
│   ├── Error handling (blurry images, etc.)
│   └── Response formatting (Pydantic models)
├── Create simple test client to verify API works
└── Deploy to localhost:8000
```

**Deliverables:**
- Stage 1 model saved (95K file)
- Stage 2 model saved (95K file)
- FastAPI running locally with /predict endpoint
- Quick test: upload image → get disease prediction

---

### Phase 3: Frontend & Integration (12-18 Hours)

**Goal:** Build React UI, connect to API, add mapping

```
Hour 12-14: Frontend Scaffold + Upload
├── Create React app (or use Vite for speed)
├── Build ImageUploader component
│   ├── Drag-and-drop file input
│   ├── Live preview
│   ├── Add location capture (geolocation API)
│   └── Error messages for invalid files
├── Set up API client (axios/fetch)
├── Create Results component
│   ├── Disease badge with color coding
│   ├── Confidence bar
│   └── Basic recommendations display
└── Connect to FastAPI backend

Hour 14-16: Results Display + Recommendations
├── Integrate DecisionEngine (backend)
├── Format recommendations in Results component:
│   ├── Treatment suggestions
│   ├── Prevention tips
│   ├── Risk level indicator
│   └── Urgency messaging
├── Add "Retake" flow for low quality/confidence
├── Add loading/error states
├── Style with Tailwind CSS (use templates, don't craft from scratch)
└── Test end-to-end: upload → predict → display

Hour 16-18: Mapping + Polish
├── Implement DiseaseMapper (backend)
│   ├── Store predictions + locations in SQLite
│   ├── Query heatmap data endpoint
│   └── Regional statistics
├── Add Map component (Leaflet.js)
│   ├── Basic heatmap layer
│   ├── Legend with disease colors
│   └── Zoom to user location
├── Add Share button (Social/Email)
├── Responsive design for mobile
└── Bug fixes + UX polish
```

**Deliverables:**
- React app running locally
- Full end-to-end workflow: upload → prediction → recommendations → map
- Mobile-responsive design
- No critical bugs

---

### Phase 4: Testing, Optimization & Presentation (18-24 Hours)

```
Hour 18-20: Testing & Bug Fixes
├── Test with 20-30 real images (get from field if possible)
├── Edge case testing:
│   ├── Very dark images → quality check catches them
│   ├── Very blurry images → quality check catches them
│   ├── Unknown diseases → defaults to "Check with expert"
│   ├── Network errors → graceful fallback
│   └── Large uploads → timeouts handled
├── Load testing (multiple simultaneous requests)
├── Database integrity checks (no duplicates, orphaned records)
└── Fix critical bugs

Hour 20-22: Documentation & Deployment
├── Write README with:
│   ├── 1-minute quick start guide
│   ├── Architecture diagram (text form)
│   ├── Model accuracy metrics
│   ├── Known limitations
│   └── Future improvements
├── Create simple deployment guide:
│   ├── Docker containerization (optional, if time)
│   ├── Usage on local machine
│   ├── Environment variables setup
│   └── Database initialization
├── Export model weights + save in git-lfs
├── Prepare demo images for live demo
└── Test deployment one final time

Hour 22-24: Demo & Presentation
├── Record short video demo (2-3 min) or prepare live demo
├── Create 2-minute pitch:
│   ├── Problem statement
│   ├── Solution
│   ├── Key innovation (two-stage + minimal preprocessing)
│   ├── Results (accuracy metrics)
│   └── Real-world impact
├── Prepare to answer questions:
│   ├── Why not use heavier preprocessing?
│   ├── How does it handle different lighting?
│   ├── Scalability / deployment options?
│   ├── What about offline inference?
│   └── Next steps?
├── Time check: ensure presentation < 5 min
└── Final code cleanup + commit
```

**Deliverables:**
- Clean GitHub repo
- Working deployed system (local or cloud)
- 5-minute presentation deck
- Live demo or pre-recorded video

---

## Common Pitfalls

### ❌ Pitfall 1: Aggressive Preprocessing Destroys Disease Features

**Problem:**
```python
# DON'T DO THIS!
image = cv2.GaussianBlur(image, (5, 5), 0)  # Destroys rust pustules
image = cv2.equalizeHist(image)               # Flattens disease texture
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Removes details
```

**Why it fails:**
- Disease features (rust spots, blight lesions, powdery coating) are ON the leaf surface
- Heavy preprocessing "smooths them away"
- Model can't see subtle signs → high false negatives

**✓ Solution:**
```python
# DO THIS - minimal preprocessing
# Only during INFERENCE:
image = cv2.resize(image, (224, 224))       # Just resize
image = image / 255.0                        # Normalize to 0-1
image = (image - mean) / std                 # ImageNet normalization

# Heavy augmentation ONLY during TRAINING to handle noise naturally
```

---

### ❌ Pitfall 2: Class Imbalance Destroys Minority Classes

**Problem:**
```python
# Standard training ignores rare classes
train_loader = DataLoader(dataset, shuffle=True, batch_size=32)
# Result: Model sees Powdery_Mildew only 2-3 times per epoch
# → Predicts "Rust" for everything
```

**Why it fails:**
- Rare diseases get insufficient gradient signal
- Model optimization favors majority class
- Recall for minority classes crashes to <30%

**✓ Solution:**
```python
# Use WeightedRandomSampler
class_weights = torch.tensor([1.0, 1.5, 1.8, 1.2])  # Powdery: highest
sampler = WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(dataset),
    replacement=True
)
train_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# ALSO use Focal Loss (not CrossEntropy)
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
```

---

### ❌ Pitfall 3: Model Overfits to Training Set

**Problem:**
```python
# Train for 100 epochs
# Epoch 20: val_loss=0.1 (great!)
# Epoch 100: val_loss=0.8 (overfitting!)
# Model memorized training examples, fails on new images
```

**Why it fails:**
- Limited training data (2000 images) → easily memorized
- No regularization → overfitting
- Model learns noise instead of patterns

**✓ Solution:**
```python
# 1. Early stopping
early_stopping = EarlyStopping(
    metric='val_loss',
    patience=5,           # Stop if no improvement for 5 epochs
    min_delta=0.001
)

# 2. Strong L2 regularization
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.001, 
                             weight_decay=1e-4)  # L2 penalty

# 3. Dropout in model
nn.Dropout(p=0.4)  # Drop 40% of neurons during training

# 4. Data augmentation (makes training set effectively infinite)
# ↑ Already implemented

# 5. Monitor train vs val loss ratio
# If val_loss grows while train_loss shrinks → overfitting
```

---

### ❌ Pitfall 4: Image Quality Checks Are Missing

**Problem:**
```python
# User uploads very blurry image
# Model gives 87% confident "Rust" prediction
# But it's just noise!

# Result: Farmer spreads fungicide unnecessarily
```

**Why it fails:**
- Model trained on clear images
- Doesn't understand its own uncertainty
- In production, should reject low-quality inputs

**✓ Solution:**
```python
def check_image_quality(image):
    # Laplacian variance test
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < 100:  # Too blurry
        return False, "Image is too blurry. Please retake."
    
    # Brightness check
    brightness = np.mean(gray)
    if brightness < 40:
        return False, "Image too dark. Try better lighting."
    
    return True, None

# Use in API
quality_ok, error = check_image_quality(image)
if not quality_ok:
    return HTTPException(400, error)  # Ask user to retake
```

---

### ❌ Pitfall 5: Confidence Threshold Too Low

**Problem:**
```python
# Accept any prediction >= 50% confidence
if prediction_confidence > 0.5:
    provide_recommendations()

# Result: 50% confidence means model is guessing!
# Farmer acts on random predictions
```

**Why it fails:**
- Multi-class classification needs higher confidence
- 25% is random chance (4 classes)
- Below 70% = model is uncertain

**✓ Solution:**
```python
# Set minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.65  # Only accept 65%+ confidence

if prediction_confidence < CONFIDENCE_THRESHOLD:
    return {
        'error': 'Confidence too low',
        'suggestion': 'Please retake image or consult agricultural expert',
        'confidence': prediction_confidence
    }

# For urgent recommendations (treatment):
# if confidence < 0.75: "Consult with expert before applying fungicide"
```

---

### ❌ Pitfall 6: No Confusion Between Similar Diseases

**Problem:**
```python
# Model confuses "Powdery_Mildew" with "Spot_Blotch"
# Both look like white/gray spots on leaves
# Model picks wrong one → wrong treatment

# Result: Fungicide A doesn't work (needed B)
# Farmer wastes money + time
```

**Why it fails:**
- Diseases can look similar
- Model is single-class classifier
- No decision uncertainty handling

**✓ Solution:**
```python
# 1. Look at confusion matrix during training
# If high confusion between A-B, note it in API

# 2. Return top-3 predictions with probabilities
predict_response = {
    'primary_disease': 'Powdery_Mildew',  # Highest confidence
    'confidence': 0.68,
    'alternative_diseases': [
        {'disease': 'Spot_Blotch', 'confidence': 0.22},
        {'disease': 'Rust', 'confidence': 0.08}
    ],
    'recommendation': 'High confidence in Powdery_Mildew. If treatment fails in 5 days, consider Spot_Blotch treatment.'
}

# 3. Always suggest: "If no improvement in 5 days, consult expert"
```

---

### ❌ Pitfall 7: Training Loop Crashes After 2 Hours

**Problem:**
```python
# GPU runs out of memory mid-training
# Or: Model weights NaN (gradient explosion)
# Loss: Hours of training wasted

# Result: No trained model, no system to demo
```

**Why it fails:**
- Batch size too large
- Learning rate too high → gradients explode
- Memory leak in training loop

**✓ Solution:**
```python
# 1. Gradient clipping (prevent explosion)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower batch size
batch_size = 32  # Not 128!

# 3. Lower learning rate
lr = 0.001  # Not 0.1!

# 4. Reduce LR further for fine-tuning
lr_stage2 = 0.0005  # Half of Stage 1

# 5. Save checkpoints frequently
if epoch % 2 == 0:
    torch.save(model.state_dict(), f'models/stage1_epoch_{epoch}.pth')

# 6. Monitor GPU memory
# If >90% used: reduce batch size immediately

# 7. Test on small subset first
train_small = DataLoader(dataset[:100], batch_size=32)
for epoch in range(1):
    train_epoch(model, train_small, ...)
# If this works, scale up
```

---

### ❌ Pitfall 8: Location Data Creates Privacy Issues

**Problem:**
```python
# Store farmer name + exact GPS coordinates
# Data leaked → competitor tracks farms
# Or: Farmer data sold to insurance company
```

**Why it fails:**
- Privacy regulations (GDPR, India laws)
- Farmer trust lost
- Legal liability

**✓ Solution:**
```python
# 1. Make location optional
@app.post("/predict")
async def predict(file: UploadFile, location: Optional[LocationData] = None):
    # Don't require location

# 2. Store location with salt/hash
import hashlib
location_hash = hashlib.sha256(
    f"{lat},{lon}".encode()
).hexdigest()[:8]
# Store hash, not exact coordinates

# 3. Add data expiration
stored_predictions.created_at + timedelta(days=30)
# Auto-delete after 30 days

# 4. Never store farmer name publicly
# Store as UUID if needed for tracking
```

---

### ❌ Pitfall 9: Model Can't Run on CPU

**Problem:**
```python
# Model trained on GPU
# GPU not available in production
# Script crashes: "CUDA out of memory" → can't

fallback to CPU
```

**Why it fails:**
- Model assumes GPU
# If farmer uses phone/cloud without GPU → system fails
- Offline deployment impossible

**✓ Solution:**
```python
# Write device-agnostic code from start
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(DEVICE)
image_tensor = image_tensor.to(DEVICE)

# Test CPU path early
# Run inference on CPU manually
test_image = ...
result = model(test_image.to('cpu'))
# Verify it works before training completes

# Use quantization for faster CPU inference
from torch.quantization import quantize_dynamic
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear},
    dtype=torch.qint8
)
# Reduces model size 4x, runs faster on CPU
```

---

### ❌ Pitfall 10: API Doesn't Handle Concurrent Requests

**Problem:**
```python
# Two farmers upload simultaneously
# Both wait for same GPU inference
# Server becomes bottleneck

# Result: 10-second delays → users think it's broken
```

**Why it fails:**
- PyTorch inference not thread-safe by default
- FastAPI tries to handle both → race conditions
- Queued requests pile up

**✓ Solution:**
```python
# Use async I/O + thread pool for inference
from concurrent.futures import ThreadPoolExecutor
import asyncio

inference_executor = ThreadPoolExecutor(max_workers=2)

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    
    # Offload inference to thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        inference_executor,
        run_inference_sync,
        image_bytes
    )
    
    return result

def run_inference_sync(image_bytes):
    # Actual inference (runs in thread pool)
    with torch.no_grad():
        return model(tensor)

# OR: Use Gunicorn with multiple workers
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

---

## Deployment Guide

### Local Deployment (Fastest for Hackathon)

```bash
# 1. Clone repo
git clone https://github.com/your-name/wheat-disease-detection
cd wheat-disease-detection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained models
python scripts/download_models.py  # Adds models/ folder with weights

# 5. Initialize database
python scripts/init_db.py

# 6. Start backend
python -m uvicorn src.api.main:app --reload --port 8000

# 7. Start frontend (in new terminal)
cd frontend
npm install
npm start

# 8. Open browser
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Docker Deployment (Production-Ready)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY models/ models/
COPY database/ database/

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t wheat-disease .
docker run -p 8000:8000 wheat-disease
```

### Cloud Deployment (AWS/GCP/Azure)

**Option 1: AWS EC2**
```bash
# 1. Launch t3.medium instance (free tier)
# 2. SSH into instance
# 3. Follow local deployment steps
# 4. Set security group to allow ports 80, 443, 3000, 8000
# 5. Use nginx to reverse proxy
```

**Option 2: Google Cloud Run (Serverless)**
```bash
gcloud run deploy wheat-disease \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300
```

---

## Summary: Key Success Factors

1. **Don't Overengineer**: Two-stage model beats five-stage complexity
2. **Minimal Preprocessing**: Let augmentation handle robustness in training
3. **Quality Checks**: Never trust user uploads blindly
4. **Handle Uncertainty**: Confidence threshold + ask for retake
5. **Simple Recommendations**: Rule-based engine, not another ML model
6. **Fast Iteration**: Prioritize E2E demo over perfection
7. **Test Offline**: Ensure works on CPU
8. **Mobile First**: Farmers use phones
9. **Clear Error Messages**: "Image too blurry" beats technical jargon
10. **Document Everything**: Future developers (and judges!) need context

---

## References & Resources

### Datasets
- Kaggle Wheat Leaf Disease: https://kaggle.com/datasets/...
- PlantVillage: https://github.com/spMohanty/PlantVillage-Dataset
- Your own field images (best for real-world testing)

### Model References
- EfficientNet: https://arxiv.org/abs/1905.11946
- Focal Loss: https://arxiv.org/abs/1708.02002
- Two-stage classifiers: https://arxiv.org/abs/1611.04797

### Frameworks
- PyTorch: https://pytorch.org
- FastAPI: https://fastapi.tiangolo.com
- React: https://react.dev
- Leaflet.js: https://leafletjs.com

### Agricultural Information
- Indian Ministry of Agriculture: https://agricoop.nic.in/
- ICRISAT (International Crops Research): https://www.icrisat.org/
- Wheat disease diagnosis guides (local agricultural college)

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-31  
**Author:** AI Architecture Team  
**Status:** Production-Ready for Hackathon

---

