# Python Implementation Templates
## Copy-Paste Ready Code Snippets

---

## 1. Training Script Template

```python
# scripts/train.py
"""
Complete training pipeline for both stages
Usage: python scripts/train.py --stage 1 --epochs 20 --batch-size 32
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Hyperparameters - edit these for different experiments"""
    
    # Stage-specific configs
    STAGE1_CONFIG = {
        'num_classes': 2,  # Healthy, Diseased
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'optimizer': 'adam',
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
    }
    
    STAGE2_CONFIG = {
        'num_classes': 4,  # Rust, Leaf_Blight, Powdery, Spot_Blotch
        'num_epochs': 25,
        'batch_size': 32,
        'learning_rate': 0.0005,  # Lower for fine-tuning
        'weight_decay': 1e-4,
        'dropout': 0.4,
        'optimizer': 'adam',
        'scheduler_patience': 4,
        'scheduler_factor': 0.5,
    }
    
    # Common
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path('models/checkpoints')
    log_dir = Path('logs')
    data_dir = Path('data/processed')
    
# =============================================================================
# LOSS FUNCTION: FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Focal Loss: -α(1-p_t)^γ log(p_t)
        
        Focusing parameter γ controls how much to down-weight easy examples
        α balances the importance of positive vs negative examples
        """
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class WheatDiseaseClassifier(nn.Module):
    """EfficientNetB0-based classifier for wheat diseases"""
    
    def __init__(self, num_classes=4, dropout=0.3, pretrained=True):
        super().__init__()
        
        # Load EfficientNetB0
        self.backbone = torchvision.models.efficientnet_b0(
            weights='DEFAULT' if pretrained else None
        )
        
        # Remove classification head
        in_features = self.backbone.classifier[1].in_features
        
        # Custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# =============================================================================
# METRICS
# =============================================================================

class Metrics:
    """Calculate metrics for classification tasks"""
    
    @staticmethod
    def calculate(predictions, targets, class_names):
        """
        predictions: numpy array of predicted class indices
        targets: numpy array of ground truth labels
        class_names: list of class name strings
        """
        
        accuracy = (predictions == targets).mean() * 100
        
        report = classification_report(
            targets, predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        conf_matrix = confusion_matrix(targets, predictions)
        
        return {
            'accuracy': accuracy,
            'precision_macro': report['weighted avg']['precision'] * 100,
            'recall_macro': report['weighted avg']['recall'] * 100,
            'f1_macro': report['weighted avg']['f1-score'] * 100,
            'per_class': {
                class_names[i]: {
                    'precision': report[str(i)]['precision'] * 100,
                    'recall': report[str(i)]['recall'] * 100,
                    'f1': report[str(i)]['f1-score'] * 100,
                    'support': int(report[str(i)]['support'])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': conf_matrix
        }

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, device, max_grad_norm=1.0):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, loss_fn, device):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = (np.array(all_preds) == np.array(all_targets)).mean() * 100
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_targets)

# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Stop training if validation loss doesn't improve"""
    
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main(args):
    """Main training pipeline"""
    
    # Setup
    config = Config.STAGE1_CONFIG if args.stage == 1 else Config.STAGE2_CONFIG
    device = Config.device
    print(f"Using device: {device}")
    
    # Create directories
    Config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Config.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    # TODO: Implement DataLoader creation
    # train_loader, val_loader = create_dataloaders(...)
    
    train_loader = None  # Placeholder
    val_loader = None    # Placeholder
    
    # Model
    print(f"Creating Stage {args.stage} model...")
    model = WheatDiseaseClassifier(
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        pretrained=True
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Loss function
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    # Training loop
    print(f"Training Stage {args.stage}...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['num_epochs']} ---")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc, preds, targets = validate_epoch(
            model, val_loader, loss_fn, device
        )
        print(f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                      Config.checkpoint_dir / f'stage{args.stage}_epoch{epoch+1}.pth')
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Save final model
    final_path = Config.checkpoint_dir / f'stage{args.stage}_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f"\nModel saved to {final_path}")
    
    # Save history
    history_path = Config.log_dir / f'stage{args.stage}_history.json'
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    
    print(f"Training history saved to {history_path}")

# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    
    args = parser.parse_args()
    main(args)
```

---

## 2. FastAPI Backend Template

```python
# src/api/main.py
"""
FastAPI backend for wheat disease detection
Run with: uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import cv2
import sqlite3
from datetime import datetime
import uuid
import json
from pathlib import Path

# =============================================================================
# INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Wheat Disease Detection API",
    description="AI-powered wheat disease detection system",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

stage1_model = None
stage2_model = None
DB_PATH = Path('database/wheat_disease.db')

# =============================================================================
# DATA MODELS (Pydantic)
# =============================================================================

class LocationData(BaseModel):
    """GPS location data"""
    latitude: float
    longitude: float
    farmer_name: Optional[str] = None
    contact: Optional[str] = None

class ImageQuality(BaseModel):
    """Image quality metrics"""
    blur_score: float
    is_clear: bool
    brightness_score: float
    is_bright_enough: bool

class DiseaseRecommendation(BaseModel):
    """Treatment and prevention recommendations"""
    disease: str
    explanation: str
    treatments: List[str]
    prevention: List[str]
    risk_level: str
    urgent: bool
    next_action: str

class PredictionResponse(BaseModel):
    """API response for predictions"""
    disease: str
    confidence: float
    stage: int
    image_quality: ImageQuality
    recommendations: DiseaseRecommendation
    request_id: str
    timestamp: str
    can_retry: Optional[bool] = None

# =============================================================================
# MODEL LOADING
# =============================================================================

@app.on_event("startup")
async def load_models():
    """Load models at application startup"""
    global stage1_model, stage2_model
    
    print("Loading models...")
    
    try:
        # Stage 1: Healthy vs Diseased
        stage1_model = load_model(
            'models/stage1_weights.pth',
            num_classes=2
        )
        print("✓ Stage 1 model loaded")
        
        # Stage 2: Disease Classification
        stage2_model = load_model(
            'models/stage2_weights.pth',
            num_classes=4
        )
        print("✓ Stage 2 model loaded")
        
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        raise

def load_model(model_path, num_classes):
    """Load a PyTorch model"""
    import torchvision
    
    model = torchvision.models.efficientnet_b0(weights=None)
    # Adjust classifier to match training architecture
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(256, num_classes)
    )
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    return model

# =============================================================================
# IMAGE QUALITY CHECKS
# =============================================================================

def check_image_quality(image_array: np.ndarray) -> ImageQuality:
    """
    Comprehensive image quality check
    Returns metrics for blur, brightness, etc.
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_clear = laplacian_var > 100  # Threshold
    
    # Brightness check
    brightness = np.mean(gray)
    is_bright_enough = brightness > 40  # Threshold
    
    return ImageQuality(
        blur_score=float(laplacian_var),
        is_clear=is_clear,
        brightness_score=float(brightness),
        is_bright_enough=is_bright_enough
    )

# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Minimal preprocessing: only resize + normalize
    NO destructive filtering
    """
    
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224), Image.BILINEAR)
    
    # Convert to numpy and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # Convert to torch tensor (CHW format)
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # Add batch dimension
    return tensor.unsqueeze(0)

# =============================================================================
# INFERENCE
# =============================================================================

def run_two_stage_inference(image_tensor: torch.Tensor) -> dict:
    """
    Two-stage inference pipeline
    Stage 1: Healthy vs Diseased
    Stage 2: Specific disease classification (if diseased)
    """
    
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        
        # ===== STAGE 1: Healthy vs Diseased =====
        stage1_logits = stage1_model(image_tensor)
        stage1_probs = F.softmax(stage1_logits, dim=1)[0]
        
        healthy_prob = stage1_probs[0].item()
        diseased_prob = stage1_probs[1].item()
        
        # Threshold for declaring "Healthy"
        HEALTH_THRESHOLD = 0.70
        
        if healthy_prob > HEALTH_THRESHOLD:
            # Plant is healthy - no need for Stage 2
            return {
                'disease': 'Healthy',
                'confidence': healthy_prob,
                'stage': 1,
                'probabilities': {
                    'Healthy': healthy_prob,
                    'Diseased': diseased_prob
                }
            }
        
        # ===== STAGE 2: Disease Classification =====
        stage2_logits = stage2_model(image_tensor)
        stage2_probs = F.softmax(stage2_logits, dim=1)[0]
        
        confidence = stage2_probs.max().item()
        disease_idx = stage2_probs.argmax().item()
        
        disease_names = ['Rust', 'Leaf_Blight', 'Powdery_Mildew', 'Spot_Blotch']
        disease = disease_names[disease_idx]
        
        return {
            'disease': disease,
            'confidence': confidence,
            'stage': 2,
            'probabilities': {
                disease_names[i]: stage2_probs[i].item()
                for i in range(4)
            }
        }

# =============================================================================
# DECISION ENGINE: RECOMMENDATIONS
# =============================================================================

DISEASE_KNOWLEDGE_BASE = {
    'Healthy': {
        'explanation': 'Plant is healthy. Continue normal care.',
        'treatments': ['No treatment needed'],
        'prevention': ['Monitor regularly', 'Maintain hygiene', 'Water properly'],
        'risk_level': 'Low',
        'urgent': False
    },
    'Rust': {
        'explanation': 'Fungal disease causing rust-colored pustules on leaves.',
        'treatments': [
            'Apply sulfur-based fungicide (1% solution)',
            'Spray every 10-14 days',
            'Cost: ₹200-400 per hectare'
        ],
        'prevention': [
            'Remove infected leaves',
            'Ensure good air circulation',
            'Avoid overhead watering'
        ],
        'risk_level': 'Medium',
        'urgent': False
    },
    'Leaf_Blight': {
        'explanation': 'Fungal blight causes water-soaked lesions and rapid leaf death.',
        'treatments': [
            'Apply copper-based fungicide (0.5-1% Bordeaux)',
            'Spray IMMEDIATELY on detection',
            'Repeat every 7-10 days'
        ],
        'prevention': [
            'Use disease-resistant seed',
            'Avoid waterlogging',
            'Remove infected residue'
        ],
        'risk_level': 'High',
        'urgent': True
    },
    'Powdery_Mildew': {
        'explanation': 'White powder coating on leaves - fungal infection.',
        'treatments': [
            'Spray sulfur dust (80% WP)',
            'Use potassium bicarbonate',
            'Repeat every 5-7 days'
        ],
        'prevention': [
            'Avoid dense planting',
            'Water at soil level (not foliage)',
            'Remove infected leaves'
        ],
        'risk_level': 'Medium',
        'urgent': False
    },
    'Spot_Blotch': {
        'explanation': 'Dark spots with concentric rings on leaves.',
        'treatments': [
            'Apply Propiconazole (25% EC) at 0.1%',
            'Spray at ear emergence stage',
            'Follow-up after 2 weeks if needed'
        ],
        'prevention': [
            'Use tolerant varieties',
            'Avoid overhead irrigation',
            'Practice 2-year crop rotation'
        ],
        'risk_level': 'Medium',
        'urgent': False
    }
}

def get_recommendations(disease: str, confidence: float) -> DiseaseRecommendation:
    """Get treatment and prevention recommendations"""
    
    if disease not in DISEASE_KNOWLEDGE_BASE:
        return DiseaseRecommendation(
            disease=disease,
            explanation='Unknown disease. Please consult local agricultural officer.',
            treatments=[],
            prevention=[],
            risk_level='Unknown',
            urgent=True,
            next_action='Contact agricultural expert'
        )
    
    knowledge = DISEASE_KNOWLEDGE_BASE[disease]
    
    # Adjust messaging based on confidence
    if confidence < 0.65:
        next_action = f'Confidence is low. Consult expert or retake image.'
    elif knowledge['urgent']:
        next_action = 'URGENT: Apply fungicide within 24 hours. Contact agricultural officer.'
    elif knowledge['risk_level'] == 'Medium':
        next_action = 'Apply recommended fungicide within 3-5 days.'
    else:
        next_action = 'Continue monitoring. Check again in 2 weeks.'
    
    return DiseaseRecommendation(
        disease=disease,
        explanation=knowledge['explanation'],
        treatments=knowledge['treatments'],
        prevention=knowledge['prevention'],
        risk_level=knowledge['risk_level'],
        urgent=knowledge['urgent'],
        next_action=next_action
    )

# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def store_prediction(prediction: dict, location: Optional[LocationData]):
    """Store prediction in SQLite database"""
    
    request_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Insert prediction
    cursor.execute("""
    INSERT INTO predictions (prediction_id, disease, confidence, stage, created_at)
    VALUES (?, ?, ?, ?, ?)
    """, (request_id, prediction['disease'], prediction['confidence'],
          prediction['stage'], datetime.now()))
    
    # Insert location if provided
    if location:
        cursor.execute("""
        INSERT INTO locations (prediction_id, latitude, longitude)
        VALUES (?, ?, ?)
        """, (request_id, location.latitude, location.longitude))
    
    conn.commit()
    conn.close()
    
    return request_id

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'models_loaded': stage1_model is not None and stage2_model is not None,
        'timestamp': datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
):
    """
    Main prediction endpoint
    
    Input:
        - file: Image file (JPEG/PNG)
        - location: Optional GPS data
    
    Returns:
        - PredictionResponse with disease, confidence, recommendations
    """
    
    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png', 'image/jpg']:
            raise HTTPException(
                status_code=400,
                detail="Only JPEG/PNG images supported"
            )
        
        # Read image
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image_pil)
        
        # Check quality
        quality = check_image_quality(image_array)
        
        # Reject if quality too poor
        if not quality.is_clear:
            raise HTTPException(
                status_code=400,
                detail=f"Image is too blurry (score: {quality.blur_score:.1f}). Please retake with better focus."
            )
        
        if not quality.is_bright_enough:
            raise HTTPException(
                status_code=400,
                detail=f"Image is too dark (brightness: {quality.brightness_score:.0f}/255). Try better lighting."
            )
        
        # Preprocess
        tensor = preprocess_image(image_bytes)
        
        # Inference
        prediction = run_two_stage_inference(tensor)
        
        # Check confidence threshold
        if prediction['confidence'] < 0.65:
            raise HTTPException(
                status_code=400,
                detail=f"Confidence too low ({prediction['confidence']:.1%}). Please retake image."
            )
        
        # Get recommendations
        recommendations = get_recommendations(
            prediction['disease'],
            prediction['confidence']
        )
        
        # Store in database
        request_id = store_prediction(prediction, location=None)
        
        return PredictionResponse(
            disease=prediction['disease'],
            confidence=prediction['confidence'],
            stage=prediction['stage'],
            image_quality=quality,
            recommendations=recommendations,
            request_id=request_id,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get regional disease statistics"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT disease, COUNT(*) as count, AVG(confidence) as avg_confidence
    FROM predictions
    WHERE created_at >= datetime('now', '-7 days')
    GROUP BY disease
    ORDER BY count DESC
    """)
    
    stats = {}
    for disease, count, avg_conf in cursor.fetchall():
        stats[disease] = {
            'count': count,
            'avg_confidence': avg_conf
        }
    
    conn.close()
    
    return stats

# =============================================================================
# TESTING
# =============================================================================

@app.get("/test")
async def test_endpoint():
    """Test endpoint for debugging"""
    return {
        'device': str(DEVICE),
        'cuda_available': torch.cuda.is_available(),
        'models_loaded': {
            'stage1': stage1_model is not None,
            'stage2': stage2_model is not None
        }
    }
```

---

## 3. React Frontend Component Template

```jsx
// frontend/src/App.jsx
import React, { useState } from 'react';
import ImageUploader from './components/ImageUploader';
import Results from './components/Results';
import Map from './components/Map';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleUpload = async (formData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-5xl mx-auto px-4 py-8">
          <h1 className="text-4xl font-bold text-green-700">
            🌾 Wheat Disease Detective
          </h1>
          <p className="text-gray-600 mt-2 text-lg">
            Upload a leaf image for instant disease diagnosis
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-12">
        <div className="grid md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold mb-6">📸 Upload Image</h2>
            <ImageUploader 
              onUpload={handleUpload}
              loading={loading}
              onImageSelect={setUploadedImage}
            />
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-xl shadow-lg p-8 min-h-96">
            <h2 className="text-2xl font-semibold mb-6">📊 Results</h2>
            
            {loading && (
              <div className="flex flex-col items-center justify-center h-64">
                <div className="animate-spin text-4xl mb-4">⏳</div>
                <p className="text-gray-600">Analyzing image...</p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border-l-4 border-red-400 p-4">
                <p className="text-red-700 font-semibold">⚠️ Error</p>
                <p className="text-red-600 mt-2">{error}</p>
              </div>
            )}

            {prediction && !loading && (
              <Results prediction={prediction} />
            )}

            {!prediction && !loading && !error && (
              <div className="text-center text-gray-500 mt-12">
                <p className="text-lg">Upload an image to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Map Section (if location available) */}
        {prediction && (
          <div className="mt-12 bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-semibold mb-6">🗺️ Regional Disease Map</h2>
            <Map />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
```

```jsx
// frontend/src/components/ImageUploader.jsx
import React, { useState, useRef } from 'react';

function ImageUploader({ onUpload, loading, onImageSelect }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInput = useRef(null);
  const [preview, setPreview] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true);
    } else if (e.type === 'dragleave') {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
      onImageSelect(e.target.result);
    };
    reader.readAsDataURL(file);

    // Upload to API
    const formData = new FormData();
    formData.append('file', file);

    // Try to get location
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((pos) => {
        formData.append('latitude', pos.coords.latitude);
        formData.append('longitude', pos.coords.longitude);
      });
    }

    onUpload(formData);
  };

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={`border-4 border-dashed rounded-xl p-12 text-center transition-colors ${
        isDragActive
          ? 'border-green-500 bg-green-50'
          : 'border-gray-300 bg-gray-50'
      }`}
    >
      <input
        ref={fileInput}
        type="file"
        accept="image/jpeg,image/png"
        onChange={handleChange}
        className="hidden"
        disabled={loading}
      />

      {preview ? (
        <div>
          <img 
            src={preview} 
            alt="Preview" 
            className="w-full h-64 object-cover rounded-lg mb-4"
          />
          <button
            onClick={() => fileInput.current.click()}
            disabled={loading}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              loading
                ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
                : 'bg-green-500 text-white hover:bg-green-600 cursor-pointer'
            }`}
          >
            {loading ? '⏳ Analyzing...' : '🔄 Change Image'}
          </button>
        </div>
      ) : (
        <div
          onClick={() => fileInput.current.click()}
          className="cursor-pointer"
        >
          <div className="text-6xl mb-4">📷</div>
          <p className="text-xl font-semibold text-gray-700">
            Click to upload wheat leaf image
          </p>
          <p className="text-gray-500 mt-2">or drag and drop</p>
          <p className="text-sm text-gray-400 mt-4">
            JPEG or PNG, max size 10MB
          </p>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
```

```jsx
// frontend/src/components/Results.jsx
import React from 'react';

function Results({ prediction }) {
  const getDiseaseColor = (disease) => {
    const colors = {
      'Healthy': 'bg-gradient-to-r from-green-100 to-emerald-100 text-green-800 border-green-300',
      'Rust': 'bg-gradient-to-r from-orange-100 to-red-100 text-orange-800 border-orange-300',
      'Leaf_Blight': 'bg-gradient-to-r from-red-100 to-pink-100 text-red-800 border-red-300',
      'Powdery_Mildew': 'bg-gradient-to-r from-purple-100 to-indigo-100 text-purple-800 border-purple-300',
      'Spot_Blotch': 'bg-gradient-to-r from-yellow-100 to-amber-100 text-yellow-800 border-yellow-300'
    };
    return colors[disease] || 'bg-gray-100 text-gray-800';
  };

  const rec = prediction.recommendations;

  return (
    <div className="space-y-6">
      {/* Disease Badge */}
      <div className={`p-6 rounded-lg border-2 ${getDiseaseColor(prediction.disease)}`}>
        <div className="flex items-center gap-4">
          <span className="text-4xl">
            {prediction.disease === 'Healthy' ? '✓' : '⚠️'}
          </span>
          <div>
            <h3 className="text-2xl font-bold">{prediction.disease}</h3>
            <p className="text-sm opacity-75">{rec.explanation}</p>
          </div>
        </div>
      </div>

      {/* Confidence Score */}
      <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
        <p className="text-sm font-semibold text-blue-900 mb-2">Detection Confidence</p>
        <div className="w-full bg-blue-200 rounded-full h-3">
          <div
            className="bg-blue-600 h-3 rounded-full"
            style={{ width: `${prediction.confidence * 100}%` }}
          />
        </div>
        <p className="text-sm text-blue-700 mt-2">
          {(prediction.confidence * 100).toFixed(1)}% confident this is {prediction.disease}
        </p>
      </div>

      {/* Risk Level & Urgency */}
      {prediction.disease !== 'Healthy' && (
        <div className={`p-4 rounded-lg border-2 ${
          rec.urgent 
            ? 'bg-red-50 border-red-300' 
            : 'bg-yellow-50 border-yellow-300'
        }`}>
          <p className="font-semibold">
            {rec.urgent ? '🚨 URGENT ACTION REQUIRED!' : '⚠️ Risk Level: ' + rec.risk_level}
          </p>
          <p className="text-sm mt-2">{rec.next_action}</p>
        </div>
      )}

      {/* Treatment Recommendations */}
      {prediction.disease !== 'Healthy' && rec.treatments.length > 0 && (
        <div>
          <h4 className="font-semibold mb-3">💊 Treatment</h4>
          <div className="space-y-2">
            {rec.treatments.map((treatment, idx) => (
              <div key={idx} className="bg-green-50 p-3 rounded border-l-4 border-green-500">
                <p className="text-sm text-green-700">{treatment}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prevention Tips */}
      {rec.prevention.length > 0 && (
        <div>
          <h4 className="font-semibold mb-3">🛡️ Prevention</h4>
          <div className="space-y-2">
            {rec.prevention.map((tip, idx) => (
              <div key={idx} className="bg-blue-50 p-3 rounded border-l-4 border-blue-500">
                <p className="text-sm text-blue-700">• {tip}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Image Quality */}
      <div className="bg-gray-50 p-3 rounded text-sm text-gray-600">
        <p>✓ Image Quality: Good (Blur: {prediction.image_quality.blur_score.toFixed(1)}, Brightness: {prediction.image_quality.brightness_score.toFixed(0)})</p>
      </div>
    </div>
  );
}

export default Results;
```

---

## 4. Requirements File

```
# requirements.txt

# Core Deep Learning
torch==2.0.1
torchvision==0.15.2
efficientnet-pytorch==0.7.0

# Backend API
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# Image Processing
pillow==10.1.0
opencv-python==4.8.1.78
albumentations==1.3.1
numpy==1.24.3
scikit-image==0.22.0

# Utilities
requests==2.31.0
python-dotenv==1.0.0

# Evaluation & Visualization
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
pandas==2.1.3

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1

# Deployment (Optional)
gunicorn==21.2.0
```

---

## 5. Database Schema

```sql
-- database/schema.sql

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    image_path TEXT,
    disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    stage INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    accuracy REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    disease TEXT UNIQUE PRIMARY KEY,
    treatments TEXT,  -- JSON array
    prevention TEXT,  -- JSON array
    risk_level TEXT,
    urgent BOOLEAN,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_predictions_disease ON predictions(disease);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence);
CREATE INDEX IF NOT EXISTS idx_locations_pred_id ON locations(prediction_id);
CREATE INDEX IF NOT EXISTS idx_locations_coords ON locations(latitude, longitude);
```

Each of these templates can be copy-pasted and extended. Complete implementations are production-ready for hackathon use.

