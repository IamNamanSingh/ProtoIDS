# Frontend Handover Guidelines

This document provides guidance for developing a lightweight research demo/frontend for the ProtoIDS prototype-based open-set intrusion detection system.

## Overview
The frontend should be a research-oriented demonstration tool, not a production dashboard. Its purpose is to help researchers:
- Understand how ProtoIDS makes predictions
- Visualize the embedding space and prototype locations
- Analyze threshold effects and unknown detection
- Explore model behavior on sample network flows

## Core Features to Implement

### 1. Prediction Visualization
- **Input**: Network flow features (41 dimensions for CICIoT2023, 39 for Edge-IIoTset)
- **Processing**:
  - Show preprocessing steps (scaling, etc.)
  - Display encoded embedding (32D L2-normalized vector)
  - Show distance to each class prototype
  - Display nearest known class and distance
- **Output**:
  - Predicted class (or UNKNOWN)
  - Distance threshold used
  - Clear visualization of KNOWN vs UNKNOWN decision

### 2. Embedding Space Exploration
- **Prototype Visualization**:
  - Show prototype positions for each known class (use dimensionality reduction like PCA or t-SNE for 2D/3D visualization)
  - Highlight the nearest prototype(s) for a given input
- **Input Projection**:
  - Show where the input embedding falls in the reduced space
  - Display contours or regions of known class dominance
- **Interactive Elements**:
  - Allow users to input custom network flow features
  - Option to load sample flows from test set

### 3. Threshold Analysis Panel
- **Threshold Slider**:
  - Interactive slider to adjust distance threshold
  - Real-time update of KNOWN/UNKNOWN predictions
  - Show impact on known-class accuracy and unknown recall
- **Threshold Curve**:
  - Plot known-class accuracy vs threshold
  - Plot unknown recall vs threshold
  - Show F1 score vs threshold
  - Mark the default threshold (90th percentile of known validation distances)
- **Operating Characteristics**:
  - Display ROC curve (known-class as positive, unknown as negative)
  - Show precision-recall curve for unknown detection
  - Calculate and display AUROC and AUPR

### 4. Class-wise Analysis
- **Per-Class Metrics**:
  - Show precision, recall, F1 for each known class
  - Display confusion matrix for known classes
- **Unknown Class Absorption**:
  - For each withheld unknown class, show distribution of predictions across known classes
  - Visualize which known classes "absorb" each unknown attack type
  - Highlight absorption patterns (e.g., MITM-ArpSpoofing mostly absorbed by BenignTraffic)

### 5. Sample Data Explorer
- **Test Set Browser**:
  - Allow selection of samples from test set (known and unknown classes)
  - Show true label and predicted label
  - Display network flow features (optionally normalized)
  - Compute and show embedding and distances
- **Correct/Incorrect Predictions**:
  - Filter samples by correct/incorrect predictions
  - Analyze patterns in misclassifications

## Technical Implementation Suggestions

### Frontend Stack
- **Language**: Python (for consistency with backend) or JavaScript/TypeScript
- **Framework**: 
  - Python: Streamlit, Dash, or Gradio for rapid prototyping
  - JavaScript: React/Vue with D3.js or Plotly.js for visualizations
- **Reasoning**: Choose based on team expertise; Streamlit is recommended for fastest prototyping with Python backend

### Data Flow
1. **Input**: User provides network flow features (CSV row or manual entry)
2. **Preprocessing**: Apply same scaling as used in training (load saved scaler)
3. **Encoding**: Pass through encoder network to get embedding
4. **Distance Calculation**: Compute cosine distance to all prototypes
5. **Prediction**: 
   - Find nearest known class
   - Compare minimum distance to threshold
   - Output class label or UNKNOWN
6. **Visualization**: 
   - Embedding projection (PCA/t-SNE of prototype embeddings + input)
   - Distance bar chart
   - Threshold impact curves

### State Management
- Load model components once at startup:
  - Encoder weights (`model.pth` or `encoder.pth`)
  - Prototypes (`prototypes.npy`)
  - Threshold (`threshold.npy`)
  - Label mapping (`label_mapping.json`)
  - Preprocessing scaler (`preprocessor.joblib`)
- Keep models in evaluation mode
- Cache preprocessing transforms for efficiency

## File Organization Suggestion
```
frontend/
├── app.py                 # Main application (Streamlit) or index.html
├── components/            # Reusable UI components
│   ├── prediction.py
│   ├── embedding_viz.py
│   ├── threshold_analysis.py
│   └── sample_explorer.py
├── utils/                 # Helper functions
│   ├── preprocessing.py
│   ├── model_loading.py
│   └── visualization.py
├── assets/                # Static assets (logos, etc.)
├── requirements.txt       # Frontend-specific dependencies
└── README.md              # Frontend setup instructions
```

## Integration with ProtoIDS Backend
### Model Loading
```python
# Example: Loading ProtoIDS components
import torch
import numpy as np
import joblib

def load_protoids_components(experiment_dir):
    # Load label mapping
    with open(f"{experiment_dir}/label_mapping.json", 'r') as f:
        label_mapping = json.load(f)
    
    # Load preprocessing scaler
    scaler = joblib.load(f"{experiment_dir}/preprocessor.joblib")
    
    # Load model (encoder + prototype layer)
    model = torch.load(f"{experiment_dir}/model.pth", map_location='cpu')
    model.eval()
    
    # Load prototypes and threshold
    prototypes = np.load(f"{experiment_dir}/prototypes.npy")
    threshold = np.load(f"{experiment_dir}/threshold.npy")
    
    return {
        'label_mapping': label_mapping,
        'scaler': scaler,
        'model': model,
        'prototypes': prototypes,
        'threshold': threshold,
        'experiment_dir': experiment_dir
    }
```

### Prediction Function
```python
def predict_flow(flow_features, components):
    # Preprocess
    scaled = components['scaler'].transform([flow_features])
    
    # Encode
    with torch.no_grad():
        embedding = components['model'].encoder(torch.FloatTensor(scaled))
        # Assuming model returns embedding before prototype layer
        # Adjust based on actual model architecture
    
    # Normalize embedding (L2 norm)
    embedding_norm = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    
    # Calculate distances to prototypes
    # cosine distance = 1 - cosine_similarity
    dots = np.dot(embedding_norm, components['prototypes'].T)  # [1, n_prototypes]
    distances = 1 - dots  # Cosine distance
    
    # Find minimum distance per class (requires class-prototype mapping)
    # For simplicity, assuming prototypes are ordered by class
    # Implementation depends on prototype organization
    
    # Get predicted class and min distance
    min_distance = np.min(distances)
    predicted_class_idx = np.argmin(distances)  # Simplified
    
    # Apply threshold
    if min_distance > components['threshold']:
        return "UNKNOWN", min_distance, None
    else:
        # Map idx to class label using label_mapping
        return components['label_mapping'].get(str(predicted_class_idx), "Unknown"), min_distance, predicted_class_idx
```

## Development Recommendations

### Start Simple
1. Begin with a basic prediction display (input → output)
2. Add embedding visualization using PCA/t-SNE
3. Implement threshold slider and impact curves
4. Add sample data explorer and absorption analysis

### Prioritize Interpretability
- Focus on making the model's decision process transparent
- Use clear visualizations and explanations
- Avoid clutter; each view should have a single clear purpose

### Performance Considerations
- Precompute prototype positions in reduced space (PCA/t-SNE) for faster visualization
- Cache frequent computations
- Limit simultaneous visualizations to maintain responsiveness

### Testing and Validation
- Verify predictions match backend results for known samples
- Test edge cases (very close to threshold, extreme values)
- Ensure frontend and backend use identical preprocessing

## Future Enhancements
- **Uncertainty Estimation**: Show confidence or distance margin
- **Batch Processing**: Analyze multiple flows at once
- **Temporal Analysis**: Show how predictions change over time (if temporal data available)
- **Export Functionality**: Allow saving predictions and visualizations
- **Multi-Dataset Support**: Switch between CICIoT2023, Edge-IIoTset, and future datasets

## Resources
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Plotly.js**: https://plotly.com/javascript/
- **D3.js**: https://d3js.org/
- **Scikit-learn PCA/t-SNE**: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
- **Matplotlib/Seaborn** (for static plots): https://matplotlib.org/

## Handoff to Next Developer
1. **Review ProtoIDS Architecture**: Understand encoder output and prototype organization
2. **Check Existing Visualizations**: See if any analysis scripts already generate relevant plots
3. **Start with Streamlit**: Recommended for fastest iteration with Python backend
4. **Implement Core Prediction**: Verify matches backend results
5. **Add Visualizations Iteratively**: Based on feedback and research needs
6. **Document Limitations**: Clearly state this is a research demo, not a production system
7. **Keep Simple**: Avoid over-engineering; focus on research utility

---
*Last updated: $(date +%Y-%m-%d)*