
import numpy as np
import joblib
import config
import pandas as pd
import json

def extract_importances():
    print("LOADING FEATURE NAMES...")
    with open(config.SOURCE_DATA_DIR / 'feature_names.txt', 'r') as f:
        feature_names = np.array([line.strip() for line in f.readlines()])
    
    print(f"Loaded {len(feature_names)} features.")

    print("\nEXTRACTING IMPORTANCES...")
    for class_name in config.CLASSES:
        detector_dir = config.MODELS_DIR / class_name
        
        # Get best model name
        with open(detector_dir / "best_model_info.json") as f:
            best_info = json.load(f)
        model_name = best_info['best_model']
        
        print(f"\n--- {class_name} ({model_name}) ---")
        
        try:
            model = joblib.load(detector_dir / f"{model_name}.joblib")
            
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            
            if importances is not None:
                # Get top 20 indices
                top_indices = np.argsort(importances)[::-1][:20]
                
                print(f"{'Feature':<60} {'Importance':>10}")
                print("-" * 75)
                for idx in top_indices:
                    fname = feature_names[idx]
                    score = importances[idx]
                    print(f"{fname:<60} {score:>10.4f}")
            else:
                print("Feature importance not available for this model type (e.g., MLP).")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    extract_importances()
