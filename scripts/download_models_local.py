"""
Download TensorFlow Hub models to local folder for offline use.
"""

import os
import shutil
import tensorflow as tf
import tensorflow_hub as hub


def download_models_to_local():
    """Download TensorFlow Hub models to local models folder."""
    
    local_models_dir = 'models/tf_hub'
    os.makedirs(local_models_dir, exist_ok=True)
    
    models = [
        {
            'name': 'movenet_lightning',
            'url': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
            'description': 'Fast pose estimation'
        },
        {
            'name': 'movenet_thunder',
            'url': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
            'description': 'Accurate pose estimation'
        },
    ]
    
    print("Downloading TensorFlow Hub models to local folder...")
    print(f"Target: {os.path.abspath(local_models_dir)}")
    print()
    
    for model_info in models:
        print(f"Downloading {model_info['name']}...")
        
        try:
            # Load model from TF Hub
            model = hub.load(model_info['url'])
            
            # Export to local folder
            model_dir = os.path.join(local_models_dir, model_info['name'])
            
            # Save using TensorFlow
            # Note: TF Hub models need special handling for local export
            tf.saved_model.save(model, model_dir)
            
            print(f"  [OK] Saved to {model_dir}")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        print()
    
    print("Download complete!")
    print()
    print("To use offline models, update the code to load from local path:")
    print(f"  hub.load('file://{os.path.abspath(local_models_dir)}/movenet_lightning')")


if __name__ == '__main__':
    download_models_to_local()
