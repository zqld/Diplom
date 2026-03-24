"""
Download TensorFlow Hub models for offline use.
Run this script once to cache models locally.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def download_models():
    """Download TensorFlow Hub models for offline use."""
    print("=" * 60)
    print("TensorFlow Hub Models Downloader")
    print("=" * 60)
    print()
    print("Models will be cached in: ~/.cache/tf-hub/")
    print()
    
    models = [
        {
            'name': 'MoveNet SinglePose Lightning',
            'url': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
            'description': 'Fast pose estimation model (recommended)',
            'size': '~10 MB'
        },
        {
            'name': 'MoveNet SinglePose Thunder',
            'url': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
            'description': 'More accurate pose estimation model',
            'size': '~20 MB'
        },
    ]
    
    print("Models to download:")
    print()
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")
        print(f"   URL: {model['url']}")
        print(f"   Description: {model['description']}")
        print(f"   Size: {model['size']}")
        print()
    
    print("-" * 60)
    response = input("Download all models? (y/n): ")
    
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print()
    print("Downloading models...")
    print()
    
    for model in models:
        print(f"Downloading {model['name']}...")
        try:
            import tensorflow_hub as hub
            model_handle = hub.load(model['url'])
            print(f"  ✓ {model['name']} downloaded successfully!")
        except Exception as e:
            print(f"  ✗ Failed to download {model['name']}: {e}")
        print()
    
    print("=" * 60)
    print("Download complete!")
    print()
    print("Models are now cached locally.")
    print("They will work offline after this first download.")
    print()
    print("To verify offline functionality:")
    print("  1. Disconnect from the internet")
    print("  2. Run the application")
    print()


def check_downloaded_models():
    """Check which models are already downloaded."""
    print("Checking downloaded models...")
    print()
    
    cache_dir = os.path.expanduser("~/.cache/tf-hub")
    if os.path.exists(cache_dir):
        print(f"Cache directory: {cache_dir}")
        print()
        contents = os.listdir(cache_dir)
        if contents:
            print("Downloaded models:")
            for item in contents:
                print(f"  - {item}")
        else:
            print("No models downloaded yet.")
    else:
        print("No cache directory found.")
    print()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download TensorFlow Hub models')
    parser.add_argument('--check', action='store_true', help='Check downloaded models')
    args = parser.parse_args()
    
    if args.check:
        check_downloaded_models()
    else:
        download_models()