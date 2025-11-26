#!/usr/bin/env python3
"""
Raspberry Pi Setup and Testing Script for Anemia RAG System
"""

import subprocess
import sys
import platform
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements for running on Raspberry Pi"""
    system_info = {
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python_version': sys.version,
        'is_raspberry_pi': False
    }
    
    # Check if running on Raspberry Pi
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo or 'BCM' in cpuinfo:
                system_info['is_raspberry_pi'] = True
    except:
        pass
    
    logger.info(f"🖥️  System: {system_info['platform']}")
    logger.info(f"🔧 Machine: {system_info['machine']}")
    logger.info(f"🐍 Python: {system_info['python_version']}")
    
    if system_info['is_raspberry_pi']:
        logger.info("🥧 Detected Raspberry Pi!")
    else:
        logger.info("💻 Running on standard computer")
    
    return system_info

def install_ollama():
    """Install Ollama if not already installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Ollama already installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    logger.info("📦 Installing Ollama...")
    try:
        # Install Ollama
        install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
        result = subprocess.run(install_cmd, shell=True, check=True)
        logger.info("✅ Ollama installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Ollama: {e}")
        return False

def setup_ollama_models(models=['llava']):
    """Download required Ollama models"""
    logger.info("🤖 Setting up Ollama models...")
    
    for model in models:
        logger.info(f"📥 Downloading model: {model}")
        try:
            result = subprocess.run(['ollama', 'pull', model], check=True, capture_output=True, text=True)
            logger.info(f"✅ Model {model} downloaded successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to download model {model}: {e}")
            return False
    
    return True

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        logger.info("✅ Ollama service is running")
        logger.info("Available models:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError:
        logger.warning("⚠️  Ollama service may not be running")
        logger.info("💡 Start Ollama service with: ollama serve")
        return False

def test_image_processing():
    """Test basic image processing capabilities"""
    logger.info("🧪 Testing image processing...")
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create a small test image
        test_image = Image.new('RGB', (100, 100), color='red')
        test_path = Path('test_image.jpg')
        test_image.save(test_path)
        
        # Try to load and process
        loaded_image = Image.open(test_path)
        image_array = np.array(loaded_image)
        
        test_path.unlink()  # Remove test file
        
        logger.info("✅ Image processing test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image processing test failed: {e}")
        return False

def create_raspberry_pi_config():
    """Create optimized configuration for Raspberry Pi"""
    config = {
        "system_optimization": {
            "description": "Optimized settings for Raspberry Pi performance",
            "clip_model": "clip-ViT-B-32",  # Lighter model for Pi
            "batch_size": 1,  # Process one image at a time
            "max_similar_cases": 3,  # Reduce context size
            "image_resize": [224, 224],  # Standard size
            "memory_limit_gb": 2  # Conservative estimate for Pi 4
        },
        "ollama_config": {
            "host": "http://localhost:11434",
            "model": "llava",
            "context_length": 2048,  # Reduced for Pi
            "num_predict": 512  # Limit response length
        },
        "database_config": {
            "type": "chromadb",
            "path": "./anemia_vectordb_pi",
            "collection_limit": 1000  # Limit collection size
        }
    }
    
    config_path = Path("raspberry_pi_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"📝 Raspberry Pi configuration saved to {config_path}")
    return config

def run_performance_test():
    """Run a basic performance test"""
    logger.info("⏱️  Running performance test...")
    
    import time
    from sentence_transformers import SentenceTransformer
    
    try:
        # Test CLIP model loading time
        start_time = time.time()
        model = SentenceTransformer('clip-ViT-B-32')
        load_time = time.time() - start_time
        
        logger.info(f"📊 CLIP model load time: {load_time:.2f} seconds")
        
        # Test embedding generation
        start_time = time.time()
        # Create dummy image for testing
        from PIL import Image
        test_image = Image.new('RGB', (224, 224), color='blue')
        embedding = model.encode([test_image])
        encode_time = time.time() - start_time
        
        logger.info(f"📊 Image encoding time: {encode_time:.2f} seconds")
        
        performance_info = {
            'clip_load_time': load_time,
            'encoding_time': encode_time,
            'embedding_size': len(embedding[0])
        }
        
        return performance_info
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return None

def main():
    """Main setup function"""
    logger.info("🚀 Starting Raspberry Pi Setup for Anemia RAG System")
    
    # Check system requirements
    system_info = check_system_requirements()
    
    # Install and setup Ollama
    if not install_ollama():
        logger.error("Failed to install Ollama. Exiting.")
        return
    
    # Setup models
    if not setup_ollama_models():
        logger.error("Failed to setup Ollama models. Exiting.")
        return
    
    # Check service
    check_ollama_service()
    
    # Test image processing
    if not test_image_processing():
        logger.error("Image processing test failed. Check PIL/Pillow installation.")
        return
    
    # Create Pi-optimized config
    config = create_raspberry_pi_config()
    
    # Run performance test
    perf_info = run_performance_test()
    if perf_info:
        logger.info("📈 Performance test results:")
        for key, value in perf_info.items():
            logger.info(f"   {key}: {value}")
    
    logger.info("✅ Raspberry Pi setup complete!")
    logger.info("💡 Next steps:")
    logger.info("   1. Start Ollama service: ollama serve")
    logger.info("   2. Run the anemia pipeline: python anemia_rag_pipeline.py")
    logger.info("   3. For remote access, update host IP in the pipeline script")

if __name__ == "__main__":
    main()