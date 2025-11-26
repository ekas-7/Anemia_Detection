#!/usr/bin/env python3
"""
Complete Anemia RAG Classification Pipeline
Combines Image RAG system with Ollama for enhanced anemia detection
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from image_rag_system import AnemiaImageRAG
from ollama_classifier import OllamaAnemiaClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnemiaRAGPipeline:
    """
    Complete pipeline combining Image RAG with Ollama for anemia classification
    """
    
    def __init__(self, dataset_path: str, ollama_host: str = "http://localhost:11434", 
                 model_name: str = "llava"):
        """
        Initialize the complete RAG pipeline
        
        Args:
            dataset_path: Path to anemia dataset
            ollama_host: Ollama server URL (use Pi IP for remote)
            model_name: Ollama model name
        """
        self.dataset_path = dataset_path
        
        logger.info("🚀 Initializing Anemia RAG Pipeline...")
        
        # Initialize RAG system
        logger.info("📚 Setting up Image RAG system...")
        self.rag_system = AnemiaImageRAG(dataset_path)
        
        # Initialize Ollama classifier
        logger.info("🤖 Setting up Ollama classifier...")
        self.classifier = OllamaAnemiaClassifier(model_name=model_name, host=ollama_host)
        
        logger.info("✅ Pipeline initialization complete!")
    
    def classify_image(self, image_path: str, image_type: str = 'original', 
                      use_rag: bool = True, n_similar: int = 5) -> Dict[str, Any]:
        """
        Classify a single image for anemia
        
        Args:
            image_path: Path to the image to classify
            image_type: Type of image ('original', 'palpebral', 'forniceal', 'combined')
            use_rag: Whether to use RAG context
            n_similar: Number of similar cases to retrieve
            
        Returns:
            Classification results with confidence and context
        """
        logger.info(f"🔍 Classifying image: {Path(image_path).name}")
        
        result = {
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'timestamp': datetime.now().isoformat(),
            'use_rag': use_rag,
            'image_type': image_type
        }
        
        if use_rag:
            # Get similar cases from RAG system
            logger.info(f"🔎 Finding {n_similar} similar cases...")
            similar_cases = self.rag_system.search_similar_images(
                query_image_path=image_path,
                image_type=image_type,
                n_results=n_similar
            )
            
            if similar_cases:
                logger.info(f"📊 Found {len(similar_cases)} similar cases")
                result['similar_cases_count'] = len(similar_cases)
                
                # Classify with RAG context
                classification = self.classifier.classify_anemia_with_rag(image_path, similar_cases)
            else:
                logger.warning("No similar cases found, falling back to basic classification")
                classification = self.classifier.classify_anemia_basic(image_path)
        else:
            # Basic classification without RAG
            classification = self.classifier.classify_anemia_basic(image_path)
        
        result.update(classification)
        return result
    
    def batch_classify(self, image_paths: List[str], image_type: str = 'original',
                      use_rag: bool = True, save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Classify multiple images in batch
        
        Args:
            image_paths: List of image paths
            image_type: Type of images
            use_rag: Whether to use RAG context
            save_results: Whether to save results to file
            
        Returns:
            List of classification results
        """
        logger.info(f"🔄 Starting batch classification of {len(image_paths)} images...")
        
        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"📸 Processing {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                result = self.classify_image(image_path, image_type, use_rag)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'image_name': Path(image_path).name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        if save_results:
            self._save_results(results, f"batch_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info(f"✅ Batch classification complete! Processed {len(results)} images")
        return results
    
    def evaluate_on_dataset(self, sample_size: int = 20, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the pipeline on a sample of the dataset
        
        Args:
            sample_size: Number of samples to test
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation metrics and results
        """
        logger.info(f"🧪 Starting evaluation on {sample_size} samples...")
        
        # Get sample images from the dataset
        test_images = self._get_test_samples(sample_size)
        
        if not test_images:
            return {"error": "No test images found"}
        
        # Run classification
        results = self.batch_classify([img['path'] for img in test_images], save_results=False)
        
        # Calculate accuracy metrics
        metrics = self._calculate_metrics(results, test_images)
        
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(test_images),
            'successful_classifications': len([r for r in results if 'error' not in r]),
            'metrics': metrics,
            'detailed_results': results
        }
        
        if save_results:
            self._save_results(evaluation_data, f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        logger.info(f"📊 Evaluation complete!")
        self._print_metrics(metrics)
        
        return evaluation_data
    
    def _get_test_samples(self, sample_size: int) -> List[Dict[str, Any]]:
        """Get random sample of images for testing"""
        test_images = []
        
        # Sample from both countries
        for country in ['India', 'Italy']:
            country_path = Path(self.dataset_path) / country
            if not country_path.exists():
                continue
            
            sample_dirs = [d for d in country_path.iterdir() 
                          if d.is_dir() and not d.name.endswith('.xlsx')]
            
            # Get up to half the sample size from each country
            country_sample_size = min(len(sample_dirs), sample_size // 2)
            
            for sample_dir in np.random.choice(sample_dirs, country_sample_size, replace=False):
                # Look for original JPG image
                jpg_files = list(sample_dir.glob('*.jpg'))
                if jpg_files:
                    image_path = jpg_files[0]
                    
                    # Get ground truth from metadata
                    sample_key = f"{country.lower()}_{sample_dir.name}"
                    metadata = self.rag_system.metadata.get(sample_key)
                    
                    if metadata:
                        test_images.append({
                            'path': str(image_path),
                            'sample_key': sample_key,
                            'ground_truth': metadata
                        })
        
        return test_images[:sample_size]
    
    def _calculate_metrics(self, results: List[Dict], test_images: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        if not results or not test_images:
            return {}
        
        # Create lookup for ground truth
        ground_truth = {img['path']: img['ground_truth'] for img in test_images}
        
        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for result in results:
            if 'error' in result or 'anemia_classification' not in result:
                continue
            
            image_path = result['image_path']
            if image_path not in ground_truth:
                continue
            
            predicted_anemic = result['anemia_classification'] == 'anemic'
            actual_anemic = ground_truth[image_path].get('anemic', False)
            
            total += 1
            if predicted_anemic == actual_anemic:
                correct += 1
            
            # Confusion matrix
            if predicted_anemic and actual_anemic:
                true_positives += 1
            elif predicted_anemic and not actual_anemic:
                false_positives += 1
            elif not predicted_anemic and actual_anemic:
                false_negatives += 1
            else:
                true_negatives += 1
        
        if total == 0:
            return {}
        
        accuracy = correct / total
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total,
            'correct_predictions': correct,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print evaluation metrics in a nice format"""
        if not metrics:
            logger.warning("No metrics to display")
            return
        
        logger.info("📊 EVALUATION METRICS:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall:    {metrics['recall']:.3f}")
        logger.info(f"   F1 Score:  {metrics['f1_score']:.3f}")
        logger.info(f"   Samples:   {metrics['total_samples']}")
    
    def _save_results(self, data: Any, filename: str):
        """Save results to JSON file"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filepath}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline"""
        rag_stats = self.rag_system.get_collection_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'ollama_host': self.classifier.host,
            'model_name': self.classifier.model_name,
            'rag_collections': rag_stats,
            'total_indexed_images': sum(stat.get('total_images', 0) for stat in rag_stats.values())
        }

def main():
    """Main function to test the complete pipeline"""
    # Configuration
    dataset_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia"
    
    # For Raspberry Pi, change this to your Pi's IP address
    # ollama_host = "http://192.168.1.100:11434"  # Example Pi IP
    ollama_host = "http://localhost:11434"  # Local testing
    
    try:
        # Initialize pipeline
        pipeline = AnemiaRAGPipeline(dataset_path, ollama_host)
        
        # Show pipeline statistics
        stats = pipeline.get_pipeline_stats()
        logger.info("📈 Pipeline Statistics:")
        for key, value in stats.items():
            if key != 'rag_collections':
                logger.info(f"   {key}: {value}")
        
        # Test with a single image
        test_image = Path(dataset_path) / "India" / "1" / "20200118_164733.jpg"
        if test_image.exists():
            logger.info("🧪 Testing single image classification...")
            result = pipeline.classify_image(str(test_image))
            logger.info("✅ Classification Result:")
            print(json.dumps({k: v for k, v in result.items() if k not in ['raw_response', 'rag_context']}, indent=2))
        
        # Run evaluation on a small sample
        logger.info("🔬 Running evaluation...")
        evaluation = pipeline.evaluate_on_dataset(sample_size=10)
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        logger.info("Make sure Ollama is running and the llava model is installed:")
        logger.info("  ollama pull llava")

if __name__ == "__main__":
    main()