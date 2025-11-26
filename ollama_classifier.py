#!/usr/bin/env python3
"""
Ollama Integration for Anemia Classification
Handles communication with Ollama vision models for anemia detection
"""

import ollama
import json
import base64
from PIL import Image
import io
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class OllamaAnemiaClassifier:
    """
    Ollama-based anemia classifier using vision models
    """
    
    def __init__(self, model_name: str = "llava", host: str = "http://localhost:11434"):
        """
        Initialize Ollama classifier
        
        Args:
            model_name: Ollama model to use (llava, bakllava, etc.)
            host: Ollama server host (for Raspberry Pi use Pi's IP)
        """
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        
        # Test connection and model availability
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server and model availability"""
        try:
            # List available models
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                logger.info(f"Please install the model with: ollama pull {self.model_name}")
            else:
                logger.info(f"✅ Connected to Ollama with model {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.host}: {e}")
            logger.info("Make sure Ollama is running and accessible")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Ollama"""
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return None
    
    def classify_anemia_basic(self, image_path: str) -> Dict[str, Any]:
        """
        Basic anemia classification without RAG context
        
        Args:
            image_path: Path to the eye/conjunctiva image
            
        Returns:
            Dict with classification results
        """
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return {"error": "Failed to encode image"}
        
        prompt = """You are a medical AI assistant specializing in anemia detection from conjunctiva images.

Analyze this eye/conjunctiva image and determine if the patient shows signs of anemia based on:
1. Pallor (paleness) of the conjunctiva
2. Color intensity of blood vessels
3. Overall redness/pinkness of the inner eyelid

Provide your analysis in this exact JSON format:
{
    "anemia_classification": "anemic" or "non-anemic",
    "confidence_score": 0.0 to 1.0,
    "key_observations": ["observation1", "observation2", "observation3"],
    "conjunctiva_color": "description of color",
    "reasoning": "detailed explanation of your decision"
}

Be precise and clinical in your assessment."""

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False
            )
            
            # Try to extract JSON from response
            response_text = response['response']
            try:
                # Look for JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    result['raw_response'] = response_text
                    return result
                else:
                    # Fallback if no JSON found
                    return {
                        "error": "No valid JSON in response",
                        "raw_response": response_text
                    }
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text
                }
                
        except Exception as e:
            logger.error(f"Error in Ollama classification: {e}")
            return {"error": str(e)}
    
    def classify_anemia_with_rag(self, image_path: str, similar_cases: List[Dict]) -> Dict[str, Any]:
        """
        Anemia classification enhanced with RAG context
        
        Args:
            image_path: Path to the query image
            similar_cases: List of similar cases from RAG system
            
        Returns:
            Dict with enhanced classification results
        """
        image_b64 = self._encode_image(image_path)
        if not image_b64:
            return {"error": "Failed to encode image"}
        
        # Prepare context from similar cases
        context_info = self._prepare_rag_context(similar_cases)
        
        prompt = f"""You are a medical AI assistant specializing in anemia detection from conjunctiva images.

You have access to a database of similar cases. Here are the most relevant cases for comparison:

SIMILAR CASES CONTEXT:
{context_info}

Now analyze this NEW eye/conjunctiva image and determine if the patient shows signs of anemia.

Consider:
1. Pallor (paleness) of the conjunctiva compared to the similar cases
2. Color intensity of blood vessels
3. Overall redness/pinkness of the inner eyelid
4. How this image compares to the anemic vs non-anemic cases shown above

Provide your analysis in this exact JSON format:
{{
    "anemia_classification": "anemic" or "non-anemic",
    "confidence_score": 0.0 to 1.0,
    "key_observations": ["observation1", "observation2", "observation3"],
    "conjunctiva_color": "description of color",
    "comparison_notes": "how this compares to similar cases",
    "reasoning": "detailed explanation referencing similar cases",
    "similar_case_influence": "how the similar cases influenced your decision"
}}

Be precise and clinical in your assessment."""

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_b64],
                stream=False
            )
            
            # Parse response similar to basic classification
            response_text = response['response']
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    result['raw_response'] = response_text
                    result['rag_context'] = similar_cases
                    return result
                else:
                    return {
                        "error": "No valid JSON in response",
                        "raw_response": response_text,
                        "rag_context": similar_cases
                    }
            except json.JSONDecodeError:
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text,
                    "rag_context": similar_cases
                }
                
        except Exception as e:
            logger.error(f"Error in Ollama RAG classification: {e}")
            return {"error": str(e)}
    
    def _prepare_rag_context(self, similar_cases: List[Dict]) -> str:
        """Prepare context string from similar cases"""
        if not similar_cases:
            return "No similar cases found."
        
        context_parts = []
        for i, case in enumerate(similar_cases[:5], 1):  # Top 5 cases
            metadata = case['metadata']
            similarity = case['similarity_score']
            
            anemia_status = "ANEMIC" if metadata.get('anemic', False) else "NON-ANEMIC"
            hgb = metadata.get('hgb', 'unknown')
            gender = metadata.get('gender', 'unknown')
            age = metadata.get('age', 'unknown')
            country = metadata.get('country', 'unknown')
            
            case_info = f"""
Case {i} (Similarity: {similarity:.2f}):
- Status: {anemia_status}
- Hemoglobin: {hgb} g/dL
- Patient: {gender}, {age} years old, {country}
- Image type: {metadata.get('image_type', 'unknown')}
"""
            context_parts.append(case_info)
        
        return "\n".join(context_parts)
    
    def batch_classify(self, image_paths: List[str], use_rag: bool = True, 
                      rag_system=None) -> List[Dict[str, Any]]:
        """
        Classify multiple images in batch
        
        Args:
            image_paths: List of image paths to classify
            use_rag: Whether to use RAG context
            rag_system: RAG system instance for similarity search
            
        Returns:
            List of classification results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            if use_rag and rag_system:
                # Get similar cases from RAG
                similar_cases = rag_system.search_similar_images(
                    query_image_path=image_path,
                    image_type='original',  # or determine from filename
                    n_results=5
                )
                result = self.classify_anemia_with_rag(image_path, similar_cases)
            else:
                result = self.classify_anemia_basic(image_path)
            
            result['image_path'] = image_path
            result['image_name'] = Path(image_path).name
            results.append(result)
        
        return results
    
    def evaluate_accuracy(self, test_results: List[Dict], ground_truth_metadata: Dict) -> Dict[str, float]:
        """
        Evaluate classification accuracy against ground truth
        
        Args:
            test_results: List of classification results
            ground_truth_metadata: Dictionary with true anemia status
            
        Returns:
            Dictionary with accuracy metrics
        """
        correct_predictions = 0
        total_predictions = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for result in test_results:
            if 'error' in result:
                continue
                
            # Extract sample key from image path
            image_path = result['image_path']
            # This would need to be adapted based on your naming convention
            # For now, a placeholder implementation
            
            predicted_anemic = result.get('anemia_classification') == 'anemic'
            
            # You'd need to implement ground truth lookup here
            # actual_anemic = ground_truth_metadata.get(sample_key, {}).get('anemic')
            
            # Update metrics (placeholder)
            total_predictions += 1
        
        if total_predictions == 0:
            return {"error": "No valid predictions to evaluate"}
        
        accuracy = correct_predictions / total_predictions
        
        return {
            "accuracy": accuracy,
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            # Add more metrics as needed
        }

def main():
    """Test the Ollama classifier"""
    # Initialize classifier
    classifier = OllamaAnemiaClassifier()
    
    # Test with a sample image (you'll need to provide a valid path)
    test_image = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia/India/1/20200118_164733.jpg"
    
    if Path(test_image).exists():
        logger.info("Testing basic classification...")
        result = classifier.classify_anemia_basic(test_image)
        print(json.dumps(result, indent=2))
    else:
        logger.warning(f"Test image not found: {test_image}")
        logger.info("Please provide a valid image path for testing")

if __name__ == "__main__":
    main()