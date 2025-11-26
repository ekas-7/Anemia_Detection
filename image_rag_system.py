#!/usr/bin/env python3
"""
Anemia Image RAG System
Creates embeddings for all anemia images and builds a vector database for similarity search
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnemiaImageRAG:
    """
    Image RAG system for anemia detection using CLIP embeddings and ChromaDB
    """
    
    def __init__(self, dataset_path: str, db_path: str = "./anemia_vectordb"):
        self.dataset_path = Path(dataset_path)
        self.db_path = db_path
        
        # Initialize CLIP model for image embeddings
        logger.info("Loading CLIP model...")
        self.clip_model = SentenceTransformer('clip-ViT-B-32')
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create collections for different image types
        self.collections = {
            'original': self._get_or_create_collection('anemia_original_images'),
            'palpebral': self._get_or_create_collection('anemia_palpebral_images'),
            'forniceal': self._get_or_create_collection('anemia_forniceal_images'),
            'combined': self._get_or_create_collection('anemia_combined_images')
        }
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(name)
    
    def _load_metadata(self) -> Dict:
        """Load hemoglobin and demographic metadata"""
        metadata = {}
        
        # Load India metadata
        india_excel = self.dataset_path / "India" / "India.xlsx"
        if india_excel.exists():
            india_df = pd.read_excel(india_excel)
            for _, row in india_df.iterrows():
                key = f"india_{int(row['Number'])}"
                metadata[key] = {
                    'country': 'India',
                    'hgb': float(row['Hgb']),
                    'gender': row['Gender'],
                    'age': float(row['Age']) if pd.notna(row['Age']) else None,
                    'anemic': self._classify_anemia(row['Hgb'], row['Gender'])
                }
        
        # Load Italy metadata
        italy_excel = self.dataset_path / "Italy" / "Italy.xlsx"
        if italy_excel.exists():
            italy_df = pd.read_excel(italy_excel)
            for _, row in italy_df.iterrows():
                key = f"italy_{int(row['Number'])}"
                hgb_clean = str(row['Hgb']).replace(',', '.')
                try:
                    hgb_val = float(hgb_clean)
                    metadata[key] = {
                        'country': 'Italy',
                        'hgb': hgb_val,
                        'gender': row['Gender'],
                        'age': float(row['Age']) if pd.notna(row['Age']) else None,
                        'anemic': self._classify_anemia(hgb_val, row['Gender'])
                    }
                except:
                    continue
        
        logger.info(f"Loaded metadata for {len(metadata)} samples")
        return metadata
    
    def _classify_anemia(self, hgb: float, gender: str) -> bool:
        """Classify anemia based on WHO standards"""
        if pd.isna(hgb) or pd.isna(gender):
            return None
        if gender == 'M':
            return hgb < 13.0
        elif gender == 'F':
            return hgb < 12.0
        return None
    
    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP embedding for an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize to standard size for consistency
            image = image.resize((224, 224))
            embedding = self.clip_model.encode([image])
            return embedding[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def _extract_sample_info(self, image_path: str) -> Tuple[str, str, str]:
        """Extract sample info from image path"""
        path_parts = Path(image_path).parts
        
        # Find country and sample number
        if 'India' in path_parts:
            country = 'india'
            sample_num = Path(image_path).parent.name
        elif 'Italy' in path_parts:
            country = 'italy'
            sample_num = Path(image_path).parent.name
        else:
            return None, None, None
            
        # Determine image type
        filename = Path(image_path).name.lower()
        if 'palpebral.png' in filename and 'forniceal' not in filename:
            image_type = 'palpebral'
        elif 'forniceal.png' in filename and 'palpebral' not in filename:
            image_type = 'forniceal'
        elif 'forniceal_palpebral.png' in filename:
            image_type = 'combined'
        elif filename.endswith('.jpg'):
            image_type = 'original'
        else:
            image_type = 'unknown'
            
        return country, sample_num, image_type
    
    def index_all_images(self):
        """Index all images in the dataset"""
        logger.info("Starting image indexing process...")
        
        indexed_count = 0
        for country in ['India', 'Italy']:
            country_path = self.dataset_path / country
            if not country_path.exists():
                continue
                
            # Process each sample directory
            for sample_dir in country_path.iterdir():
                if not sample_dir.is_dir() or sample_dir.name.endswith('.xlsx'):
                    continue
                    
                sample_key = f"{country.lower()}_{sample_dir.name}"
                if sample_key not in self.metadata:
                    continue
                    
                # Process all images in the sample directory
                for image_path in sample_dir.glob('*'):
                    if image_path.suffix.lower() not in ['.jpg', '.png']:
                        continue
                        
                    country_code, sample_num, image_type = self._extract_sample_info(str(image_path))
                    if not all([country_code, sample_num, image_type]) or image_type == 'unknown':
                        continue
                    
                    # Check if this collection type exists
                    if image_type not in self.collections:
                        continue
                        
                    # Generate embedding
                    embedding = self._get_image_embedding(str(image_path))
                    if embedding is None:
                        continue
                    
                    # Create document ID
                    doc_id = f"{country_code}_{sample_num}_{image_type}"
                    
                    # Prepare metadata for storage
                    sample_metadata = self.metadata[sample_key].copy()
                    sample_metadata.update({
                        'image_path': str(image_path),
                        'image_type': image_type,
                        'sample_id': sample_key,
                        'doc_id': doc_id
                    })
                    
                    # Add to ChromaDB
                    try:
                        self.collections[image_type].add(
                            embeddings=[embedding.tolist()],
                            documents=[f"Anemia image - {country} sample {sample_num} - {image_type} view"],
                            metadatas=[sample_metadata],
                            ids=[doc_id]
                        )
                        indexed_count += 1
                        
                        if indexed_count % 50 == 0:
                            logger.info(f"Indexed {indexed_count} images...")
                            
                    except Exception as e:
                        logger.error(f"Error adding to ChromaDB: {e}")
                        continue
        
        logger.info(f"✅ Image indexing complete! Indexed {indexed_count} images total")
        
        # Print collection stats
        for name, collection in self.collections.items():
            count = collection.count()
            logger.info(f"   {name}: {count} images")
    
    def search_similar_images(self, query_image_path: str, image_type: str = 'original', 
                            n_results: int = 5, anemic_only: bool = None) -> List[Dict]:
        """
        Search for similar images in the database
        
        Args:
            query_image_path: Path to the query image
            image_type: Type of image to search ('original', 'palpebral', 'forniceal', 'combined')
            n_results: Number of results to return
            anemic_only: Filter by anemia status (True/False/None for all)
        
        Returns:
            List of similar images with metadata
        """
        if image_type not in self.collections:
            raise ValueError(f"Image type '{image_type}' not supported")
        
        # Generate embedding for query image
        query_embedding = self._get_image_embedding(query_image_path)
        if query_embedding is None:
            return []
        
        # Search in ChromaDB
        results = self.collections[image_type].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 3  # Get more results for filtering
        )
        
        # Process and filter results
        similar_images = []
        for i, (distance, metadata, document) in enumerate(zip(
            results['distances'][0], 
            results['metadatas'][0], 
            results['documents'][0]
        )):
            # Filter by anemia status if specified
            if anemic_only is not None and metadata.get('anemic') != anemic_only:
                continue
                
            similar_images.append({
                'similarity_score': 1 - distance,  # Convert distance to similarity
                'metadata': metadata,
                'description': document,
                'rank': len(similar_images) + 1
            })
            
            if len(similar_images) >= n_results:
                break
        
        return similar_images
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed collections"""
        stats = {}
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {'total_images': count}
            
            if count > 0:
                # Sample some metadata to get anemia distribution
                sample_results = collection.get(limit=min(count, 1000))
                if sample_results['metadatas']:
                    anemic_count = sum(1 for meta in sample_results['metadatas'] if meta.get('anemic', False))
                    stats[name]['anemic_images'] = anemic_count
                    stats[name]['non_anemic_images'] = len(sample_results['metadatas']) - anemic_count
        
        return stats

def main():
    """Main function to index the anemia dataset"""
    # Initialize the RAG system
    dataset_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia"
    rag_system = AnemiaImageRAG(dataset_path)
    
    # Check if we need to index (if collections are empty)
    needs_indexing = any(collection.count() == 0 for collection in rag_system.collections.values())
    
    if needs_indexing:
        logger.info("🚀 Starting initial indexing of anemia dataset...")
        rag_system.index_all_images()
    else:
        logger.info("📚 Database already indexed!")
    
    # Show statistics
    stats = rag_system.get_collection_stats()
    logger.info("📊 Collection Statistics:")
    for collection_name, stat in stats.items():
        logger.info(f"  {collection_name}: {stat}")
    
    return rag_system

if __name__ == "__main__":
    rag_system = main()