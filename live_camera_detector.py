#!/usr/bin/env python3
"""
Live Camera Anemia Detection System
Real-time camera capture with RAG + Ollama anemia classification
"""

import cv2
import numpy as np
import time
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from anemia_rag_pipeline import AnemiaRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveAnemiaDetector:
    """
    Real-time anemia detection using camera feed and RAG pipeline
    """
    
    def __init__(self, dataset_path: str, ollama_host: str = "http://localhost:11434"):
        """
        Initialize live detector
        
        Args:
            dataset_path: Path to anemia dataset
            ollama_host: Ollama server URL
        """
        self.dataset_path = dataset_path
        self.ollama_host = ollama_host
        
        # Initialize camera
        self.camera = None
        self.camera_index = 0
        
        # Processing queue and threads
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # Results storage
        self.latest_result = None
        self.processing_status = "Initializing..."
        
        # Initialize RAG pipeline
        logger.info("🚀 Initializing Anemia RAG Pipeline...")
        self.pipeline = AnemiaRAGPipeline(dataset_path, ollama_host)
        
        # Create results directory
        self.results_dir = Path("live_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("✅ Live Anemia Detector initialized!")
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize camera capture
        
        Args:
            camera_index: Camera device index
            
        Returns:
            True if camera initialized successfully
        """
        logger.info(f"📷 Initializing camera {camera_index}...")
        
        self.camera = cv2.VideoCapture(camera_index)
        
        if not self.camera.isOpened():
            logger.error(f"❌ Failed to open camera {camera_index}")
            return False
        
        # Set camera properties for better quality
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Test camera
        ret, frame = self.camera.read()
        if not ret:
            logger.error("❌ Failed to read from camera")
            return False
        
        logger.info(f"✅ Camera initialized: {frame.shape}")
        return True
    
    def capture_and_save_frame(self, frame: np.ndarray) -> str:
        """
        Save frame to disk for processing
        
        Args:
            frame: Camera frame
            
        Returns:
            Path to saved image
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"live_capture_{timestamp}.jpg"
        filepath = self.results_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
    
    def process_frame_worker(self):
        """
        Background worker thread for processing frames
        """
        logger.info("🔄 Starting frame processing worker...")
        
        while self.is_running:
            try:
                # Get frame from queue (timeout to allow clean shutdown)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                if frame_data is None:  # Shutdown signal
                    break
                
                frame, capture_time = frame_data
                
                self.processing_status = "Processing image..."
                
                # Save frame and process
                image_path = self.capture_and_save_frame(frame)
                
                logger.info(f"🔍 Processing captured frame: {Path(image_path).name}")
                
                # Run classification through RAG pipeline
                start_time = time.time()
                result = self.pipeline.classify_image(
                    image_path=image_path,
                    image_type='original',
                    use_rag=True,
                    n_similar=3  # Fewer similar cases for faster processing
                )
                
                processing_time = time.time() - start_time
                
                # Add timing and capture info
                result.update({
                    'capture_time': capture_time,
                    'processing_time': processing_time,
                    'saved_image_path': image_path
                })
                
                # Send result to main thread
                self.result_queue.put(result)
                
                self.processing_status = f"✅ Completed in {processing_time:.1f}s"
                
                logger.info(f"✅ Classification complete: {result.get('anemia_classification', 'unknown')} "
                          f"(confidence: {result.get('confidence_score', 0):.2f})")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Error processing frame: {e}")
                self.processing_status = f"❌ Error: {str(e)[:50]}..."
                
        logger.info("🛑 Frame processing worker stopped")
    
    def start_processing(self):
        """Start background processing thread"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_frame_worker, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop background processing"""
        self.is_running = False
        
        # Send shutdown signal
        if not self.frame_queue.full():
            self.frame_queue.put(None)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
    
    def add_overlay_text(self, frame: np.ndarray, result: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Add overlay text with classification results
        
        Args:
            frame: Camera frame
            result: Classification result
            
        Returns:
            Frame with overlay text
        """
        overlay_frame = frame.copy()
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Background for text
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay_frame, 0.7, overlay, 0.3, 0, overlay_frame)
        
        # Title
        cv2.putText(overlay_frame, "Live Anemia Detection System", 
                   (20, 40), font, font_scale, (255, 255, 255), thickness)
        
        # Processing status
        cv2.putText(overlay_frame, f"Status: {self.processing_status}", 
                   (20, 70), font, font_scale - 0.2, (255, 255, 0), thickness - 1)
        
        if result:
            # Classification result
            classification = result.get('anemia_classification', 'unknown')
            confidence = result.get('confidence_score', 0)
            
            # Color based on classification
            color = (0, 255, 0) if classification == 'non-anemic' else (0, 0, 255)
            
            cv2.putText(overlay_frame, f"Classification: {classification.upper()}", 
                       (20, 110), font, font_scale, color, thickness)
            
            cv2.putText(overlay_frame, f"Confidence: {confidence:.2f}", 
                       (20, 140), font, font_scale - 0.2, (255, 255, 255), thickness - 1)
            
            # Processing time
            proc_time = result.get('processing_time', 0)
            cv2.putText(overlay_frame, f"Processing Time: {proc_time:.1f}s", 
                       (20, 170), font, font_scale - 0.2, (200, 200, 200), thickness - 1)
        
        # Instructions
        cv2.putText(overlay_frame, "Press 'c' to capture | 'q' to quit | 's' to save results", 
                   (20, overlay_frame.shape[0] - 20), font, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def run_live_detection(self):
        """
        Main loop for live detection
        """
        if not self.initialize_camera():
            logger.error("❌ Failed to initialize camera")
            return
        
        logger.info("🎥 Starting live detection...")
        logger.info("📋 Controls:")
        logger.info("   - Press 'c' to capture and analyze current frame")
        logger.info("   - Press 'q' to quit")
        logger.info("   - Press 's' to save latest results")
        
        # Start background processing
        self.start_processing()
        
        last_capture_time = 0
        auto_capture_interval = 30  # Auto-capture every 30 seconds
        
        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("❌ Failed to read camera frame")
                    break
                
                # Check for new results
                try:
                    while not self.result_queue.empty():
                        self.latest_result = self.result_queue.get_nowait()
                except queue.Empty:
                    pass
                
                # Add overlay
                display_frame = self.add_overlay_text(frame, self.latest_result)
                
                # Show frame
                cv2.imshow("Live Anemia Detection", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("🛑 Quit requested")
                    break
                elif key == ord('c'):
                    # Manual capture
                    if not self.frame_queue.full():
                        capture_time = datetime.now().isoformat()
                        self.frame_queue.put((frame.copy(), capture_time))
                        self.processing_status = "Queued for processing..."
                        logger.info("📸 Frame captured for analysis")
                    else:
                        logger.warning("⚠️ Processing queue full, skipping capture")
                elif key == ord('s'):
                    # Save latest results
                    if self.latest_result:
                        self.save_result(self.latest_result)
                        logger.info("💾 Latest result saved")
                
                # Auto-capture periodically
                current_time = time.time()
                if (current_time - last_capture_time) > auto_capture_interval:
                    if not self.frame_queue.full():
                        capture_time = datetime.now().isoformat()
                        self.frame_queue.put((frame.copy(), capture_time))
                        last_capture_time = current_time
                        logger.info("⏰ Auto-capture triggered")
        
        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def save_result(self, result: Dict[str, Any]):
        """Save classification result to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"live_result_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"💾 Result saved to {result_file}")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up...")
        
        # Stop processing
        self.stop_processing()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        logger.info("✅ Cleanup complete")

def main():
    """Main function"""
    dataset_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia"
    
    # For Raspberry Pi, change this to your Pi's IP
    ollama_host = "http://localhost:11434"
    
    try:
        # Initialize live detector
        detector = LiveAnemiaDetector(dataset_path, ollama_host)
        
        # Run live detection
        detector.run_live_detection()
        
    except Exception as e:
        logger.error(f"❌ Error in live detection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()