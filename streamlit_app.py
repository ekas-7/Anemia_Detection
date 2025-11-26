#!/usr/bin/env python3
"""
Streamlit Web Interface for Live Anemia Detection
User-friendly web interface with camera capture and real-time results
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
import json
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image
import pandas as pd

from anemia_rag_pipeline import AnemiaRAGPipeline

# Page configuration
st.set_page_config(
    page_title="Live Anemia Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitAnemiaDetector:
    """Streamlit-based anemia detector"""
    
    def __init__(self):
        self.dataset_path = "/Users/ekas/Desktop/Anemia_Detection/dataset anemia"
        self.ollama_host = "http://localhost:11434"
        
        # Initialize session state
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = None
            st.session_state.camera_active = False
            st.session_state.latest_result = None
            st.session_state.processing_status = "Ready"
            st.session_state.results_history = []
    
    def initialize_pipeline(self):
        """Initialize the RAG pipeline"""
        if st.session_state.pipeline is None:
            with st.spinner("🚀 Initializing Anemia RAG Pipeline..."):
                try:
                    st.session_state.pipeline = AnemiaRAGPipeline(
                        self.dataset_path, 
                        self.ollama_host
                    )
                    st.success("✅ Pipeline initialized successfully!")
                    return True
                except Exception as e:
                    st.error(f"❌ Failed to initialize pipeline: {e}")
                    return False
        return True
    
    def capture_camera_frame(self):
        """Capture frame from camera"""
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("❌ Could not open camera")
            return None
        
        ret, frame = camera.read()
        camera.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            st.error("❌ Failed to capture frame")
            return None
    
    def save_uploaded_image(self, uploaded_file):
        """Save uploaded image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_image_{timestamp}.jpg"
        filepath = Path("live_results") / filename
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(exist_ok=True)
        
        # Save image
        image = Image.open(uploaded_file)
        image.save(filepath)
        
        return str(filepath)
    
    def process_image(self, image_path: str, progress_bar=None):
        """Process image through RAG pipeline"""
        try:
            if progress_bar:
                progress_bar.progress(20, "🔍 Finding similar cases...")
            
            result = st.session_state.pipeline.classify_image(
                image_path=image_path,
                image_type='original',
                use_rag=True,
                n_similar=3
            )
            
            if progress_bar:
                progress_bar.progress(100, "✅ Analysis complete!")
            
            # Add to history
            st.session_state.results_history.append(result)
            st.session_state.latest_result = result
            
            return result
            
        except Exception as e:
            if progress_bar:
                progress_bar.empty()
            st.error(f"❌ Processing failed: {e}")
            return None
    
    def display_result(self, result):
        """Display classification result"""
        if not result or 'error' in result:
            st.error("❌ No valid result to display")
            return
        
        # Main result
        col1, col2 = st.columns(2)
        
        with col1:
            classification = result.get('anemia_classification', 'unknown')
            confidence = result.get('confidence_score', 0)
            
            if classification == 'anemic':
                st.markdown(f"""
                <div class="status-box status-error">
                    <h3>🔴 ANEMIC DETECTED</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            elif classification == 'non-anemic':
                st.markdown(f"""
                <div class="status-box status-success">
                    <h3>🟢 NON-ANEMIC</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-box status-warning">
                    <h3>⚠️ UNKNOWN</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Analysis Details")
            
            # Key observations
            observations = result.get('key_observations', [])
            if observations:
                for obs in observations[:3]:  # Show top 3
                    st.write(f"• {obs}")
            
            # Conjunctiva color
            color_desc = result.get('conjunctiva_color', 'Not specified')
            st.write(f"**Conjunctiva Color:** {color_desc}")
        
        # Detailed analysis
        with st.expander("🔬 Detailed Medical Analysis"):
            reasoning = result.get('reasoning', 'No detailed reasoning available')
            if isinstance(reasoning, list):
                reasoning = ' '.join(reasoning)
            st.write(reasoning)
            
            # Similar cases influence
            similar_influence = result.get('similar_case_influence', 'No information available')
            if isinstance(similar_influence, list):
                similar_influence = ' '.join(similar_influence)
            st.write(f"**Similar Cases Analysis:** {similar_influence}")
        
        # Technical details
        with st.expander("⚙️ Technical Details"):
            st.write(f"**Processing Time:** {result.get('processing_time', 'N/A'):.2f}s")
            st.write(f"**Similar Cases Found:** {result.get('similar_cases_count', 'N/A')}")
            st.write(f"**Image Type:** {result.get('image_type', 'original')}")
            st.write(f"**Timestamp:** {result.get('timestamp', 'N/A')}")
    
    def sidebar_controls(self):
        """Render sidebar controls"""
        st.sidebar.title("🔬 Control Panel")
        
        # Pipeline status
        if st.session_state.pipeline:
            st.sidebar.success("✅ Pipeline Ready")
        else:
            st.sidebar.warning("⚠️ Pipeline Not Initialized")
        
        # System stats
        if st.session_state.pipeline:
            stats = st.session_state.pipeline.get_pipeline_stats()
            st.sidebar.markdown("### 📊 System Stats")
            st.sidebar.write(f"**Indexed Images:** {stats.get('total_indexed_images', 'N/A')}")
            st.sidebar.write(f"**Model:** {stats.get('model_name', 'N/A')}")
        
        # Results history
        if st.session_state.results_history:
            st.sidebar.markdown("### 📈 Results History")
            st.sidebar.write(f"Total Analyses: {len(st.session_state.results_history)}")
            
            # Quick stats
            anemic_count = sum(1 for r in st.session_state.results_history 
                             if r.get('anemia_classification') == 'anemic')
            non_anemic_count = len(st.session_state.results_history) - anemic_count
            
            st.sidebar.write(f"🔴 Anemic: {anemic_count}")
            st.sidebar.write(f"🟢 Non-anemic: {non_anemic_count}")
        
        # Download results
        if st.session_state.results_history:
            if st.sidebar.button("📥 Download Results History"):
                self.download_results()
    
    def download_results(self):
        """Generate downloadable results"""
        results_df = pd.DataFrame(st.session_state.results_history)
        csv = results_df.to_csv(index=False)
        
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"anemia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def main_interface(self):
        """Main Streamlit interface"""
        # Header
        st.markdown('<h1 class="main-header">🔬 Live Anemia Detection System</h1>', 
                   unsafe_allow_html=True)
        
        # Initialize pipeline
        if not self.initialize_pipeline():
            st.stop()
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📷 Live Camera", "📁 Upload Image", "📊 Results Dashboard"])
        
        with tab1:
            st.markdown("### 📷 Camera Capture")
            st.write("Click the button below to capture an image from your camera and analyze it for anemia.")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if st.button("📸 Capture & Analyze", type="primary", use_container_width=True):
                    # Capture frame
                    with st.spinner("📷 Capturing image..."):
                        frame = self.capture_camera_frame()
                    
                    if frame is not None:
                        # Save captured frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_path = f"live_results/camera_capture_{timestamp}.jpg"
                        Path("live_results").mkdir(exist_ok=True)
                        
                        # Convert to PIL and save
                        pil_image = Image.fromarray(frame)
                        pil_image.save(temp_path)
                        
                        # Process image
                        progress = st.progress(0, "🔄 Starting analysis...")
                        result = self.process_image(temp_path, progress)
                        progress.empty()
                        
                        if result:
                            # Display captured image
                            with col2:
                                st.image(frame, caption="Captured Image", use_container_width=True)
                            
                            # Display results
                            st.markdown("---")
                            self.display_result(result)
            
            with col2:
                st.markdown("**Instructions:**")
                st.write("• Ensure good lighting on the eye area")
                st.write("• Position the lower eyelid clearly in view") 
                st.write("• Hold steady when capturing")
                st.write("• Analysis takes 20-30 seconds")
        
        with tab2:
            st.markdown("### 📁 Upload Image")
            
            uploaded_file = st.file_uploader(
                "Choose an eye image...", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of an eye showing the conjunctiva (inner eyelid)"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔬 Analyze Image", type="primary"):
                    # Save uploaded image
                    image_path = self.save_uploaded_image(uploaded_file)
                    
                    # Process image
                    progress = st.progress(0, "🔄 Starting analysis...")
                    result = self.process_image(image_path, progress)
                    progress.empty()
                    
                    if result:
                        st.markdown("---")
                        self.display_result(result)
        
        with tab3:
            st.markdown("### 📊 Results Dashboard")
            
            if not st.session_state.results_history:
                st.info("No results yet. Capture or upload some images to see the dashboard.")
            else:
                # Summary metrics
                total_results = len(st.session_state.results_history)
                anemic_results = sum(1 for r in st.session_state.results_history 
                                   if r.get('anemia_classification') == 'anemic')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyses", total_results)
                with col2:
                    st.metric("Anemic Cases", anemic_results)
                with col3:
                    st.metric("Non-anemic Cases", total_results - anemic_results)
                
                # Results table
                st.markdown("### 📋 Recent Results")
                
                # Convert results to DataFrame
                df_data = []
                for i, result in enumerate(st.session_state.results_history[-10:]):  # Last 10 results
                    df_data.append({
                        'Index': len(st.session_state.results_history) - len(st.session_state.results_history[-10:]) + i + 1,
                        'Timestamp': result.get('timestamp', 'N/A'),
                        'Classification': result.get('anemia_classification', 'unknown'),
                        'Confidence': f"{result.get('confidence_score', 0):.2f}",
                        'Processing Time': f"{result.get('processing_time', 0):.1f}s"
                    })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
        
        # Sidebar
        self.sidebar_controls()

def main():
    """Main Streamlit app"""
    detector = StreamlitAnemiaDetector()
    detector.main_interface()

if __name__ == "__main__":
    main()