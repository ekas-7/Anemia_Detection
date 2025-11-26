#!/usr/bin/env python3
"""
Live Anemia Detection Launcher
Choose between OpenCV camera interface or Streamlit web interface
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_camera_access():
    """Check if camera is accessible"""
    try:
        import cv2
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            camera.release()
            return ret
        return False
    except:
        return False

def run_opencv_interface():
    """Run OpenCV camera interface"""
    logger.info("🚀 Starting OpenCV camera interface...")
    try:
        from live_camera_detector import main
        main()
    except KeyboardInterrupt:
        logger.info("👋 OpenCV interface closed by user")
    except Exception as e:
        logger.error(f"❌ Error in OpenCV interface: {e}")

def run_streamlit_interface():
    """Run Streamlit web interface"""
    logger.info("🚀 Starting Streamlit web interface...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                      cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        logger.info("👋 Streamlit interface closed by user")
    except Exception as e:
        logger.error(f"❌ Error in Streamlit interface: {e}")

def main():
    """Main launcher"""
    print("""
🔬 ANEMIA DETECTION SYSTEM LAUNCHER
===================================

Choose your interface:

1. 📷 OpenCV Camera Interface (Direct camera feed with real-time overlay)
   - Real-time camera preview with detection results
   - Keyboard controls for capture and analysis
   - Faster, more responsive
   - Best for: Desktop use, real-time monitoring

2. 🌐 Streamlit Web Interface (User-friendly web app)
   - Modern web interface with file upload
   - Results dashboard and history
   - Better visualization and controls
   - Best for: Ease of use, detailed analysis

3. ❌ Exit
""")
    
    # Check camera access
    if check_camera_access():
        print("✅ Camera detected and accessible")
    else:
        print("⚠️  Camera not detected or not accessible")
        print("   - Check camera permissions")
        print("   - Make sure no other app is using the camera")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            
            if choice == "1":
                if not check_camera_access():
                    print("❌ Camera not accessible. Please check camera permissions.")
                    continue
                run_opencv_interface()
                break
            
            elif choice == "2":
                run_streamlit_interface()
                break
            
            elif choice == "3":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            break

if __name__ == "__main__":
    main()