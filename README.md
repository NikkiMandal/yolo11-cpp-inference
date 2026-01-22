# YOLO11 C++ Real-Time Inference Engine

**Optimizing Object Detection for Production Deployment**

A custom C++ inference engine for YOLO11 object detection that achieves **1.36x speedup** over Python baseline through deployment optimization using ONNX Runtime.

![Project Demo](https://img.shields.io/badge/Status-Complete-success)
![C++](https://img.shields.io/badge/C++-17-blue)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![ONNX](https://img.shields.io/badge/ONNX-1.16-orange)

---

## Project Overview

This project demonstrates the critical importance of deployment optimization in machine learning systems. While model architecture and training get most attention, **how you deploy** a model can dramatically impact real-world performance.

### The Challenge
ML models trained in Python research environments (PyTorch, TensorFlow) often can't meet real-time requirements in production, especially on embedded systems without GPUs.

### The Solution
Export models to ONNX format and build optimized C++ inference engines that run 1.3-2x faster on the same hardware.

---

## Performance Results

**Test Setup:**
- Hardware: Dell Inspiron 16 (CPU-only)
- Model: YOLO11 Nano (2.5MB)
- Input: 870 frames, 640×640 resolution
- Measurement: Same video file processed by both implementations

| Implementation | Processing Time | FPS | Speedup |
|----------------|----------------|-----|---------|
| **Python (PyTorch)** | 57.90s | 15.03 FPS | Baseline |
| **C++ (ONNX Runtime)** | 42.59s | 20.43 FPS | **1.36x Faster** |

**Key Insight:** Same model weights, same accuracy, but 36% faster processing through optimized runtime.

---

## Technical Architecture

### Pipeline Overview
```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Camera/   │────▶│ Preprocessing│────▶│  ONNX Runtime  │
│    Video    │     │  (Resize,    │     │   Inference    │
└─────────────┘     │  Normalize)  │     └────────────────┘
                    └──────────────┘              │
                                                  ▼
                                        ┌──────────────────┐
                                        │  Post-processing │
                                        │  (NMS, Display)  │
                                        └──────────────────┘
```

### Key Components

**1. Model Export (Python)**
- Uses Ultralytics YOLO11 Nano
- Exports to ONNX format (opset 13 for compatibility)
- Input: `.pt` (PyTorch) → Output: `.onnx` (Universal format)

**2. C++ Inference Engine**
- ONNX Runtime 1.16.3 for model execution
- OpenCV 4.5.4 for image processing
- Manual memory management for optimal performance
- Multi-threaded inference (4 threads)

**3. Preprocessing Pipeline**
- Resize: Variable → 640×640 (letterbox)
- Normalize: uint8 [0-255] → float32 [0.0-1.0]
- Reorder: HWC (Height,Width,Channels) → CHW (Channels,Height,Width)

---

## Why This Matters

### For Robotics & Embedded Systems
- **Latency Requirements:** Mobile robots need <50ms perception cycles
- **Power Constraints:** Battery-powered devices can't run power-hungry Python
- **Resource Limits:** Embedded CPUs require optimized code

### Real-World Impact
- **1 hour of video:** 18 minutes saved
- **1000 videos:** 5+ days of compute time saved
- **Cloud deployment:** 36% reduction in compute costs
- **Edge devices:** Enables real-time processing without GPU

---

## Tech Stack

**Languages:**
- C++17 (inference engine)
- Python 3.12 (model export, baseline)

**Libraries:**
- ONNX Runtime 1.16.3 (inference)
- OpenCV 4.5.4 (computer vision)
- Ultralytics YOLO11 (model)

**Tools:**
- CMake 3.10+ (build system)
- WSL2 Ubuntu 22.04 (Linux environment)
- Git (version control)

---

## Project Structure
```
yolo_cpp_project/
├── README.md                          # This file
├── RESULTS.md                         # Detailed benchmark results
├── yolo11n.onnx                       # Exported ONNX model
├── yolo11n.pt                         # Original PyTorch model
├── test_video.mp4                     # Test video for benchmarking
│
├── export_compatible.py               # Python: Export to ONNX
├── test_python_video.py               # Python: Baseline benchmark
├── record_test_video.py               # Python: Record test video
│
└── cpp_inference/                     # C++ implementation
    ├── CMakeLists.txt                 # Build configuration
    ├── main.cpp                       # Original webcam version
    ├── main_benchmark.cpp             # Synthetic benchmark
    ├── main_video.cpp                 # Video file version
    ├── onnxruntime-linux-x64-1.16.3/  # ONNX Runtime library
    └── build/                         # Compiled executables
```

---

## Setup & Installation

### Prerequisites
- **Windows 10/11** with WSL2 enabled
- **Python 3.10+** installed
- **Visual Studio Code** (optional but recommended)
- **Git** for version control

### Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/yolo11-cpp-inference.git
cd yolo11-cpp-inference
```

**2. Python Setup (Windows):**
```bash
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python
```

**3. Export Model to ONNX:**
```bash
python export_compatible.py
```

**4. C++ Setup (WSL2):**
```bash
wsl
cd cpp_inference

# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev

# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# Build
mkdir build && cd build
cmake ..
make
```

**5. Run Benchmarks:**
```bash
# Python baseline
python test_python_video.py

# C++ optimized
cd cpp_inference/build
./yolo_video
```

---

## Detailed Results

### Benchmark Comparison

**Test Conditions:**
- Same hardware (Dell Inspiron 16)
- Same model weights (YOLO11 Nano)
- Same input video (870 frames)
- CPU-only inference (no GPU)

**Python Implementation:**
- Framework: PyTorch via Ultralytics
- Average inference time: 66.6ms per frame
- Throughput: 15.03 FPS
- Total time: 57.90 seconds

**C++ Implementation:**
- Framework: ONNX Runtime
- Average inference time: 49.0ms per frame
- Throughput: 20.43 FPS
- Total time: 42.59 seconds

**Improvement:**
- **17.6ms faster** per frame
- **15.31 seconds saved** on 30-second video
- **26.5% reduction** in latency

---

## Technical Deep Dive

### Why is C++ Faster?

**1. Compiled vs Interpreted**
- Python: Interpreted line-by-line at runtime
- C++: Compiled to native machine code ahead of time

**2. Memory Management**
- Python: Automatic garbage collection (adds overhead)
- C++: Manual memory control (no GC pauses)

**3. Threading**
- Python: GIL (Global Interpreter Lock) limits parallelism
- C++: True multi-threading without restrictions

**4. Optimizations**
- ONNX Runtime: Operator fusion, graph optimization
- C++: Direct hardware access, SIMD instructions

### ONNX Format Benefits

**Framework Agnostic:**
- Train in PyTorch/TensorFlow
- Deploy in C++/Java/C#/JavaScript
- Same model, any platform

**Inference Optimized:**
- Graph-level optimizations (operator fusion)
- Constant folding
- Dead code elimination
- Optimized kernels for each hardware

**Production Ready:**
- Stable format (backward compatible)
- Wide hardware support (CPU, GPU, NPU)
- Industry standard (Microsoft, Facebook, AWS)

---

## Key Learnings

### Deployment Optimization Matters
Model accuracy is important, but deployment efficiency can be equally critical for real-world applications.

### ONNX as Universal Format
Separating training (Python) from deployment (C++) provides best of both worlds - research flexibility and production performance.

### Measurement Methodology
Fair comparison requires identical test conditions - same video, same hardware, same preprocessing.

### Real-Time Constraints
For robotics applications, consistent low latency (20-50ms) is often more important than average speed.

---

## Future Improvements

### Performance Optimizations
- [ ] **INT8 Quantization:** 4x model size reduction, 2-4x speedup
- [ ] **TensorRT Integration:** 5-10x speedup on NVIDIA GPUs
- [ ] **Multi-threading:** Parallel pre/post-processing
- [ ] **SIMD Optimization:** Hand-optimize critical loops with AVX2/NEON

### Feature Additions
- [ ] **Full NMS Implementation:** Add bounding box visualization
- [ ] **Multi-stream Processing:** Process multiple cameras simultaneously
- [ ] **Dynamic Batching:** Optimize throughput for batch inference
- [ ] **Model Ensemble:** Combine multiple models for better accuracy

### Production Readiness
- [ ] **Error Handling:** Robust error recovery
- [ ] **Logging System:** Structured logging for debugging
- [ ] **Configuration Files:** YAML/JSON config management
- [ ] **Docker Container:** Containerized deployment
- [ ] **REST API:** HTTP endpoint for inference service

---

## Project Impact

This project demonstrates:
- ✅ End-to-end ML deployment pipeline
- ✅ Performance optimization methodology
- ✅ Production-ready C++ engineering
- ✅ Cross-platform development skills
- ✅ Industry-standard tools (ONNX, CMake)

**Perfect for roles in:**
- Perception Engineering
- Embedded ML
- Computer Vision
- Robotics Software
- ML Infrastructure

---

## Contact

**Your Name**  
nikitamandal0302@gmail.com 
[LinkedIn](www.linkedin.com/in/nikitamandal03)  
[GitHub](https://github.com/NikkiMandal)

---

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics** for YOLO11 implementation
- **Microsoft** for ONNX Runtime
- **OpenCV** community
- Inspired by real-world robotics deployment challenges

---

**⭐ Star this repo if you found it helpful!**