#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "YOLO11 C++ Benchmark (No Webcam)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize ONNX Runtime
    std::cout << "\n[1] Initializing ONNX Runtime..." << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    
    // Load the ONNX model
    std::cout << "[2] Loading YOLO11 model..." << std::endl;
    Ort::Session session(env, "yolo11n.onnx", session_options);
    std::cout << " Model loaded successfully!" << std::endl;

    // Create a test image (640x640 with random-ish pattern)
    std::cout << "\n[3] Creating test image..." << std::endl;
    cv::Mat test_image(640, 640, CV_8UC3, cv::Scalar(100, 150, 200));
    // Add some patterns so it's not just solid color
    cv::rectangle(test_image, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 0, 0), -1);
    cv::rectangle(test_image, cv::Point(350, 350), cv::Point(500, 500), cv::Scalar(0, 255, 0), -1);
    std::cout << " Test image created!" << std::endl;

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    
    // Prepare input tensor
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    size_t input_tensor_size = 1 * 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);

    // Preprocess the test image once
    std::cout << "\n[4] Preprocessing..." << std::endl;
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 640; h++) {
            for (int w = 0; w < 640; w++) {
                input_tensor_values[c * 640 * 640 + h * 640 + w] = 
                    test_image.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
    std::cout << " Preprocessing complete!" << std::endl;

    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_shape.data(), input_shape.size()
    );

    // Run benchmark
    std::cout << "\n[5] Running benchmark (1000 iterations)..." << std::endl;
    std::cout << "This will take about 30-60 seconds..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    int iterations = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1
        );
        
        // Progress indicator
        if ((i + 1) % 100 == 0) {
            std::cout << "Completed: " << (i + 1) << "/" << iterations << " iterations" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    // Calculate results
    double total_seconds = total_time / 1000.0;
    double fps = iterations / total_seconds;
    double ms_per_frame = total_time / (double)iterations;

    // Final results
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << " FINAL RESULTS:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
    std::cout << "Total time: " << total_seconds << " seconds" << std::endl;
    std::cout << "Average time per inference: " << ms_per_frame << " ms" << std::endl;
    std::cout << " C++ FPS: " << fps << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n COMPARISON:" << std::endl;
    std::cout << "Python baseline: 11.58 FPS" << std::endl;
    std::cout << "C++ optimized: " << fps << " FPS" << std::endl;
    std::cout << "Speedup: " << (fps / 11.58) << "x faster!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}