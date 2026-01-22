#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <chrono>

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "YOLO11 C++ Test (Video File)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize ONNX Runtime
    std::cout << "\n[1] Initializing ONNX Runtime..." << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    
    std::cout << "[2] Loading YOLO11 model..." << std::endl;
    Ort::Session session(env, "yolo11n.onnx", session_options);
    std::cout << " Model loaded successfully!" << std::endl;

    // Open video file
    std::cout << "\n[3] Opening test video..." << std::endl;
    cv::VideoCapture cap("test_video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open test_video.mp4!" << std::endl;
        return -1;
    }
    
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << " Video opened! Total frames: " << total_frames << std::endl;

    // Setup
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    
    std::vector<int64_t> input_shape = {1, 3, 640, 640};
    size_t input_tensor_size = 1 * 3 * 640 * 640;
    std::vector<float> input_tensor_values(input_tensor_size);

    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    double fps = 0.0;

    std::cout << "\n[4] Processing video..." << std::endl;
    std::cout << "Press 'q' to stop early" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // Resize
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 640));

        // Preprocess
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 640; h++) {
                for (int w = 0; w < 640; w++) {
                    input_tensor_values[c * 640 * 640 + h * 640 + w] = 
                        resized.at<cv::Vec3b>(h, w)[c] / 255.0f;
                }
            }
        }

        // Create tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size()
        );

        // Inference
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
            output_names, 1
        );

        frame_count++;
        
        // Update FPS
        if (frame_count % 10 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - start_time
            ).count();
            fps = (frame_count * 1000.0) / elapsed;
        }

        // Display
        std::string fps_text = "C++ FPS: " + std::string(std::to_string(fps).substr(0, 5));
        cv::putText(frame, fps_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::imshow("YOLO11 C++ - Video Test", frame);
        
        if (cv::waitKey(1) == 'q') break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();
    double final_fps = (frame_count * 1000.0) / total_time;

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << " C++ RESULTS (Video File):" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Frames processed: " << frame_count << std::endl;
    std::cout << "Total time: " << total_time / 1000.0 << " seconds" << std::endl;
    std::cout << " C++ FPS: " << final_fps  << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}