"""
Inference on Raspberry Pi.

Author: Daniele Faggi
Date: February 2026

Usage:
    # Scan all experiments
    python scripts/generate_models_csv.py

    # Debug mode: limit to 2 experiments
    python scripts/generate_models_csv.py --limit 2

    # Custom output CSV
    python scripts/generate_models_csv.py --output my_results.csv

    # Custom data directory
    python scripts/generate_models_csv.py --data_dir data/processed/vww_subset/test
"""
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import threading
import time
import argparse

# 1. Base Configuration
MODEL_PATH = "model.tflite"
input_shape = (224, 224)

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        # Start a thread to read frames
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def preprocess_image(frame, target_size, debug=False):

    if debug:
        cv2.imwrite("capture.jpg", frame)

    # Change color space from BGR (OpenCV) to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize image to 224x224
    img = cv2.resize(img, target_size)
    
    # Normalization: transform pixels from [0, 255] to [0.0, 1.0]
    # Note: Some models want [-1, 1], in that case use: (img / 127.5) - 1.0
    img = img.astype(np.float32) / 255.0

    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # Normalize with ImageNet mean and std
    img = (img - mean) / std    

    # Add batch dimension: from (224, 224, 3) to (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    return img


def inference(model_path, input_shape, camera_id, debug, callback=None):
    """
    Run inference on the Raspberry Pi.
    """ 

    # Initialize the interpreter
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    vs = VideoStream(src=camera_id).start()
    time.sleep(2.0) # Time to warm up camera
    print("Capturing image")

    try:
        while True:
            frame = vs.read()
            if frame is not None:
                if(debug):
                    print("Preprocessing")
                input_data = preprocess_image(frame, input_shape, debug)
                if(debug):
                    print("TFLite invoke")

                # 2. Execution 
                interpreter.set_tensor(input_details[0]['index'], input_data)
                start_time = time.perf_counter()
                interpreter.invoke()
                end_time = time.perf_counter()
                
                # 3. Output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Find the index of the class with the highest score
                predicted_class_index = np.argmax(output_data[0])
                if callback == None:
                    print("Predicted class:", predicted_class_index)
                else:
                    callback(predicted_class_index)
                
                last_inference_time = (end_time - start_time)
                inference_time = last_inference_time * 1000

                if(debug):
                    # Debug
                    print(f"Raw output: {output_data}, Predicted class: {predicted_class_index}, Inference time: {inference_time:.2f} ms")
                    # Show image
                    cv2.imshow("Live AI", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on the Raspberry Pi.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the TFLite model.')
    parser.add_argument('--input_shape', type=int, default=input_shape, help='Input shape for the model.')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID.')
    parser.add_argument('--debug', type=str, default=False, help='Debug mode.')
    args = parser.parse_args()
    inference(args.model_path, args.input_shape, args.camera_id, args.debug)