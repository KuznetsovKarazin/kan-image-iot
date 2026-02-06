import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="tflite_models\\test_model_w0.25_int8.tflite")
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()

for t in tensor_details:
    if t["dtype"] == tf.float32:
        print("Name:", t["name"])
        print("  Index:", t["index"])
        print("  Shape:", t["shape"])
        print("  Dtype:", t["dtype"])
        print("  Quantization:", t["quantization"])
        print()
