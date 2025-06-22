import os
import tensorflow as tf
import csv

# Ensure output folder exists
os.makedirs("model1", exist_ok=True)

# 1) Load your Keras model
model = tf.keras.models.load_model(
    r"C:\Users\AliRAMADAN\Desktop\mobilenetv3_hagrid_finetuned.keras"
)

# 2) Convert to TFLite (with default quantization)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join("model1", "gesture_classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model to {tflite_path}")

# 3) Determine num_classes from the model’s output shape
num_classes = model.output_shape[-1]
print(f"Model predicts {num_classes} classes.")

# 4) Provide the exact class_names order you printed earlier
class_names = [
    "call","dislike","fist","four","like","mute","nothing",
    "ok","one","palm","peace","peace_inverted","rock",
    "stop","stop_inverted","three","three2","two_up","two_up_inverted"
]

# 5) Write label CSV
csv_path = os.path.join("model1", "gesture_labels.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    for name in class_names:
        writer.writerow([name])
print(f"Saved label CSV to {csv_path}")
