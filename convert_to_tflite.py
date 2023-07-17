import tensorflow as tf

# Convert the model
saved_model_dir = "/home/gab/PycharmProjects/tf_similarity/model_efficientNet"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model_efficientNet.tflite', 'wb') as f:
  f.write(tflite_model)