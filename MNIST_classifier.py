# Import libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from kerastuner import RandomSearch
from mlflow import log_param, log_metric, log_artifact, start_run

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# Define hyperparameter search space
def build_model(hp):
  model = keras.Sequential()
  model.add(layers.Flatten(input_shape=(28, 28)))
  for _ in range(hp.Int('n_hidden_layers', 1, 3)):
    model.add(layers.Dense(units=hp.Int('units_per_layer', 32, 128, step=16), activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# Use MLFlow for experiment tracking
with start_run():
  
  # Log hyperparameters
  log_param('n_hidden_layers', hp.Int('n_hidden_layers', 1, 3))
  log_param('units_per_layer', hp.Int('units_per_layer', 32, 128, step=16))

  # Create and run the tuner
  tuner = RandomSearch(
      build_model,
      objective='val_accuracy',
      max_trials=10,
      executions_per_trial=3,
  )

  # Train the model with hyperparameter tuning
  tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[TensorBoard()])

  # Get the best model
  best_model = tuner.get_best_models()[0]

  # Evaluate the best model
  test_loss, test_acc = best_model.evaluate(test_images, test_labels)

  # Log metrics
  log_metric('test_loss', test_loss)
  log_metric('test_accuracy', test_acc)

  # Log model as artifact
  best_model.save('best_model.h5')
  log_artifact('best_model.h5')

# Print final results
print("Test accuracy:", test_acc)
