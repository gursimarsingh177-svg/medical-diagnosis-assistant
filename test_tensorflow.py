import tensorflow as tf


def get_tensorflow_info():
	"""Return TensorFlow version and the devices available to it."""
	return {
		"version": tf.__version__,
		"devices": tf.config.list_physical_devices(),
	}


if __name__ == "__main__":
	info = get_tensorflow_info()
	print("TensorFlow version:", info["version"])
	print("Available devices:", info["devices"])