import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

IMG_WIDTH = 256
IMG_HEIGHT = 256

def downsample(filters, size, apply_batchnorm=True):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(
			tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
														 kernel_initializer=initializer, use_bias=False))

	if apply_batchnorm:
		result.add(tf.keras.layers.BatchNormalization())

	result.add(tf.keras.layers.LeakyReLU())

	return result

def upsample(filters, size, apply_dropout=False):
	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(
		tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
																		padding='same',
																		kernel_initializer=initializer,
																		use_bias=False))

	result.add(tf.keras.layers.BatchNormalization())

	if apply_dropout:
			result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

def generate_images(model, test_input, tar, img_dir = "img", i=0):
		#print(test_input.shape)
		prediction = model(test_input, training=True)
		plt.figure(figsize=(15, 15))

		#print(tar.shape)
		tar_img = tf.keras.layers.Concatenate()([tar[0, ...], test_input[0, ...]])
		tar_img = tar_img * 0.5 + 0.5
		tar_img = tf.image.hsv_to_rgb(tar_img)
		#print(test_input[0][0][0])
		pred_img = tf.keras.layers.Concatenate()([prediction[0, ...], test_input[0, ...]])
		pred_img = pred_img * 0.5 + 0.5
		pred_img = tf.image.hsv_to_rgb(pred_img)

		display_list = [test_input[0] * 0.5 + 0.5, tar_img, pred_img]
		title = ['Input Image', 'Ground Truth', 'Predicted Image']

		for i in range(3):
				plt.subplot(1, 3, i+1)
				plt.title(title[i])
				# Getting the pixel values in the [0, 1] range to plot.
				if(i == 0):
						plt.imshow(display_list[i], cmap='gray')
				else:
						plt.imshow(display_list[i])
						plt.axis('off')
		plt.savefig(img_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+str(i)+".jpg")
		plt.show()
		
		
def generate_images2(model, example_images, img_dir = "img"):
	#print(test_input.shape)
	fig, ax = plt.subplots(10, 3, figsize=(9, 30))
	for idx, (inp, tar) in enumerate(example_images):

		prediction = model(inp, training=True)
		#print(tar.shape)
		tar_img = tf.keras.layers.Concatenate()([tar[0, ...], inp[0, ...]])
		tar_img = tar_img * 0.5 + 0.5
		tar_img = tf.image.hsv_to_rgb(tar_img)
		#print(test_input[0][0][0])
		pred_img = tf.keras.layers.Concatenate()([prediction[0, ...], inp[0, ...]])
		pred_img = pred_img * 0.5 + 0.5
		pred_img = tf.image.hsv_to_rgb(pred_img)

		display_list = [inp[0] * 0.5 + 0.5, tar_img, pred_img]
		title = ['Input Image', 'Ground Truth', 'Predicted Image']

		for i in range(3):
		#plt.subplot(idx + 1, 3, i+1)
			if idx == 0:
				ax[idx,i].set_title(title[i])
		# Getting the pixel values in the [0, 1] range to plot.
			if(i == 0):
				ax[idx, i].imshow(display_list[i], cmap='gray')
			else:
				ax[idx, i].imshow(display_list[i])
			ax[idx, i].axis('off')
	fig.savefig(img_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".jpg")
	fig.savefig(img_dir + "latest.jpg")


def load(image_file):
	# Read and decode an image file to a uint8 tensor
	image = tf.io.read_file(image_file)
	image = tf.io.decode_jpeg(image)

	# Split each image tensor into two tensors:
	# - one with a real building facade image
	# - one with an architecture label image 
	input_image = tf.image.rgb_to_grayscale(image)
	real_image = image

	# Convert both images to float32 tensors
	input_image = tf.cast(input_image, tf.float32)
	real_image = tf.cast(real_image, tf.float32)


	real_image = tf.image.rgb_to_hsv(real_image/255.0)
	real_image = real_image * 255.0

	# real image is only the first two channels
	real_image = real_image[:,:,:2]

	return input_image, real_image

	### Augmentation

def resize(real_image, height, width):
	real_image = tf.image.resize(real_image, [height, width],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return real_image

def resize_2_images(input_image, real_image, height, width):
	real_image = tf.image.resize(real_image, [height, width],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	input_image = tf.image.resize(input_image, [height, width],
															method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return input_image, real_image

def random_crop(real_image):
	cropped_image = tf.image.random_crop(
		real_image, size=[1, IMG_HEIGHT, IMG_WIDTH, 3])

	return cropped_image[0]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
	input_image = (input_image / 127.5) - 1
	real_image = (real_image / 127.5) - 1

	return input_image, real_image

def load_image_train(image_file):
	input_image, real_image = load(image_file)
	#input_image, real_image = random_jitter(real_image)
	input_image, real_image = resize_2_images(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image

def load_image_test(image_file):
	input_image, real_image = load(image_file)
	input_image, real_image = resize_2_images(input_image, real_image,
																	IMG_HEIGHT, IMG_WIDTH)
	input_image, real_image = normalize(input_image, real_image)

	return input_image, real_image