
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk


# example of loading a pix2pix model and using it for one-off image translation
from PIL import Image
from keras.models import load_model
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
#from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
 
# load an image
def load_image(filename, size=(256,256)):
 # load image with the preferred size
 pixels = load_img(filename, target_size=size)
 # convert to numpy array
 pixels = img_to_array(pixels)
 # scale from [0,255] to [-1,1]
 pixels = (pixels - 127.5) / 127.5
 # reshape to 1 sample
 pixels = expand_dims(pixels, 0)
 return pixels
 
# load source image
src_image = load_image("xyz.jpg")
print('Loaded', src_image.shape)
# load model
model = load_model('model_010960.h5')
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
print(gen_image)
# plot the image
pyplot.imshow(gen_image[0])
pyplot.axis('off')
filename1 = 'gen_xyz.jpg'
pyplot.savefig(filename1)



