from keras.utils.image_utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

""" ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]
labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
"""
labels = [
    "\u091E",
    "\u091F",
    "\u0920",
    "\u0921",
    "\u0922",
    "\u0923",
    "\u0924",
    "\u0925",
    "\u0926",
    "\u0927",
    "\u0915",
    "\u0928",
    "\u092A",
    "\u092B",
    "\u092c",
    "\u092d",
    "\u092e",
    "\u092f",
    "\u0930",
    "\u0932",
    "\u0935",
    "\u0916",
    "\u0936",
    "\u0937",
    "\u0938",
    "\u0939",
    "ksha",
    "tra",
    "gya",
    "\u0917",
    "\u0918",
    "\u0919",
    "\u091a",
    "\u091b",
    "\u091c",
    "\u091d",
    "\u0966",
    "\u0967",
    "\u0968",
    "\u0969",
    "\u096a",
    "\u096b",
    "\u096c",
    "\u096d",
    "\u096e",
    "\u096f",
]
#
import numpy as np
from keras.preprocessing import image

test_image = cv2.imread("./tese images/abc.png")
image = cv2.resize(test_image, (32, 32))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=0)
image = np.expand_dims(image, axis=3)
print("[INFO] loading network...")
import tensorflow as tf

model = tf.keras.models.load_model("./HindiModel2.h5")
lists = model.predict(image)[0]
print("The letter is ", labels[np.argmax(lists)])
