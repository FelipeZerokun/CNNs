from keras.models import load_model
from keras.preprocessing import image
import numpy as np

try:
    cnn = load_model('model/cnn_model.h5')
    test_image = image.load_img('single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image/255.0)
    if result[0][0] > 0.5:
        print(f"I am {result[0][0]*100}% sure it is a dog!")
    else:
        print(f"I am {(1 -result[0][0])*100}% sure it is a cat!")


except:
    print("The model cannot be imported")


