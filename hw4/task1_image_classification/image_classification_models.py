from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

from keras.preprocessing import image
import numpy as np

models = {
    # 'InceptionV3': InceptionV3,
    # 'InceptionResNetV2': InceptionResNetV2,
    # 'Xception': Xception,
    # 'MobileNet': MobileNet,
    # 'MobileNetV2': MobileNetV2,
    # 'DenseNet121': DenseNet121,
    # 'ResNet50': ResNet50,
    # 'VGG16': VGG16,
    # 'VGG19': VGG19,
}
for key, model_type in models.items():
    model = model_type(weights='imagenet')

    img_path = 'input_images/lemur.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    # Convert image into numpy array of pixels and prepare input for recognition
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # print the classification: the image class and probability for the top-5 highest
    decoded_predictions = decode_predictions(predictions, top=5)
    print('{} model predicted:\n'.format(key))
    for i in range(0, len(decoded_predictions[0])):
        result = decoded_predictions[0][i]
        print('%s (%.2f%%);' % (result[1], result[2] * 100))
