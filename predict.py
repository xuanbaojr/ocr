from ultralytics import YOLO
model = YOLO("./weights/yolo.pt")
import json

with open("./char_list.json", "r") as file:
    char_list = json.load(file)
# import our model, different layers and activation function 
from tensorflow.keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional, Add, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu, sigmoid, softmax
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

inputs = Input(shape=(118,1167,1))
 
# Block 1
x = Conv2D(64, (3,3), padding='same')(inputs)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_1 = x 

# Block 2
x = Conv2D(128, (3,3), padding='same')(x)
x = MaxPool2D(pool_size=3, strides=3)(x)
x = Activation('relu')(x)
x_2 = x

# Block 3
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_3 = x

# Block4
x = Conv2D(256, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_3])
x = Activation('relu')(x)
x_4 = x

# Block5
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x_5 = x

# Block6
x = Conv2D(512, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = Add()([x,x_5])
x = Activation('relu')(x)

# Block7
x = Conv2D(1024, (3,3), padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3, 1))(x)
x = Activation('relu')(x)

x = MaxPool2D(pool_size=(3, 1))(x)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
act_model = Model(inputs, outputs)

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

act_model.load_weights('./weights/models.keras')

import numpy as np
import os, cv2
im_dir = "./input/valid"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

ims = [os.path.join(im_dir, im_name) for im_name in os.listdir(im_dir)]
import os
import cv2
import numpy as np


def evaluate():
    def get_result(image):
        try:
            # Tiền xử lý ảnh
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            gray = cv2.resize(gray, (int(118 / height * width), 118))
            height, width = gray.shape
            gray = np.pad(gray, ((0, 0), (0, 1167 - width)), 'median')
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            eroded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 9, -1)
            eroded = cv2.resize(eroded, (1167, 118))
            img = np.expand_dims(eroded, axis=2)
            img_ = img / 255.0

            # Chuẩn bị dữ liệu và dự đoán
            valid_img = np.expand_dims(img_, axis=0)  # Đảm bảo đúng kích thước [batch, height, width, channels]
            prediction = act_model.predict(valid_img)
            out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                                        greedy=True)[0][0])

            # Chuyển đổi kết quả dự đoán thành chuỗi
            for x in out:
                pred = [char_list[int(p)] for p in x if int(p) != -1]
                string_ = "".join(pred)
            return string_

        except Exception as e:
            print(f"Error during prediction: {e}")
            return ""

    for im_path in ims:
        try:
            im = cv2.imread(im_path)
            results = model(source=im)  # Phát hiện bounding box
            
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    image = im[y1:y2, x1:x2]  # Cắt vùng bounding box
                    
                    # Gọi hàm predict để dự đoán chuỗi
                    string_ = get_result(image)

                    # Vẽ bounding box và chuỗi kết quả
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im, string_, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)
            
            # Lưu ảnh đã xử lý
            output_path = os.path.join(output_dir, os.path.basename(im_path))
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, im)
            print(f"Processed {im_path}, saved to {output_path}")

        except Exception as e:
            print(f"Error processing {im_path}: {e}")

