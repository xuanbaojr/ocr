import pandas as pd
import numpy as np
import cv2, os
import json

with open("./char_list.json", "r") as file:
    char_list = json.load(file)
    
def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print("No found in char_list :", char)
            
    return dig_lst
TIME_STEPS = 120


dict_ = {}
all_image_paths = []
train_dir = "./dataset/train"
for v_name in os.listdir(train_dir):
    v_path = os.path.join(train_dir, v_name)
    label_path = os.path.join(v_path, "labels.json")
    with open(label_path, "r") as file:
        data = json.load(file)

    for item in data:
        dict_[f"{v_name}_{item}_pre.png"] = str(int(float(data[item])))
    all_image_paths += [os.path.join(v_path, "passed", im_name) for im_name in os.listdir(os.path.join(v_path, "passed")) if "pre" in im_name]
    
print("len(all_image_paths)", len(all_image_paths), all_image_paths[0])
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
i=0
for train_img_path in all_image_paths:
    eroded = cv2.cvtColor(cv2.imread(train_img_path), cv2.COLOR_BGR2GRAY)
    eroded = cv2.resize(eroded, (1167, 118))
    img = np.expand_dims(eroded , axis = 2)
    img = img/255.
    label = dict_[os.path.basename(train_img_path)]     
    train_label_length.append(len(label))
    train_input_length.append(TIME_STEPS)
    training_img.append(img)
    training_txt.append(encode_to_labels(label)) 
    i+=1
    if (i%500 == 0):
        print ("has processed trained {} files".format(i))

# import matplotlib.pyplot as plt
# for i in range(len(training_img[50:70])):
#     plt.figure(figsize=(15,2))
#     plt.imshow(training_img[i][:,:,0], cmap='gray')
#     plt.show()


max_label_len = TIME_STEPS
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = 0)
train_padded_txt[0]


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

# pooling layer with kernel size (2,2) to make the height/2 #(1,9,512)
x = MaxPool2D(pool_size=(3, 1))(x)
 
# # to remove the first dimension of one: (1, 31, 512) to (31, 512) 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(x)
 
# # # bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(512, return_sequences=True, dropout = 0.2))(blstm_1)

# # this is our softmax character proprobility with timesteps 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time

act_model = Model(inputs, outputs)

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')

# define the length of input and label for ctc
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
 
# define a ctc lambda function to take arguments and return ctc_bach_cost
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

#model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)  # all must be equal length to T timesteps
train_label_length = np.array(train_label_length)  # different length (only the same in Captc

print(training_img.shape, train_padded_txt.shape, train_input_length.shape, train_label_length.shape)
batch_size = 16
epochs = 100


callbacks = [
    TensorBoard(
        log_dir='./logs',
        histogram_freq=10,
        profile_batch=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch"),
    ModelCheckpoint(
        filepath=os.path.join('./models.keras'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1),
    EarlyStopping(
        monitor='val_loss',
        min_delta=1e-8,
        patience=20,
        restore_best_weights=True,
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        min_delta=1e-8,
        factor=0.2,
        patience=10,
        verbose=1)
]
callbacks_list = callbacks

valid_label_path = "./dataset/valid/labels.json"
with open(valid_label_path, "r") as file:
    valid_dict = json.load(file)

im_dir = "./dataset/valid/passed"
valid_image_paths = [os.path.join(im_dir, im_name) for im_name in os.listdir(im_dir) if "pre" in im_name]
print("len(all_image_paths)", len(valid_image_paths), valid_image_paths[1])
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
i=0
for valid_im_path in valid_image_paths:
    eroded = cv2.cvtColor(cv2.imread(valid_im_path), cv2.COLOR_BGR2GRAY)
    eroded = cv2.resize(eroded, (1167, 118))
    img = np.expand_dims(eroded , axis = 2)
    img = img/255.
    label = valid_dict[os.path.basename(valid_im_path).split("_")[0]]     
    valid_label_length.append(len(label))
    valid_input_length.append(TIME_STEPS)
    valid_img.append(img)
    valid_txt.append(encode_to_labels(label)) 

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)  # all must be equal length to T timesteps
valid_label_length = np.array(valid_label_length)  # different length (only the same in Captc
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = 0)


history = model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], 
          y=np.zeros(len(training_img)),
          batch_size=batch_size, 
          epochs = epochs,
          validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]),
          verbose = 1, callbacks = callbacks_list)

