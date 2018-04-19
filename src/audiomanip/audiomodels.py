import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Reshape
from keras.layers import Dropout, GRU, ELU, Permute
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalMaxPooling2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

# @Class: ModelZoo
# @Description: Set of models to use to solve the classification problem.
class ModelZoo(object):
  # @Method: cnn_melspect
  # @Description: 
  #  Method used for classify data from GTZAN in the 
  # MelSpectrogram input format.
  @staticmethod
  def cnn_melspect_1D(input_shape):
    kernel_size = 3
   # activation_func = LeakyReLU()
    activation_func = Activation('relu')
    inputs = Input(input_shape)
    print('input_shape: ', input_shape)

    # Convolutional block_1
    conv1 = Conv1D(32, kernel_size, kernel_initializer='he_normal')(inputs)
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

    # Convolutional block_2
    conv2 = Conv1D(64, kernel_size, kernel_initializer='he_normal')(pool1)
    act2 = activation_func(conv2)
    bn2 = BatchNormalization()(act2)
    pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

    # Convolutional block_3
    conv3 = Conv1D(128, kernel_size, kernel_initializer='he_normal')(pool2)
    act3 = activation_func(conv3)
    bn3 = BatchNormalization()(act3)
    
    
       
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(bn3)
    gmeanpl = GlobalAveragePooling1D()(bn3)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

    # Regular MLP
    dense1 = Dense(512,
        kernel_initializer='he_normal', # 'glorot_normal',
        bias_initializer='he_normal' # 'glorot_normal'
                  )(mergedlayer)
    actmlp = activation_func(dense1)
    reg = Dropout(0.5)(actmlp)

    dense2 = Dense(512,
        kernel_initializer='he_normal',   #  'glorot_normal',
        bias_initializer='he_normal', # 'glorot_normal'
                  )(reg)
    actmlp = activation_func(dense2)
    reg = Dropout(0.5)(actmlp)
    
    dense2 = Dense(10, activation='softmax')(reg)

    model = Model(inputs=[inputs], outputs=[dense2])
    return model

  # @Method: cnn_melspect
  # @Description: 
  #  Method used for classify data from GTZAN in the 
  # MelSpectrogram input format.
  @staticmethod
  def cnn_melspect_2D(input_shape):
    kernel_size = [3, 3]
    maxpool_size = [2,  2]
    stride_one = [1, 1]
    stride_two = [2, 2]
    num_classes = 10

   # activation_func = LeakyReLU()
    activation_func = Activation('relu')
    inputs = Input(input_shape)
    print('input_shape: ', input_shape)


    
    # Convolutional block_1
    conv1 = Conv2D(32, kernel_size, kernel_initializer='he_normal')(inputs)
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling2D(pool_size=maxpool_size, strides=stride_two)(bn1)



    # Convolutional block_2
    conv2 = Conv2D(64, kernel_size, kernel_initializer='he_normal')(pool1)
    act2 = activation_func(conv2)
    bn2 = BatchNormalization()(act2)
    pool2 = MaxPooling2D(pool_size=maxpool_size, strides=stride_two)(bn2)

    

    # Convolutional block_3
    conv3 = Conv2D(128, kernel_size, kernel_initializer='he_normal')(pool2)
    act3 = activation_func(conv3)
    bn3 = BatchNormalization()(act3)
    
    

    # Global Layers
    gmaxpl = GlobalMaxPooling2D()(bn3)
    gmeanpl = GlobalAveragePooling2D()(bn3)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)


    # Regular MLP
    dense1 = Dense(512,
        kernel_initializer='he_normal', # 'glorot_normal',
        bias_initializer='he_normal' # 'glorot_normal'
                  )(mergedlayer)
    actmlp = activation_func(dense1)
    reg = Dropout(0.5)(actmlp)


    dense2 = Dense(512,
        kernel_initializer='he_normal',   #  'glorot_normal',
        bias_initializer='he_normal', # 'glorot_normal'
                  )(reg)
    actmlp = activation_func(dense2)
    reg = Dropout(0.5)(actmlp)
    

    dense2 = Dense(num_classes, activation='softmax')(reg)
    model = Model(inputs=[inputs], outputs=[dense2])
    return model

  # @Method: crnn_melspect
  # @Description:
  #  Method used for classify data from GTZAN in the
  # MelSpectrogram input format.
  @staticmethod
  def crnn_melspect_2D(input_shape):
    pair_two = [2, 2]
    pair_three = [3, 3]
    pair_four = [4, 4]

    num_classes = 10
    drop_ratio = 0.1
    channel_axis = 1

    # activation_func = LeakyReLU()
    activation_func = Activation('relu')
    inputs = Input(input_shape)
    print('input_shape: ', input_shape)

    # Convolutional block_1
    conv1 = Conv2D(64, kernel_size=pair_three, name='conv1')(inputs)
    bn1 = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(conv1)
    elu1 = ELU()(bn1)
    pool1 = MaxPooling2D(pool_size=pair_two, strides=pair_two, name='pool1')(elu1)
    dr1 = Dropout(drop_ratio, name='dropout1')(pool1)

    # Convolutional block_2
    conv2 = Conv2D(128, kernel_size=pair_three,  name='conv2')(dr1)
    bn2 = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(conv2)
    elu2 = ELU()(bn2)
    pool2 = MaxPooling2D(pool_size=pair_two, strides=pair_two, name='pool2')(elu2)
    dr2 = Dropout(drop_ratio, name='dropout2')(pool2)

    # Convolutional block_3
    conv3 = Conv2D(128, kernel_size=pair_three, name='conv3')(dr2)
    bn3 = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(conv3)
    elu3 = ELU()(bn3)
    pool3 = MaxPooling2D(pool_size=pair_three, strides=pair_three, name='pool3')(elu3)
    dr3 = Dropout(drop_ratio, name='dropout3')(pool3)

    # Convolutional block_4
    conv4 = Conv2D(128, kernel_size=pair_three,  name='conv4')(dr3)
    bn4 = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(conv4)
    elu4 = ELU()(bn4)
    pool4 = MaxPooling2D(pool_size=pair_four, strides=pair_four, name='pool4')(elu4)
    dr4 = Dropout(drop_ratio, name='dropout4')(pool4)
    print('dr4shape:', K.get_variable_shape(dr4))

    # Reshaping
    # x = Permute((3, 1, 2))(dr4)

    rs = Reshape((25, 128))(dr4)    # 15, 128

    # GRU block 1, 2, output
    gru1 = GRU(32, return_sequences=True, name='gru1')(rs)
    gru2 = GRU(32, return_sequences=False, name='gru2')(gru1)
    reg = Dropout(0.3)(gru2)

    dense2 = Dense(num_classes, activation='sigmoid', name='output')(reg)
    model = Model(inputs=[inputs], outputs=[dense2])
    model.summary()
    return model
