import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras import backend as K

class UNetSubModel():
    def __init__(self, input_layer, output_channels, kernel_sizes):
        self.input_layer = input_layer

        self.conv1 = Conv2D(filters = 32, kernel_size = (kernel_sizes[0], kernel_sizes[0]), padding = 'same')(self.input_layer)
        self.conv1 = LeakyReLU(alpha = 0.1)(self.conv1)
        self.conv1 = Conv2D(filters = 32, kernel_size = (kernel_sizes[0], kernel_sizes[0]), padding = 'same')(self.conv1)
        self.conv1 = LeakyReLU(alpha = 0.1)(self.conv1)
        self.pool1 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv1)

        self.conv2 = Conv2D(filters = 64, kernel_size = (kernel_sizes[1], kernel_sizes[1]), padding = 'same')(self.pool1)
        self.conv2 = LeakyReLU(alpha = 0.1)(self.conv2)
        self.conv2 = Conv2D(filters = 64, kernel_size = (kernel_sizes[1], kernel_sizes[1]), padding = 'same')(self.conv2)
        self.conv2 = LeakyReLU(alpha = 0.1)(self.conv2)
        self.pool2 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv2)

        self.conv3 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(self.pool2)
        self.conv3 = LeakyReLU(alpha = 0.1)(self.conv3)
        self.conv3 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(self.conv3)
        self.conv3 = LeakyReLU(alpha = 0.1)(self.conv3)
        self.pool3 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv3)

        self.conv4 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(self.pool3)
        self.conv4 = LeakyReLU(alpha = 0.1)(self.conv4)
        self.conv4 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(self.conv4)
        self.conv4 = LeakyReLU(alpha = 0.1)(self.conv4)
        self.pool4 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv4)

        self.conv5 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.pool4)
        self.conv5 = LeakyReLU(alpha = 0.1)(self.conv5)
        self.conv5 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.conv5)
        self.conv5 = LeakyReLU(alpha = 0.1)(self.conv5)
        self.pool5 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv5)

        self.conv6 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.pool5)
        self.conv6 = LeakyReLU(alpha = 0.1)(self.conv6)
        self.conv6 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.conv6)
        self.conv6 = LeakyReLU(alpha = 0.1)(self.conv6)
        #self.pool6 = AveragePooling2D(pool_size = (2, 2), strides = 2)(self.conv6)

        self.deconv1 = UpSampling2D()(self.conv6)
        self.deconv1 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.deconv1)
        self.deconv1 = LeakyReLU(alpha = 0.1)(self.deconv1)
        self.deconv1 = Concatenate()([self.deconv1, self.conv5])
        self.deconv1 = Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same')(self.deconv1)
        self.deconv1 = LeakyReLU(alpha = 0.1)(self.deconv1)

        self.deconv2 = UpSampling2D()(self.deconv1)
        self.deconv2 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(self.deconv2)
        self.deconv2 = LeakyReLU(alpha = 0.1)(self.deconv2)
        self.deconv2 = Concatenate()([self.deconv2, self.conv4])
        self.deconv2 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(self.deconv2)
        self.deconv2 = LeakyReLU(alpha = 0.1)(self.deconv2)

        self.deconv3 = UpSampling2D()(self.deconv2)
        self.deconv3 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(self.deconv3)
        self.deconv3 = LeakyReLU(alpha = 0.1)(self.deconv3)
        self.deconv3 = Concatenate()([self.deconv3, self.conv3])
        self.deconv3 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(self.deconv3)
        self.deconv3 = LeakyReLU(alpha = 0.1)(self.deconv3)

        self.deconv4 = UpSampling2D()(self.deconv3)
        self.deconv4 = Conv2D(filters = 64, kernel_size = (kernel_sizes[1], kernel_sizes[1]), padding = 'same')(self.deconv4)
        self.deconv4 = LeakyReLU(alpha = 0.1)(self.deconv4)
        self.deconv4 = Concatenate()([self.deconv4, self.conv2])
        self.deconv4 = Conv2D(filters = 64, kernel_size = (kernel_sizes[0], kernel_sizes[0]), padding = 'same')(self.deconv4)
        self.deconv4 = LeakyReLU(alpha = 0.1)(self.deconv4)

        self.deconv5 = UpSampling2D()(self.deconv4)
        self.deconv5 = Conv2D(filters = 32, kernel_size = (7, 7), padding = 'same')(self.deconv5)
        self.deconv5 = LeakyReLU(alpha = 0.1)(self.deconv5)
        self.deconv5 = Concatenate()([self.deconv5, self.conv1])
        self.deconv5 = Conv2D(filters = 32, kernel_size = (7, 7), padding = 'same')(self.deconv5)
        self.deconv5 = LeakyReLU(alpha = 0.1)(self.deconv5)

        self.output_layer = Conv2D(filters = output_channels, kernel_size = (1, 1))(self.deconv5)
        self.output_layer = LeakyReLU(alpha = 0.1)(self.output_layer)

    def get_output(self):
        return self.conv6, self.output_layer
    
class Network():
    def __init__(self, height = 512, width = 512, input_channels = 6, timestamp = 0.5):
        self.frame_1_input = Input(shape = (height, width, input_channels))(frame_1)
        self.frame_2_input = Input(shape = (height, width, input_channels))
        self.timestamp_input = Input(shape = (1, 1, 1))

        self.input_1 = Concatenate()([self.frame_1_input, self.frame_2_input])
        #self.input_layer_1 = Input()(self.input_1)
        self.unet_encoding_output, self.unet_final_output = UNetSubModel(self.input_1, output_channels = 4, kernel_sizes = (7, 5)).get_output()
        #self.unet_final_output = LeakyReLU(alpha = 0.1)(self.unet_final_output)

        self.output_1, self.output_2 = self.unet_final_output[:, :, :, :2], self.unet_final_output[:, :, :, 2:]
        self.output_1_1 = (-1 * (1 - self.timestamp_input) * self.timestamp_input * self.output_1) + (self.timestamp_input * self.timestamp_input * self.output_2)
        self.output_2_1 = ((1 - self.timestamp_input) * (1 - self.timestamp_input) * self.output_1) - (self.timestamp_input * (1 - self.timestamp_input) * self.output_2)

        self.input_2 = Concatenate(axis = 3)([self.frame_1_input, self.frame_2_input, backward_warping(self.frame_2_input, self.output_2_1), backward_warping(self.frame_1_input, self.output_1_1), self.output_1_1, self.output_2_1])

        _, self.output_2 = UNetSubModel(self.input_2, output_channels = 5, kernel_sizes = (3, 3)).get_output()

        self.delta_1, self.delta_2, self.visibility_1 = self.output_2[:, :, :, :2], self.output_2[:, :, :, 2:4], self.output_2[:, :, :, 4:5]
        #self.delta_1 = LeakyReLU(alpha = 0.1)(self.delta_1)
        #self.delta_2 = LeakyReLU(alpha = 0.1)(self.delta_2)
        self.visibility_1 = Activation('sigmoid')(self.visibility_1)
        self.visibility_1 = K.tile(self.visibility_1, (1, 1, 1, input_channels))

        self.visibility_2 = 1 - self.visibility_1

        self.output_1_2 = self.output_1_1 + self.delta_1
        self.output_2_2 = self.output_2_1 + self.delta_2

        self.output = ((1 - self.timestamp_input) * self.visibility_1) * backward_warping(self.frame_1_input, self.output_1_2) + (self.timestamp_input * self.visibility_2) * backward_warping(self.frame_2_input, self.output_2_2)
        self.output = (self.output) / ((1 - self.timestamp_input) * self.visibility_1 + self.timestamp_input * self.visibility_2 + 1e-12)

        self.model = Model(inputs = [self.timestamp_input, self.frame_1_input, self.frame_2_input], outputs = self.output)
        self.intermediate_values = [self.frame_1_input, self.frame_2_input, self.output_1, self.output_2, self.output_1_1, self.output_2_1]
    
    def get_model(self):
        return self.model
    
    def get_intermediate_values(self):
        return self.intermediate_values

def backward_warping(frame, output, crop_size = None, perform_resize = False, normalize = False, output_type = 'CONSTANT'):
    shape = tf.shape(frame)

    if(crop_size is None):
        H = shape[1]
        W = shape[2]
        H_1 = shape[1]
        W_1 = shape[2]
        h = 0
        w = 0
    else:
        H = shape[1]
        W = shape[2]
        H_1 = crop_size[1] - crop_size[0]
        W_1 = crop_size[3] - crop_size[2]
        h = crop_size[0]
        w = crop_size[2]
    
    if(perform_resize):
        output = tf.image.resize_bilinear(output, [H, W])

    if(output_type == 'CONSTANT'):
        frame = tf.pad(frame, ((0, 0), (1, 1), (1, 1), (0, 0)), mode = 'CONSTANT')
    elif(output_type == 'EDGE'):
        frame = tf.pad(frame, ((0, 0), (1, 1), (1, 1), (0, 0)), mode = 'REFLECT')

    output_y, output_x = tf.split(output, 2, axis = 3)
    n, h, w = get_grid(shape[0], H, W, h, w)

    if(normalize):
        output_y = output_y * tf.cast(H, dtype = tf.float32)
        output_y /= 2
        output_x = output_y * tf.cast(W, dtype = tf.float32)
        output_x /= 2

    output_x_0 = tf.floor(output_x)
    output_y_0 = tf.floor(output_y)
    output_x_1 = output_x_0 + 1
    output_y_1 = output_y_0 + 1

    H_2 = tf.cast(H_1 + 1, tf.float32)
    W_2 = tf.cast(W_1 + 1, tf.float32)
    iy_0 = tf.clip_by_value(output_y_0 + h, 0, H_2)
    ix_0 = tf.clip_by_value(output_x_0 + w, 0, W_2)
    iy_1 = tf.clip_by_value(output_y_1 + h, 0, H_2)
    ix_1 = tf.clip_by_value(output_x_1 + w, 0, W_2)

    i_00 = tf.concat([n, iy_0, ix_0], axis = 3)
    i_00 = tf.cast(i_00, dtype = tf.int32)
    i_01 = tf.concat([n, iy_1, ix_0], axis = 3)
    i_01 = tf.cast(i_01, dtype = tf.int32)
    i_10 = tf.concat([n, iy_0, ix_1], axis = 3)
    i_10 = tf.cast(i_10, dtype = tf.int32)
    i_11 = tf.concat([n, iy_1, ix_1], axis = 3)
    i_11 = tf.cast(i_11, dtype = tf.int32)

    x_00 = tf.gather_nd(frame, i_00)
    x_01 = tf.gather_nd(frame, i_01)
    x_10 = tf.gather_nd(frame, i_10)
    x_11 = tf.gather_nd(frame, i_11)

    w_00 = tf.cast((output_x_1 - output_x) * (output_y_1 - output_y), dtype = tf.float32)
    w_01 = tf.cast((output_x_1 - output_x) * (output_y - output_y_0), dtype = tf.float32)
    w_10 = tf.cast((output_x - output_x_0) * (output_y_1 - output_y), dtype = tf.float32)
    w_11 = tf.cast((output_x - output_x_0) * (output_y - output_y_0), dtype = tf.float32)
    output = tf.add_n([w_00 * x_00, w_01 * x_01, w_10 * x_10, w_11 * x_11])
    
    return output

def get_grid(channels, H, W, h, w):
    channels_1 = tf.range(channels)
    H_1 = tf.range(h + 1, h + H + 1)
    W_1 = tf.range(w + 1, w + W + 1)
    n, h, w = tf.meshgrid(channels_1, H_1, W_1, indexing = 'ij')
    n = tf.expand_dims(n, axis = 3)
    n = tf.cast(n, dtype = tf.float32)
    h = tf.expand_dims(h, axis = 3)
    h = tf.cast(h, dtype = tf.float32)
    w = tf.expand_dims(w, axis = 3)
    w = tf.cast(w, dtype = tf.float32)
    return n, h, w