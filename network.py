from GroupNorm3D import GroupNormalization
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

class context_encoding_module:
    def __init__(self, inChannels, numofclasses, **kwargs):
        super(context_encoding_module, self).__init__(**kwargs)
        self.inChannels = inChannels
        self.numofclasses = numofclasses
        
        self.context_encoder_conv1 = Conv3D(self.inChannels, kernel_size=1, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.context_encoder_conv2 = Conv3D(self.numofclasses, kernel_size=1, strides=1, padding='valid', data_format='channels_first', kernel_initializer='he_normal')
    
    def __call__(self, x):
        x = self.context_encoder_conv1(x)
        x = tf.nn.relu(x)
        x = self.context_encoder_conv2(x)
        
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])
        
        x = tf.reshape(x, [x.shape[0], x.shape[1]*x.shape[2]*x.shape[3], x.shape[4]])
        x = tf.math.sigmoid(x)
        
        return x

class VnetVariant(Model):

    def __init__(self, num_input_channel, patch_size, numofclasses, **kwargs):
        super(VnetVariant, self).__init__(**kwargs)

        self.num_input_channel = num_input_channel
        self.patch_size = patch_size
        self.numofclasses = numofclasses
        
        self.conv1 = Conv3D(16, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv2_1 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv2_2 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv3_1 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv3_2 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv4_1 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv4_2 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv5_1 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv5_2 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv6_1 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv6_2 = Conv3D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv7_1 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv7_2 = Conv3D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv8_1 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.conv8_2 = Conv3D(64, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.conv9 = Conv3D(32, kernel_size=3, strides=1, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.gn1 = GroupNormalization(16, axis=1)
        
        self.gn2_1 = GroupNormalization(32, axis=1)
        self.gn2_2 = GroupNormalization(32, axis=1)
        
        self.gn3_1 = GroupNormalization(64, axis=1)
        self.gn3_2 = GroupNormalization(64, axis=1)
        
        self.gn4_1 = GroupNormalization(128, axis=1)
        self.gn4_2 = GroupNormalization(128, axis=1)
        
        self.gn5_1 = GroupNormalization(256, axis=1)
        self.gn5_2 = GroupNormalization(256, axis=1)
        
        self.gn6_1 = GroupNormalization(256, axis=1)
        self.gn6_2 = GroupNormalization(256, axis=1)
        
        self.gn7_1 = GroupNormalization(128, axis=1)
        self.gn7_2 = GroupNormalization(128, axis=1)
        
        self.gn8_1 = GroupNormalization(64, axis=1)
        self.gn8_2 = GroupNormalization(64, axis=1)
        
        self.gn9 = GroupNormalization(32, axis=1)
        
        self.pool1 = Conv3D(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.pool2 = Conv3D(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.pool3 = Conv3D(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.pool4 = Conv3D(256, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.upsample1 = Conv3DTranspose(128, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.upsample2 = Conv3DTranspose(64, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.upsample3 = Conv3DTranspose(32, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.upsample4 = Conv3DTranspose(16, 2, strides=2, padding='same', data_format='channels_first', kernel_initializer='he_normal')
        
        self.classifier_conv = Conv3D(self.numofclasses, kernel_size=(1, 1, 1), padding='same', data_format='channels_first', kernel_initializer='he_normal')
        self.classifier_softmax = Lambda(lambda x: K.softmax(x, axis=1))
        
        self.context_encoder = context_encoding_module(128, self.numofclasses)
        
    
    def call(self, inputs, training=None):

        conv1 = self.conv1(inputs)
        conv1 = self.gn1(conv1)
        conv1 = tf.nn.relu(conv1)
        
        down1 = self.pool1(conv1)
        
        conv2 = self.conv2_1(down1)
        conv2 = self.gn2_1(conv2)
        conv2 = tf.nn.relu(conv2)
        
        conv2 = self.conv2_2(conv2)
        conv2 = self.gn2_2(conv2)
        conv2 = tf.nn.relu(conv2)
    
        down2 = self.pool2(conv2)
        
        conv3 = self.conv3_1(down2)
        conv3 = self.gn3_1(conv3)
        conv3 = tf.nn.relu(conv3)
        
        conv3 = self.conv3_2(conv3)
        conv3 = self.gn3_2(conv3)
        conv3 = tf.nn.relu(conv3)
    
        down3 = self.pool3(conv3)
        
        conv4 = self.conv4_1(down3)
        conv4 = self.gn4_1(conv4)
        conv4 = tf.nn.relu(conv4)
        
        conv4 = self.conv4_2(conv4)
        conv4 = self.gn4_2(conv4)
        conv4 = tf.nn.relu(conv4)
    
        down4 = self.pool4(conv4)
    
        conv5 = self.conv5_1(down4)
        conv5 = self.gn5_1(conv5)
        local_context = conv5 = tf.nn.relu(conv5)
    
        conv5 = self.conv5_2(conv5)
        conv5 = self.gn5_2(conv5)
        conv5 = tf.nn.relu(conv5)
        
        up1 = self.upsample1(conv5)
        concat1 = tf.concat([up1, conv4], axis=1)
        
        conv6 = self.conv6_1(concat1)
        conv6 = self.gn6_1(conv6)
        conv6 = tf.nn.relu(conv6)
        
        conv6 = self.conv6_2(conv6)
        conv6 = self.gn6_2(conv6)
        conv6 = tf.nn.relu(conv6)
        
        up2 = self.upsample2(conv6)
        concat2 = tf.concat([up2, conv3], axis=1)
        
        conv7 = self.conv7_1(concat2)
        conv7 = self.gn7_1(conv7)
        conv7 = tf.nn.relu(conv7)
        
        conv7 = self.conv7_2(conv7)
        conv7 = self.gn7_2(conv7)
        conv7 = tf.nn.relu(conv7)
        
        up3 = self.upsample3(conv7)
        concat3 = tf.concat([up3, conv2], axis=1)
        
        conv8 = self.conv8_1(concat3)
        conv8 = self.gn8_1(conv8)
        conv8 = tf.nn.relu(conv8)
        
        conv8 = self.conv8_2(conv8)
        conv8 = self.gn8_2(conv8)
        conv8 = tf.nn.relu(conv8)
        
        up4 = self.upsample4(conv8)
        concat4 = tf.concat([up4, conv1], axis=1)
    
        conv9 = self.conv9(concat4)
        conv9 = self.gn9(conv9)
        conv9 = tf.nn.relu(conv9)
        
        # output 1 #
        logits = self.classifier_conv(conv9)
        output1 = self.classifier_softmax(logits)
    
        # output 2 #
        output2 = self.context_encoder(local_context)
        
        return output1, output2