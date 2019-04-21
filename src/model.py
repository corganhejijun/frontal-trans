# -*- coding: utf-8 -*- 
import tensorflow as tf

class ScaleGan(object):
    def __init__(self, sess, origin_size=64, img_size=256, phase='test'):
        self.batch_size = 1
        self.img_dim = 3 # image file color channel
        self.sess = sess
        self.img_size = img_size 
        self.build_model()

    def build_model(self)
        self.input_img = tf.placeholder(tf.float32, 
            [self.batch_size, self.img_size, self.img_size, self.img_dim*2],
            name='input_A_and_B_images')
        sizes = []
        size = origin_size
        while(size < img_size):
            size.append(size)
            size = size*2
        if size != img_size:
            print("size error when building model. origin size is " 
                    + str(self.origin_size) + " and target image size is " + str(self.img_size))
            return
        # A is sample, B is ground truth, 
        self.real_A = []
        A = self.input_img[:, :, :, :self.img_dim]
        for size in sizes:
            self.real_A.append(tf.image.resize_images(A, (size, size)))
        self.real_A.append(A)
        self.real_B = []
        B = self.input_img[:, :, :, self.img_dim:2*self.img_dim]
        for size in sizes:
            self.real_B.append(tf.image.resize_images(B, (size, size)))
        self.real_B.append(B)

        self.fake_B = self.generator(self.real_A)
        self.fake_sample = self.sampler(self.real_A)

        self.real_AB_d4 = tf.concat([self.real_A_d4, self.real_B_d4], 3)
        self.fake_AB_d4 = tf.concat([self.real_A_d4, self.fake_B_d4], 3)
        self.fake_AB_d2 = tf.concat([self.real_A_d2, self.fake_B_d2], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.BB = tf.concat([self.real_B, self.fake_B], 3)
        self.BB_d2 = tf.concat([self.real_B_d2, self.fake_B_d2], 3)
        self.BB_d4 = tf.concat([self.real_B_d4, self.fake_B_d4], 3)

        self.D_real_AB_d4, self.D_real_AB_d4_logits = self.discriminator(self.real_AB_d4, "discriminator_AB_d4", reuse=False)
        self.D_fake_AB_d4, self.D_fake_AB_d4_logits = self.discriminator(self.fake_AB_d4, "discriminator_AB_d4", reuse=True)
        self.D_fake_AB_d2, self.D_fake_AB_d2_logits = self.discriminator(self.fake_AB_d2, "discriminator_AB_d2")
        self.D_fake_AB, self.D_fake_AB_logits = self.discriminator(self.fake_AB, "discriminator_AB")

        self.D_BB, self.D_BB_logits = self.discriminator(self.BB, "discriminator_BB")
        self.D_BB_d2, self.D_BB_logits_d2 = self.discriminator(self.BB_d2, "discriminator_BB_d2")
        self.D_BB_d4, self.D_BB_logits_d4 = self.discriminator(self.BB_d4, "discriminator_BB_d4")

    def generator(self, img):

    def sampler(self, img):