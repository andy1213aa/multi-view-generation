import tensorflow as tf

def generator_loss(fake_logit):
    loss = tf.math.reduce_mean(tf.math.log(1-fake_logit))
    return loss

def discriminator_loss(real_logit, fake_logit):
    loss = -tf.math.reduce_mean(tf.math.log(real_logit)) - tf.math.reduce_mean(tf.math.log(1-fake_logit))
    return loss

def real_view_classification_loss(real_view_logit):
    loss = -tf.math.reduce_mean(tf.math.log(real_view_logit))
    return loss

def fake_view_classification_loss(fake_view_logit):
    loss = tf.math.reduce_mean(-tf.math.log(fake_view_logit))
    return loss

def cycle_consistency_loss(X, fake):
    loss = tf.math.reduce_mean(tf.abs(X-fake))
    return loss

def identification_loss(real_logit, fake_logit):
    loss = tf.math.reduce_mean(tf.math.log(real_logit)) + tf.math.reduce_mean(tf.math.log(1-fake_logit))
    return loss
