'''
Implement "Multi-Task GANs for View-Specific Feature Learning in Gait Recognition"

'''

from re import sub
import numpy as np
import tensorflow as tf
from model.GaitNet import GaitNet

from utlis.loss_function import GaitSimpleNet_Loss
from utlis.create_training_data import create_training_data
from utlis.save_model import Save_Model
from utlis.config.data_info import OU_MVLP_triplet_train, training_info


def main():

    def reduce_dict(d: dict):
        """ inplace reduction of items in dictionary d """
        return {
            k: data_loader.strategy.reduce(
                tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in d.items()
        }

    @tf.function
    def distributed_train_step(imgs, subject):
        results = data_loader.strategy.run(train_step, args=(imgs, subject))
        results = reduce_dict(results)
        return results

    def train_step(imgs, subject):

        imgs = tf.reshape(
            imgs, (imgs.shape[0]*imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]))
        subject = tf.reshape(subject, (subject.shape[0]*subject.shape[1], ))

        result = {}

        with tf.GradientTape() as tape:

            # calcluate GaitSimpleNet loss
            imgs_embedding = GaitSimpleNet(imgs, training=True)

            triplet_loss = GaitSimpleNet_Loss(imgs_embedding, subject)

        result.update({'loss_D/triplet_loss': triplet_loss})

        GaitSimpleNet_grad = tape.gradient(
            triplet_loss, GaitSimpleNet.trainable_variables)

        GaitSimpleNet_optimizer.apply_gradients(
            zip(GaitSimpleNet_grad, GaitSimpleNet.trainable_variables))

        return result

    def combineImages(images, col=4, row=4):
        images = (images+1)/2
        images = images.numpy()
        b, h, w, _ = images.shape
        imagesCombine = np.zeros(shape=(h*col, w*row, 3))
        for y in range(col):
            for x in range(row):
                imagesCombine[y*h:(y+1)*h, x*w:(x+1)*w] = images[x+y*row]
        return imagesCombine

    # def test_accuracy(embedding, label):

    data_loader = create_training_data('OU_MVLP_triplet')
    training_batch = data_loader.read()

    with data_loader.strategy.scope():

        # GaitSimpleNet = GaitNet(32).model((128, 88, 3))
        mode = 'gait_recognition'
        date = '2022_6_28_11_8'
        GaitSimpleNet = tf.keras.models.load_model(
            f'/home/aaron/Desktop/Aaron/College-level_Applied_Research/gait_log/{mode}/{date}/GaitNet/trained_ckpt')

        GaitSimpleNet_optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
        GaitSimpleNet.compile(optimizer=GaitSimpleNet_optimizer)

    log_path = training_info['save_model']['logdir']
    save_model = Save_Model(GaitSimpleNet, info=training_info)
    summary_writer = tf.summary.create_file_writer(
        f'{log_path}/{save_model.startingDate}')
    iteration = 0
    while iteration < 2000:
        for step, batch in enumerate(training_batch):

            imgs, subject = batch
            # imgs = tf.reshape(imgs, (OU_MVLP_triplet_train[]))
            result = distributed_train_step(imgs, subject)
            triplet_loss = result['loss_D/triplet_loss']
            # dis_real_loss, dis_fake_loss = train_GaitSimpleNet(
            #     batch_subjects, batch_angles, batch_images_ang1, batch_images_ang2)

            # gen_fake_loss, disparate = train_generator(
            #     batch_subjects, batch_angles, batch_images_ang1, batch_images_ang2)

            with summary_writer.as_default():
                #     tf.summary.scalar('disRealLoss', dis_real_loss, GaitSimpleNet_optimizer.iterations)
                #     tf.summary.scalar('disFakeLoss', dis_fake_loss, GaitSimpleNet_optimizer.iterations)

                tf.summary.scalar('triplet_loss', triplet_loss,
                                  GaitSimpleNet_optimizer.iterations)
            #     tf.summary.scalar('genLoss', gen_fake_loss, generator_optimizer.iterations)
            #     tf.summary.scalar('disparate', disparate, generator_optimizer.iterations)

        print(f'Epoch: {iteration:6} Triplet_loss: {triplet_loss:5}')
        # print(f'Epoch: {iteration:6} Batch: {step:3} Disparate:{disparate:4.5} G_loss: {gen_fake_loss:4.5} D_real_loss: {dis_real_loss:4.5} D_fake_loss: {dis_fake_loss:4.5}')
        iteration += 1

        # if generator_optimizer.iterations % 10 == 0:
        #     encode_angle1 = encoder(batch_images_ang1, training = False)
        #     view_transform = view_transform_layer([encode_angle1, batch_angles])
        #     predict_ang2 = generator(view_transform)
        #     rawImage = combineImages(batch_images_ang2)
        #     fakeImage = combineImages(predict_ang2)
        #     with summary_writer.as_default():
        #         tf.summary.image('rawImage', [rawImage], step=generator_optimizer.iterations)
        #         tf.summary.image('fakeImage', [fakeImage], step=generator_optimizer.iterations)
        save_model.save()


if __name__ == '__main__':
    main()
