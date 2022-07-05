import numpy as np
import tensorflow as tf
from model.Generator import Generator
from model.Discriminator import Discriminator
# from model.Identification_discriminator import Identification_discriminator
import utlis.loss_function as utlis_loss
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
    def distributed_train_step(batch):
        results = data_loader.strategy.run(train_step, args=(batch))
        results = reduce_dict(results)
        return results

    def train_step(batch):

        source_img, target_img, source_angle, target_angle = batch
        result = {}
        with tf.GradientTape() as tape:

            '''
            Cycle Consistency Loss
            '''
            fake_target_image = generator(
                [source_img, target_angle], training=True)
            reconstruct_source_image = generator(
                fake_target_image, source_angle, training=True)
            cycle_loss = utlis_loss.cycle_consistency_loss(
                source_img, reconstruct_source_image)

            '''
            Adversarial Loss
            '''
            fake_logit = discriminator(fake_target_image, training=True)
            real_logit = discriminator(target_img, training=True)
            adversarial_generator_loss = utlis_loss.generator_loss(fake_logit)
            adversarial_discriminator_loss = utlis_loss.discriminator_loss(
                real_logit)

            '''
            View Classification Loss
            '''
            predict_label_from_real_logit = View_discriminator(target_img)
            predict_label_from_fake_logit = View_discriminator(
                fake_target_image)

            real_view_classification_loss = utlis_loss.real_view_classification_loss(
                predict_label_from_real_logit)
            fake_view_classification_loss = utlis_loss.fake_view_classification_loss(
                predict_label_from_fake_logit
            )

            '''
            Total Loss
            '''
            generator_loss = cycle_loss + \
                fake_view_classification_loss + adversarial_generator_loss
            discriminator_loss = adversarial_discriminator_loss
            View_discriminator_loss = real_view_classification_loss + \
                fake_view_classification_loss

        result.update({'loss_G': generator_loss,
                       'loss_D': discriminator_loss,
                       'loss_Dview': View_discriminator_loss})

        generator_gradient = tape.gradient(
            generator_loss, generator.trainable_variables)

        discriminator_gradient = tape.gradient(
            discriminator_loss, discriminator.trainable_variables)

        View_discriminator_gradient = tape.gradient(
            View_discriminator_loss, View_discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(generator_gradient, generator.trainable_variables))

        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradient, discriminator.trainable_variables))

        View_discriminator_optimizer.apply_gradients(
            zip(View_discriminator_gradient, View_discriminator.trainable_variables))

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

    data_loader = create_training_data('OU_MVLP_multi_view')
    training_batch = data_loader.read()

    with data_loader.strategy.scope():

        generator = Generator(32).model((128, 88, 3), 14)
        discriminator = Discriminator(32).model((128, 88, 3))
        View_discriminator = Discriminator(32).model((128, 88, 3))
        # # Identification_discriminator = Identification_discriminator(
        # 32).model((128, 88, 3))

        generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(
            lr=1e-4, decay=1e-4)
        View_discriminator_optimizer = tf.keras.optimizers.Adam(
            lr=1e-4, decay=1e-4)
        # Identification_discriminator_optimizer = tf.keras.optimizers.Adam(
        # lr=1e-4, decay=1e-4)

        generator.compile(optimizer=generator_optimizer)
        discriminator.compile(optimizer=discriminator_optimizer)
        View_discriminator.compile(optimizer=View_discriminator_optimizer)
        # Identification_discriminator.compile(
        # optimizer=Identification_discriminator_optimizer)

    models = {
        'generator': generator,
        'discriminator': discriminator,
        'View_discriminator': View_discriminator,
        # # 'Identification_discriminator': Identification_discriminator
    }

    log_path = training_info['save_model']['logdir']
    save_model = Save_Model(models, info=training_info)
    summary_writer = tf.summary.create_file_writer(
        f'{log_path}/{save_model.startingDate}')
    iteration = 0
    while iteration < 2000:
        for step, batch in enumerate(training_batch):

            result = distributed_train_step(batch)
            output_message = ''

            with summary_writer.as_default():

                for loss_name, loss in result.items():

                    tf.summary.scalar(loss_name, loss,
                                      generator_optimizer.iterations)
                    output_message += f'{loss_name}: loss, '

        print(f'Epoch: {iteration:6}' + output_message)

        iteration += 1

        if iteration % 10 == 0:
            source_img, target_img, source_angle, target_angle = batch
            fake_target_img = generator([source_img, target_angle])

            rawImage = combineImages(target_img)
            fakeImage = combineImages(fake_target_img)

            with summary_writer.as_default():
                tf.summary.image('rawImage', [rawImage], step=iteration)
                tf.summary.image('fakeImage', [fakeImage], step=iteration)
        save_model.save()


if __name__ == '__main__':
    main()
