import tensorflow as tf
from tensorflow_triplet_loss.model import triplet_loss


def GaitSimpleNet_Loss(imgs_embedding,subject):
    loss = triplet_loss.batch_hard_triplet_loss(subject, imgs_embedding, margin=0.8)
    return loss