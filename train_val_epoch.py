import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf


def train_epoch(model, d_model, batch_size, src_gen, tar_gen):
    # count correct domain classification instances

    # keep values to calculate mean of them
    seg_loss_list = []
    seg_acc_list = []
    d_src_loss_list = []
    d_tar_loss_list = []
    d_src_acc_list = []
    d_tar_acc_list = []

    for i in range(len(src_gen)):
        # read data from data generator
        x_src, y_src = src_gen[i]
        y_src_domain = np.ones((batch_size, 1), dtype=np.float32)
        
        i = i % len(tar_gen)
        x_tar, y_tar = tar_gen[i]
        y_tar_domain = np.zeros((batch_size, 1), dtype=np.float32)

        # train segmentor
        seg_loss, seg_acc = model.train_on_batch(x_src, y_src)
        seg_loss_list.append(seg_loss)
        seg_acc_list.append(seg_acc)
        

        # train domain classifier
        d_src_loss, d_src_acc = d_model.train_on_batch(x_src, y_src_domain)
        d_src_loss_list.append(d_src_loss)
        d_src_acc_list.append(d_src_acc)

        d_tar_loss, d_tar_acc = d_model.train_on_batch(x_tar, y_tar_domain)
        d_tar_loss_list.append(d_tar_loss)
        d_tar_acc_list.append(d_tar_acc)
        
#         print(d_src_loss, d_tar_loss)


    return (sum(seg_loss_list)/len(seg_loss_list), 
            sum(d_src_loss_list)/len(d_src_loss_list), 
            sum(d_tar_loss_list)/len(d_tar_loss_list), 
            sum(seg_acc_list)/len(seg_acc_list), 
            sum(d_src_acc_list)/len(d_src_acc_list), 
            sum(d_tar_acc_list)/len(d_tar_acc_list)
           )


def validation_epoch(model, d_model, batch_size, src_gen, tar_gen):
    # keep values in list to calculate their means
    seg_loss_list = []
    seg_acc_list = []
    seg_loss_list_tar = []
    seg_acc_list_tar = []
    d_src_loss_list = []
    d_tar_loss_list = []
    d_src_acc_list = []
    d_tar_acc_list = []

    # perform validation on src val files
    for i in range(len(src_gen)):
        # read src data
        x_src, y_src = src_gen[i]
        y_src_domain = np.ones((batch_size, 1), dtype=np.float32)

        # evaluate src files on segmentor model
        seg_loss, seg_acc = model.evaluate(x_src, y_src, verbose=0)
        seg_loss_list.append(seg_loss)
        seg_acc_list.append(seg_acc)

        # evaluate src files on domain classification model
        d_src_loss, d_src_acc = d_model.evaluate(x_src, y_src_domain, verbose=0)
        d_src_loss_list.append(d_src_loss)
        d_src_acc_list.append(d_src_acc)

    # perform validation on tar val files
    for i in range(len(tar_gen)):
        # read tar data
        x_tar, y_tar = tar_gen[i]
        y_tar_domain = np.zeros((batch_size, 1), dtype=np.float32)


        # evaluate tar files on segmentor model
        seg_loss, seg_acc = model.evaluate(x_tar, y_tar, verbose=0)
        seg_loss_list_tar.append(seg_loss)
        seg_acc_list_tar.append(seg_acc)

        # evaluate tar files on domain classification model
        d_tar_loss, d_tar_acc = d_model.evaluate(x_tar, y_tar_domain, verbose=0)
        d_tar_loss_list.append(d_tar_loss)
        d_tar_acc_list.append(d_tar_acc)

    return (sum(seg_loss_list)/len(seg_loss_list), 
            sum(seg_loss_list_tar)/len(seg_loss_list_tar),
            sum(d_src_loss_list)/len(d_src_loss_list), 
            sum(d_tar_loss_list)/len(d_tar_loss_list), 
            sum(seg_acc_list)/len(seg_acc_list), 
            sum(seg_acc_list_tar)/len(seg_acc_list_tar),
            sum(d_src_acc_list)/len(d_src_acc_list), 
            sum(d_tar_acc_list)/len(d_tar_acc_list))


def ignore_case_train_epoch(model, 
                            d_model, 
                            batch_size, 
                            src_gen, 
                            tar_gen,
                            weighted_crossentropy,
                            binary_crossentropy,
                            optimizer1,
                            optimizer2,
                            accuracy,
                            binary_accuracy
                            ):
    # count correct domain classification instances
    num_corrects_src = 0
    total_src = 0

    num_corrects_tar = 0
    total_tar = 0

    # keep values to calculate mean of them
    segmentation_loss = keras.metrics.Mean()
    segmentation_accuracy = keras.metrics.Mean()
    domain_loss = keras.metrics.Mean()
    domain_accuracy = keras.metrics.Mean()

    for i in range(len(src_gen)):
        # read data from data generator
        x_src, y_src = src_gen[i]
        y_src_domain = K.ones((batch_size, 1), dtype=tf.float32)
        
        j = i % len(tar_gen)
        x_tar, y_tar = tar_gen[j]
        y_tar_domain = K.zeros((batch_size, 1), dtype=tf.float32)

        # train segmentor
        with tf.GradientTape() as tape:
            logits = model(x_src, training=True)
            seg_loss = weighted_crossentropy(y_src, logits)

        grads = tape.gradient(seg_loss, model.trainable_variables)
        optimizer1.apply_gradients(zip(grads, model.trainable_variables))

        seg_acc = accuracy(y_src, logits)

        segmentation_loss.update_state(seg_loss)
        segmentation_accuracy.update_state(seg_acc)
        
        x_domain = K.concatenate((x_src, x_tar), axis=0)
        y_domain = K.concatenate((y_src_domain, y_tar_domain), axis=0)

        # train domain classifier
        with tf.GradientTape() as tape:
            logits = d_model(x_domain, training=True)
            d_loss = binary_crossentropy(y_domain, logits)
        
        grads = tape.gradient(d_loss, d_model.trainable_variables)
        optimizer2.apply_gradients(zip(grads, d_model.trainable_variables))

        d_acc = binary_accuracy(y_domain, logits)

        domain_loss.update_state(d_loss)
        domain_accuracy.update_state(d_acc)


    return (
                segmentation_loss.result(),
                segmentation_accuracy.result(),
                domain_loss.result(),
                domain_accuracy.result()
           )


def ignore_case_validation_epoch(model, 
                                d_model, 
                                batch_size, 
                                src_gen, 
                                tar_gen,
                                weighted_crossentropy,
                                binary_crossentropy,
                                accuracy,
                                binary_accuracy
                                ):
    # keep values in list to calculate their means
    segmentatin_loss = keras.metrics.Mean()
    segmentation_accuracy = keras.metrics.Mean()
    domain_loss = keras.metrics.Mean()
    domain_accuracy = keras.metrics.Mean()

    # perform validation on src val files
    for i in range(len(src_gen)):
        # read src data
        x_src, y_src = src_gen[i]
        y_src_domain = K.ones((batch_size, 1), dtype=tf.float32)

        # read tar data
        x_tar, y_tar = tar_gen[i]
        y_tar_domain = K.zeros((batch_size, 1), dtype=tf.float32)

        x_domain = K.concatenate((x_src, x_tar), axis=0)
        y_domain = K.concatenate((y_src_domain, y_tar_domain), axis=0)

        # evaluate src files on segmentor model
        logits = model(x_src, training=False)
        seg_loss = weighted_crossentropy(y_src, logits)
        seg_acc = accuracy(y_src, logits)

        segmentatin_loss.update_state(seg_loss)
        segmentation_accuracy.update_state(seg_acc)

        # evaluate src files on domain classification model
        logits = d_model(x_domain, training=False)
        d_loss = binary_crossentropy(logits, y_domain)
        d_acc = binary_accuracy(logits, y_domain)

        domain_loss.update_state(d_loss)
        domain_accuracy.update_state(d_acc)


    return (
        segmentatin_loss.result(),
        segmentation_accuracy.result(),
        domain_loss.result(),
        domain_accuracy.result(),
    )