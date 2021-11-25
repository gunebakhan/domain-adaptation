from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Model
from data_genetator import DataGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_confusion_matrix(model, gen):
    preds = []
    trains = []
    for i, (image, label) in enumerate(gen):
        pred = model.predict(image)
        pred = np.argmax(pred, axis=-1)
        # pred = np.where(pred > 0.5, 1, 0)
        pred = pred.flatten()
        preds.append(pred)
        label = np.argmax(label, axis=-1)
        label = label.flatten()
        trains.append(label)

    trains = np.concatenate(trains)
    preds = np.concatenate(preds) 

    lin_preds = preds.flatten()

    lin_trains = trains.flatten()
    cm = confusion_matrix(lin_trains, lin_preds, normalize='true')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=['others', 'Corn', 'Cotton', 'rice']).plot(cmap=plt.cm.Blues)
    plt.show()


def meanf1_iou(model, gen):
    preds = []
    trains = []
    for i, (image, label) in enumerate(gen):
        pred = model.predict(image)
        pred = np.argmax(pred, axis=-1)
        # pred = np.where(pred > 0.5, 1, 0)
        pred = pred.flatten()
        preds.append(pred)
        label = np.argmax(label, axis=-1)
        label = label.flatten()
        trains.append(label)

    trains = np.concatenate(trains)
    preds = np.concatenate(preds) 

    lin_preds = preds.flatten()

    lin_trains = trains.flatten()
    
    cm = tf.math.confusion_matrix(lin_trains, lin_preds)
#     cms = cm/cm.numpy().sum(axis=1)[:, tf.newaxis]
    
    # Get confusion matrix
#     cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
#                              tf.reshape(y_pred, [-1]))
    
    # Get precisions
    TP = tf.linalg.diag_part(cm)
    precisions = TP / tf.reduce_sum(cm, axis=0)
    
    # Get recalls
    TP = tf.linalg.diag_part(cm)
    recalls = TP / tf.reduce_sum(cm, axis=1)
    
    # Get F1s
    f1s = (2 * precisions * recalls) / (precisions + recalls)
    
    sns.heatmap(
        cm, annot=True,
        xticklabels=[0, 1],
        yticklabels=[0, 1])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    
    plt.show()
    
#     return np.mean(f1s[1:])
    m = tf.keras.metrics.MeanIoU(num_classes=4)
    m.update_state(lin_trains, lin_preds)
    print(m.result().numpy())
    print(np.mean(f1s))


def plot_feature_space(model, src_files, tar_files):
    # Testing
    mid_layer_index = len(model.layers) // 2
    mid_model = Model(inputs=[model.input], outputs=[model.layers[mid_layer_index].output])

    maximum, minimum = 14.733826, -49.208305

    train_gen = DataGenerator(image_paths=src_files, batch_size=len(tar_files),  augment=True, 
                          shuffle=True, normalize=False, gan=True, maximum=maximum, minimum=minimum)


    tar_gen = DataGenerator(image_paths=tar_files, batch_size=len(tar_files), augment=False, shuffle=False,
                         normalize=True, maximum=maximum, minimum=minimum)

    
    x_test, _ = train_gen[0]
    x_tar, _ = tar_gen[0]
    x = np.vstack((x_test, x_tar))

    features = mid_model.predict(x)


    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(features)

    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(pca_result)

    color_map = np.concatenate((np.ones(features.shape[0]//2), np.zeros(features.shape[0]//2)))

    sns.set(rc={'figure.figsize':(11.7,8.27)})
    palette = sns.color_palette("bright", 2)
    sns.scatterplot(tsne_results[:,0], tsne_results[:,1], hue=color_map, legend='full', palette=palette)