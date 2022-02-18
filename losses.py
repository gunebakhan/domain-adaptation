import tensorflow.keras.backend as K
import tensorflow as tf


def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index   
        classSelectors = [K.equal(i, classSelectors) for i in np.arange(len(weightsList), dtype='int64')]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc


def IgnoreCaseWeightedLoss(originalLossFunc, weightsList, ignore_label=4):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 
            #if your loss is sparse, use only true as classSelectors

        # Find indices which don't contain class 4 labels
        indices = tf.where(K.not_equal(classSelectors, ignore_label))
        classSelectors = tf.gather_nd(classSelectors, indices)
        # true = tf.gather_nd(true, indices)
        pred = tf.gather_nd(pred, indices)

        # Create weight tensor corresponds to each label
        weightMultiplier = deep_copy(classSelectors)
        weightMultiplier = tf.cast(weightMultiplier, dtype=tf.float32)
        for i, weight in enumerate(weightsList):
            weightMultiplier = tf.where(K.equal(weightMultiplier, i), weight, weightMultiplier)

        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        # true = keras.utils.to_categorical(K.argmax(true, axis=-1), num_classes=4)
        true = keras.utils.to_categorical(classSelectors, num_classes=4)
        loss = originalLossFunc(true, pred) 
        loss = loss * weightMultiplier
        # loss = K.mean(K.sum(loss, axis=(1, 2)))
        # return K.mean(loss)
        return loss
    return lossFunc


def distribution_loss():
    def loss(distb1, distb2):
        distb1_mean = K.mean(distb1, axis=0)
        distb2_mean = K.mean(distb2, axis=0)
        diff = distb1_mean - distb2_mean
        squared_diff = K.square(diff)
        sqrt_squared_diff = K.sqrt(squared_diff)
        mean_squared_diff = K.mean(squared_diff)

        return mean_squared_diff
    return loss


def reconstruction_loss():
    def loss(x, xhat):
        diff = x - xhat
        squared_diff = K.square(diff)
        mean_squared_diff = K.mean(squared_diff)
        return mean_squared_diff
    return loss