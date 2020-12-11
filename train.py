import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
import os
# from tensorflow.keras.optimizers import SGD, adadelta, adagrad, Adam, adamax, nadam
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from model import Network#, backward_warping

'''class DataGenerator(Sequence):
    def __init__(self, files, file_path, data_type, batch_size, shuffle):
        self.batch_size = batch_size
        self.files = files
        self.file_path = file_path
        self.data_type = data_type
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        indexes = self.indexes[start:end]
        files_temp = [self.files[i] for i in indexes]

        X, Y = self.__data_generation(files_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if(self.shuffle):
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):
        for f in files:
            pass '''


# Loss Functions
def l1_loss(y_true, y_prediction):
    return K.mean(K.abs(y_prediction - y_true), axis=[1, 2, 3])

def wrapping_loss(network_intermediate_values, y_prediction):
    loss_1 = l1_loss(network_intermediate_values[0],
                     backward_warping(network_intermediate_values[1], network_intermediate_values[2]))
    loss_2 = l1_loss(network_intermediate_values[1],
                     backward_warping(network_intermediate_values[0], network_intermediate_values[3]))
    loss_3 = l1_loss(y_prediction, backward_warping(network_intermediate_values[0], network_intermediate_values[4]))
    loss_4 = l1_loss(y_prediction, backward_warping(network_intermediate_values[1], network_intermediate_values[5]))
    return loss_1 + loss_2 + loss_3 + loss_4

def smoothness_loss(network_intermediate_values):
    loss_1 = K.mean(K.abs(network_intermediate_values[2][:, 1:, :, :] - network_intermediate_values[2][:, :-1, :, :])) + \
             K.mean(K.abs(network_intermediate_values[2][:, :, 1:, :] - network_intermediate_values[2][:, :, :-1, :]))
    loss_2 = K.mean(K.abs(network_intermediate_values[3][:, 1:, :, :] - network_intermediate_values[3][:, :-1, :, :])) + \
             K.mean(K.abs(network_intermediate_values[3][:, :, 1:, :] - network_intermediate_values[3][:, :, :-1, :]))
    return loss_1 + loss_2
'''
def loss_function(network_intermediate_values):
    def loss(y_true, y_prediction):
        l1_loss_val = l1_loss(y_true, y_prediction)
        wrapping_loss_val = wrapping_loss(network_intermediate_values, y_prediction)
        smoothness_loss_val = smoothness_loss(network_intermediate_values)
        return smoothness_loss_val + 102 * wrapping_loss_val + 204 * l1_loss_val
    return loss
'''
def loss_function(y_true, y_prediction):
    #return l1_loss(y_true[:,:,:,0:3], y_prediction)
    loss = tf.keras.losses.MSE
    return loss(y_true[:,:,:,0:3], y_prediction)

def charbonnier_loss(y_true, y_predicted):
    return K.sqrt(K.square(y_true - y_predicted) + 0.01 ** 2)

# Metrics
def psnr_metric(y_true, y_prediction):
    y_true = y_true[:,:,:,0:3]
    mean_squared_error = K.mean(K.square(y_true - y_prediction))
    return (10.0 * K.log(1.0 / (mean_squared_error + 1e-10))) / K.log(10.0)

names = ["1607557976", "1607558097", "1607561410", "1607561536", "1607561833", "1607561957", "1607562091", "1607557387",
         "1607557509", "1607557632", "1607557754", "1607557887", "1607558006", "1607558131", "1607561440", "1607561570",
         "1607561863", "1607561991", "1607562128", "1607557416", "1607557536", "1607557662", "1607557790", "1607557914",
         "1607558039", "1607561345", "1607561474", "1607561773", "1607561892", "1607562030", "1607557446", "1607557566",
         "1607557693", "1607557819", "1607557943", "1607558067", "1607561377", "1607561504", "1607561805", "1607561925",
         "1607562061", "1607557355", "1607557476"]
test_names = ["1607557600", "1607557726", "1607557853"]

# Gets a batch of images from one of the handmade videos
def final_validation_batch_generator(batch_size):
    dir = "/blue/cis6930/andrew.watson/"
    folders = ["FreeMountain", "SampleScene", "demoScene_free"]
    firstFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    middleFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    lastFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    timestamp = np.zeros(shape=(batch_size, 1, 1, 1), dtype="float16")

    random.seed()

    for i in range(batch_size):
        randDir = random.choice(folders)
        randNum = 0
        if randDir == "FreeMountain":
            randNum = random.randrange(3, 497)
        elif randDir == "SampleScene":
            randNum = random.randrange(3, 198)
        elif randDir == "demoScene_free":
            randNum = random.randrange(3, 498)

        colorFirst = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecFirst = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr",
                                 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        firstFrame[i] = np.concatenate((colorFirst, motVecFirst), 2)

        colorMiddle = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr",
                                 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecMiddle = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr",
                                  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        middleFrame[i] = np.concatenate((colorMiddle, motVecMiddle), 2)

        colorLast = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr",
                               cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecLast = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lastFrame[i] = np.concatenate((colorLast, motVecLast), 2)

        timestamp[i] = np.array([[[0.5]]])

        X_batch = np.array([firstFrame, lastFrame], dtype='float16')
        y_batch = middleFrame

        return X_batch, y_batch

def get_predicted_data():
    dir = "/blue/cis6930/andrew.watson/FreeMountain/"
    num = 23
    firstFrame = np.zeros(shape=(16, 512, 512, 6), dtype="float16")
    lastFrame = np.zeros(shape=(16, 512, 512, 6), dtype="float16")
    timestamp = np.zeros(shape=(16, 1, 1, 1), dtype="float16")
    for i in range(16):

        colorFirst = cv2.imread(dir + "final" + str(num + (2*i)).zfill(4) + ".exr")
        motVecFirst = cv2.imread(dir + "motVec" + str(num + (2*i)).zfill(4) + ".exr")

        colorLast = cv2.imread(dir + "final" + str(num + (2*i) +2).zfill(4) + ".exr")
        motVecLast = cv2.imread(dir + "motVec" + str(num + (2*i) +2).zfill(4) + ".exr")


        firstFrame[i] = np.concatenate([colorFirst, motVecFirst], 2)
        lastFrame[i] = np.concatenate([colorLast, motVecLast], 2)
        timestamp[i] = np.array([[[0.5]]])

    X = [timestamp, firstFrame, lastFrame]
    return X



# Gets a batch of images from the autogeneratedimages
def batch_generator(batch_size, files, num_channels=6, batch_image_size=512):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    while True:
        firstFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
        middleFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
        lastFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
        timestamp = np.zeros(shape=(batch_size, 1, 1, 1), dtype="float16")

        random.seed()

        # Load images from folder
        for i in range(batch_size):
            randNum = random.randrange(3, 398)
            randDir = random.choice(files)

            while True:
                files_to_check = []
                files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr")
                files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr")
                files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr")

                files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr")
                files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr")
                files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr")

                exist = True
                for item in files_to_check:
                    if cv2.imread(item, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) is None:  # If happened to choose a rare invalid file (numbers can jump sometimes), choose another
                        #print("Tried to use invalid file", item)
                        randNum = random.randrange(3, 398)
                        randDir = random.choice(files)
                        exist = False
                if exist == True:
                    break
                

            colorFirst = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr",
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            motVecFirst = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr",
                                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            firstFrame[i] = np.concatenate((colorFirst, motVecFirst), 2)



            colorMiddle = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr",
                                     cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            motVecMiddle = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr",
                                      cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if motVecMiddle is None:
                print("SHOULDN'T GET HERE!")
                print(randDir)
                print(randNum)
                print(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr")

            middleFrame[i] = np.concatenate((colorMiddle, motVecMiddle), 2)

            

            colorLast = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr",
                                   cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            motVecLast = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr",
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            lastFrame[i] = np.concatenate((colorLast, motVecLast), 2)

            timestamp[i] = np.array([[[0.5]]])

        X_batch = [timestamp, firstFrame, lastFrame]
        #X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
        y_batch = middleFrame

        # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
        # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
        yield X_batch, y_batch

def get_batch(batch_size, files, num_channels=6, batch_image_size=512):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    firstFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    
    lastFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    timestamp = np.zeros(shape=(batch_size, 1, 1, 1), dtype="float16")
    random.seed()

    # Load images from folder
    for i in range(batch_size):
        randNum = random.randrange(3, 398)
        randDir = random.choice(files)

        while True:
            files_to_check = []
            files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr")
            files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr")
            files_to_check.append(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr")
            files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr")
            files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr")
            files_to_check.append(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr")
            for item in files_to_check:
                if cv2.imread(item, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) is None:  # If happened to choose a rare invalid file (numbers can jump sometimes), choose another
                    #print("Tried to use invalid file", item)
                    randNum = random.randrange(3, 398)
                    randDir = random.choice(files)
                    continue
            break

        colorFirst = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecFirst = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr",
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        firstFrame[i] = np.concatenate((colorFirst, motVecFirst), 2)

        '''colorMiddle = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr",
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecMiddle = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr",
                                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            
        middleFrame[i] = np.concatenate((colorMiddle, motVecMiddle), 2)
        '''
        

        colorLast = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecLast = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lastFrame[i] = np.concatenate((colorLast, motVecLast), 2)

        timestamp[i] = np.array([[[0.5]]])

    X_batch = [timestamp, firstFrame, lastFrame]
    # X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
    # y_batch = middleFrame

    # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
    # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
    return X_batch

class predict_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        predict_batch = get_batch(16, test_names)
        predictions = self.model.predict(predict_batch)
        print("Saving predictions at epoch", epoch)
        save_predictions(predictions, epoch)


def save_predictions(predictions, prefix):
    for i, prediction in enumerate(predictions):
        image = prediction[:, :, :3]
        image = (np.abs(image)**(1.0/2.2)) * 255  # linear to gamma
        cv2.imwrite("/blue/cis6930/andrew.watson/out/" + str(prefix) + "_" + str(i) + ".png", image)

def run():

    predicting = True

    resume = False

    tf.compat.v1.disable_eager_execution()  # Prevents some weird bugs
    mirrored_strategy = tf.distribute.MirroredStrategy()
    frame_interpolation = Network()
    with mirrored_strategy.scope():
        model = frame_interpolation.get_model()
        optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        metric = psnr_metric
    #model_intermediate_values = frame_interpolation.get_intermediate_values()

    # a = batch_generator(32)
    # print(a.shape)
    # print(a)

    reduced_names = names

    epochs = 30
    batch_size = 4
    train_steps = 20  # Batches per "epoch"
    test_steps = 20

    train_generator = batch_generator(batch_size, reduced_names)
    test_generator = batch_generator(batch_size, test_names)

    checkpoint_path = "/blue/cis6930/andrew.watson/saved/cp_epoch-{epoch:04d}_loss-{loss:.3f}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Save after every "epoch", and save whole model (so we can pick up where we left off)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)  

    #model.compile(loss=loss_function(model_intermediate_values), optimizer=optimizer, metrics=[metric])
    model.compile(loss=loss_function, optimizer=optimizer, metrics=[metric])

    if resume:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("Latest checkpoint:", latest)
        print("Loading weights from file...")
        model.load_weights(latest)

    if predicting:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print("Latest checkpoint:", latest)
        print("Loading weights from file...")
        model.load_weights(latest)
        print("Predicting...")
        #predict_batch = get_batch(16, test_names)
        predict_batch = get_predicted_data()
        predictions = model.predict(predict_batch)
        save_predictions(predictions, "initial")
        print("Predictions saved.")
    

    print("Training...")
    model.fit(train_generator, steps_per_epoch=train_steps, epochs=epochs, callbacks=[cp_callback, predict_callback()])  
    print("Evaluating model...")
    model.evaluate(test_generator, steps=test_steps)

    

if __name__ == '__main__':
    run()
