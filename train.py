import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
# from tensorflow.keras.optimizers import SGD, adadelta, adagrad, Adam, adamax, nadam
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from model import Network, backward_warping

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
    return K.mean(K.abs(y_prediction, y_true), axis=[1, 2, 3])


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


def loss_function(network_intermediate_values):
    def loss(y_true, y_prediction):
        l1_loss_val = l1_loss(y_true, y_prediction)
        wrapping_loss_val = wrapping_loss(network_intermediate_values, y_prediction)
        smoothness_loss_val = smoothness_loss(network_intermediate_values)
        return smoothness_loss_val + 102 * wrapping_loss_val + 204 * l1_loss_val

    return loss


def charbonnier_loss(y_true, y_predicted):
    return K.sqrt(K.square(y_true - y_predicted) + 0.01 ** 2)


# Metrics
def psnr_metric(y_true, y_prediction):
    mean_squared_error = K.mean(K.square(y_true - y_prediction))
    return (10.0 * K.log(1.0 / (mean_squared_error + 1e-10))) / K.log(10.0)


names = ["1607557976", "1607558097", "1607561410", "1607561536", "1607561833", "1607561957", "1607562091", "1607557387",
         "1607557509", "1607557632", "1607557754", "1607557887", "1607558006", "1607558131", "1607561440", "1607561570",
         "1607561863", "1607561991", "1607562128", "1607557416", "1607557536", "1607557662", "1607557790", "1607557914",
         "1607558039", "1607561345", "1607561474", "1607561773", "1607561892", "1607562030", "1607557446", "1607557566",
         "1607557693", "1607557819", "1607557943", "1607558067", "1607561377", "1607561504", "1607561805", "1607561925",
         "1607562061"]
test_names = ["1607557355", "1607557476", "1607557600", "1607557726", "1607557853"]


# Gets a batch of images from one of the handmade videos
def final_validation_batch_generator(batch_size):
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

        X_batch = np.array([firstFrame, lastFrame, timestamp], dtype = 'float16')
        #X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
        y_batch = middleFrame

        # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
        # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
        yield X_batch, y_batch


def get_batch(batch_size, files, num_channels=6, batch_image_size=512):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    firstFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    middleFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    lastFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")

    random.seed()

    # Load images from folder
    for i in range(batch_size):
        randNum = random.randrange(3, 398)  # Toss out first 2 frames, as they're invalid
        randDir = random.choice(files)
        colorFirst = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum - 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecFirst = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum - 1).zfill(4) + ".exr",
                                 cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        firstFrame[i] = np.concatenate((colorFirst, motVecFirst), 2)

        '''colorMiddle = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecMiddle = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        middleFrame[i] = np.concatenate((colorMiddle, motVecMiddle), 2)'''

        colorLast = cv2.imread(dir + randDir + "/" + randDir + "final" + str(randNum + 1).zfill(4) + ".exr",
                               cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        motVecLast = cv2.imread(dir + randDir + "/" + randDir + "motVec" + str(randNum + 1).zfill(4) + ".exr",
                                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        lastFrame[i] = np.concatenate((colorLast, motVecLast), 2)

    timestamp = np.array([[[0.5]]])
    X_batch = np.array([firstFrame, lastFrame, timestamp], dtype='float16')
    # X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
    # y_batch = middleFrame

    # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
    # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
    return X_batch


def save_predictions(predictions):
    for prediction in predictions:
        image = prediction[:, :, :3]


def run():
    frame_interpolation = Network()
    model = frame_interpolation.get_model()
    model_intermediate_values = frame_interpolation.get_intermediate_values()
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = charbonnier_loss

    # a = batch_generator(32)
    # print(a.shape)
    # print(a)

    train_items = len(names) * 398  # 398 images per set, as 0 and 1 aren't proper data
    test_items = len(test_names) * 398
    epochs = 2
    batch_size = 2
    train_steps = int(np.floor(train_items / batch_size))
    test_steps = int(np.floor(test_items / batch_size))

    train_generator = batch_generator(batch_size, names)
    test_generator = batch_generator(batch_size, test_names)

    model.compile(loss=loss_function(model_intermediate_values), optimizer=optimizer, metrics=[psnr_metric])
    model.fit(train_generator, steps_per_epoch=train_steps, epochs=epochs)
    model.evaluate(test_generator, steps=test_steps)

    # predict_batch = get_batch(32, test_names)
    # predictions = model.predict(predict_batch)
    # save_predictions(predictions)


if __name__ == '__main__':
    run()
