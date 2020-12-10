import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
#from tensorflow.keras.optimizers import SGD, adadelta, adagrad, Adam, adamax, nadam
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
    return K.mean(K.abs(y_prediction, y_true), axis = [1, 2, 3])

def wrapping_loss(network_intermediate_values, y_prediction):
    loss_1 = l1_loss(network_intermediate_values[0], backward_warping(network_intermediate_values[1], network_intermediate_values[2]))
    loss_2 = l1_loss(network_intermediate_values[1], backward_warping(network_intermediate_values[0], network_intermediate_values[3]))
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
    return K.sqrt(K.square(y_true - y_predicted) + 0.01**2)

# Metrics
def psnr_metric(y_true, y_prediction):
    mean_squared_error = K.mean(K.square(y_true - y_prediction))
    return (10.0 * K.log(1.0 / (mean_squared_error + 1e-10))) / K.log(10.0)

names = ["1607310362", "1607310679", "1607311001", "1607311333", "1607311650", "1607311980", "1607312321", "1607312724",
         "1607313098", "1607313507", "1607310394", "1607310710", "1607311033", "1607311358", "1607311680", "1607312014",
         "1607312358", "1607312752", "1607313128", "1607313545", "1607310421", "1607310738", "1607311061", "1607311388",
         "1607311711", "1607312049", "1607312414", "1607312788", "1607313168", "1607313575", "1607310446", "1607310767",
         "1607311090", "1607311416", "1607311741", "1607312078", "1607312456", "1607312824", "1607313202", "1607310471",
         "1607310797", "1607311120", "1607311441", "1607311773", "1607312108", "1607312486", "1607312857", "1607313239",
         "1607310499", "1607310827", "1607311148", "1607311471", "1607311804", "1607312140", "1607312517", "1607312894",
         "1607313274", "1607310530", "1607310855", "1607311177", "1607311503", "1607311835", "1607312173", "1607312556",
         "1607312920", "1607313319", "1607310560", "1607310885", "1607311207", "1607311536", "1607311866", "1607312204",
         "1607312589", "1607312955", "1607313349", "1607310590", "1607310913", "1607311237", "1607311563", "1607311895",
         "1607312233", "1607312618", "1607312985", "1607313393", "1607310619", "1607310940", "1607311266", "1607311590",
         "1607311922", "1607312264", "1607312655", "1607313027"]

test_names = ["1607313429", "1607310647", "1607310969", "1607311300",
              "1607311622", "1607311952", "1607312294", "1607312693", "1607313060", "1607313473"]


def final_validation_batch_generator(batch_size):
    folders = ["FreeMountain", "SampleScene", "demoScene_free"]
    firstFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    middleFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    lastFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")

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

        X_batch = np.array([firstFrame, lastFrame], dtype='float16')
        y_batch = middleFrame

        return X_batch, y_batch

def batch_generator(batch_size, files, num_channels = 6, batch_image_size = 512):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    while(True):
        firstFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
        middleFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
        lastFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")

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

        timestamp = np.array([[[0.5]]])
        X_batch = np.array([firstFrame, lastFrame, timestamp], dtype = 'float16')
        #X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
        y_batch = middleFrame

        # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
        # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
        yield X_batch, y_batch

def get_batch(batch_size, files, num_channels = 6, batch_image_size = 512):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    firstFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    middleFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")
    lastFrame = np.zeros(shape=(batch_size, batch_image_size, batch_image_size, num_channels), dtype="float16")

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
    X_batch = np.array([firstFrame, lastFrame, timestamp], dtype = 'float16')
    #X_batch = np.concatenate((firstFrame, lastFrame), 3)  # Concatenate along channels dimension
    #y_batch = middleFrame

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
    optimizer = Adam(learning_rate = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
    loss = charbonnier_loss

    #a = batch_generator(32)
    #print(a.shape)
    #print(a)

    train_items = len(names) * 400
    test_items = len(test_names) * 400
    epochs = 2
    batch_size = 2
    train_steps = int(np.floor(train_items / batch_size))
    test_steps = int(np.floor(test_items / batch_size))

    train_generator = batch_generator(batch_size, names)
    test_generator = batch_generator(batch_size, test_names)

    model.compile(loss = loss_function(model_intermediate_values), optimizer = optimizer, metrics = [psnr_metric])
    model.fit(train_generator, steps_per_epoch = train_steps, epochs = epochs)
    model.evaluate(test_generator, steps = test_steps)
    
    #predict_batch = get_batch(32, test_names)
    #predictions = model.predict(predict_batch)
    #save_predictions(predictions)

if __name__ == '__main__':
    run()
