import cv2
import numpy as np
import random

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
         "1607311922", "1607312264", "1607312655", "1607313027", "1607313429", "1607310647", "1607310969", "1607311300",
         "1607311622", "1607311952", "1607312294", "1607312693", "1607313060", "1607313473", ]


def batch_generator(batch_size, num__channels, batch_image_size):
    dir = "/blue/cis6930/andrew.watson/AutoScene1/"
    firstFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    middleFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")
    lastFrame = np.zeros(shape=(batch_size, 512, 512, 6), dtype="float16")

    random.seed()

    # Load images from folder
    for i in range(batch_size):
        randNum = random.randrange(3, 398)
        randDir = random.choice(names)
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

        X_batch = np.concatenate((firstFrame, lastFrame), 2)
        y_batch = middleFrame

        # X_batch should be N x 512 x 512 x 12 (N batches, 2 6-layer images)
        # y_batch should be N x 512 x 512 x 6  (N batches, 1 6-layer image)
        yield X_batch, y_batch


def run():
    batch_generator(16, None, None)


if __name__ == '__main__':
    run()
