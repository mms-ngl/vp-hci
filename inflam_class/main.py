import os
import sys

from sklearn.metrics import confusion_matrix, accuracy_score
import keras
from keras.applications.vgg19 import VGG19
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse
import json

import pandas as pd
import numpy as np

import shutil
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt


def plot_conf_matrix(cm, class_names):
    cm = np.array(cm)

    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=True,
                                    show_normed=True,
                                    class_names=class_names)
    plt.show()

def copy_to_classified_folders(filename_test, label_dict, test_output, test_name, predictions):

    file_list=pd.read_csv(filename_test, skipinitialspace = True, usecols=["filename"])
    file_list = file_list.values.flatten().tolist()

    if not len(file_list) == len(predictions):
        raise Exception("Mismatch between prediction list and file list")
    for label in label_dict.values():
        os.makedirs(os.path.join(test_output, test_name, label), exist_ok=True)

    for i, file in enumerate(file_list):
        filename = file.split("/")[-1]

        new_path = os.path.join(test_output, test_name, str(predictions[i]), filename)
        shutil.copy(file, new_path)

def test(config, copy=True):

    with open(config, 'r') as f:
        cfg = json.load(f)

    shape = (int(cfg['height']), int(cfg['width']), 3)
    n_class = int(cfg['class_number'])
    filenames_test = cfg["filenames_test"]

    save_dir = cfg["save_dir"]

    if "fixed_layers" in cfg:
        fixed_layers = int(cfg["fixed_layers"])
    else:
        fixed_layers = 17

    model = build(n_class, fixed_layers, cfg["model"])

    if args.model:
        model_path = args.model
    else:
        model_path = os.path.join(save_dir, '{}_weights.h5'.format(cfg['name']))
    
    model.load_weights(model_path, by_name=True)

    opt = SGD(lr=0.0001, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    datagen = ImageDataGenerator(rescale=None)

    test_generator, count = create_generator_from_csv_files(
        datagen, filenames_test, 1, shape[:2], limit=False)
    y_true = test_generator.labels

    labels = (test_generator.class_indices)
    label_list = list(labels.keys())
    labels2 = dict((v, k) for k, v in labels.items())

    # with open('label_dict.json', 'w') as fp:
    #     json.dump(labels2, fp)

    predict = model.predict_generator(test_generator, steps=count)

    y_pred = np.argmax(predict, axis=1)
    predictions = [labels2[k] for k in y_pred]


    if copy:
        copy_to_classified_folders(filenames_test, labels2, cfg["test_output"], cfg["name"], predictions)

    cm = confusion_matrix(y_true, y_pred)
    plot_conf_matrix(cm, cfg["confusion_matrix_classes"])

def create_generator_from_csv_files(datagen, filenames_file, batch_size, shape, limit=False):
    """Creates train/validation generators by filling them with image data from csv files
    """

    with open(filenames_file, "r") as readfile:
        filenames=readfile.readlines()
    count=len(filenames)

    df = pd.read_csv(filenames_file)

    if limit:
        limit_size=100
        df=df.head(limit_size)
        count=limit_size

    generator=datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col="filename",
        y_col="label",
        class_mode="categorical",
        batch_size=batch_size,
        validate_filenames=False,
        shuffle=False,
        target_size=shape
    )

    return generator, count-1

def generate(batch_size, filenames_train, filenames_val, shape):
    """Data generation and augmentation
    """

    datagen1 = ImageDataGenerator(
        rescale=None, shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2=ImageDataGenerator(rescale=None)

    train_generator, count1=create_generator_from_csv_files(
        datagen1, filenames_train, batch_size, shape)
    validation_generator, count2=create_generator_from_csv_files(
        datagen2, filenames_val, batch_size, shape)

    return train_generator, validation_generator, count1, count2


def build(n_class, fixed_layers, network):
    """Builds model for the respective number of classes.
    """
    if network == "vgg19":
        base_model=VGG19(include_top=False, weights='imagenet')
    
    # global spatial average pooling layer
    x=base_model.output
    x=GlobalAveragePooling2D()(x)

    # a fully-connected layer
    x=Dense(1024, activation='relu')(x)
    x=Dense(1024, activation='relu')(x)

    # a logistic layer
    predictions=Dense(n_class, activation='softmax')(x)

    # predictions=Dense(n_class, activation='sigmoid')(x)
    model=Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:fixed_layers]:
        layer.trainable=False
    for layer in model.layers[fixed_layers:]:
        layer.trainable=True

    return model
  

def train(config):
    """Trains the model using the parameters specified in config
    """
    with open(config, 'r') as f:
        cfg=json.load(f)

    save_dir = cfg['save_dir']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # shutil.copy(config, os.path.join(save_dir, "config.json"))

    shape=(int(cfg['height']), int(cfg['width']), 3)
    n_class=int(cfg['class_number'])
    batch=int(cfg['batch'])

    if "fixed_layers" in cfg:
        fixed_layers = int(cfg["fixed_layers"])
    else:
        fixed_layers = 17


    model=build(n_class, fixed_layers, cfg["model"])

    opt=SGD(learning_rate=cfg["learning_rate"])

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    train_generator, validation_generator, count1, count2 = generate(batch,
                                                                     cfg['filenames_train'],
                                                                     cfg['filenames_val'],
                                                                     shape[:2])

    hist=model.fit(
        train_generator,
        workers=8,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=cfg['epochs'])

    df=pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(save_dir, 'hist.csv'), encoding='utf-8', index=False)
    model.save_weights(os.path.join(save_dir, '{}_weights.h5'.format(cfg['name'])))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to json config file")
    parser.add_argument("-m", "--model", help="Explicit model path")

    parser.add_argument("-t", "--train", help="Train the model",
                        default=False, action="store_true")

    parser.add_argument("-e", "--eval", help="Evaluate the model",
                        default=False, action="store_true")

    args=parser.parse_args()

    if args.train:
        train(args.config)
    if args.eval:
        test(args.config, copy=True)


# python main.py -c ./configs/inflam_detect_config.json --train
# python main.py -c ./configs/inflam_detect_config.json --eval

# python main.py -c ./configs/inflam_classify_config.json --train
# python main.py -c ./configs/inflam_classify_config.json --eval


