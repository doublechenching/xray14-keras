#encoding: utf-8
from __future__ import print_function
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import plot_model

def mobile_net(input_shape, n_labels):
    base_mobilenet_model = MobileNet(input_shape=input_shape,
                                    include_top=False)
    multi_disease_model = Sequential()
    multi_disease_model.add(base_mobilenet_model)
    multi_disease_model.add(GlobalAveragePooling2D())
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(512))
    multi_disease_model.add(Dropout(0.5))
    multi_disease_model.add(Dense(n_labels, activation='sigmoid'))
    multi_disease_model.compile(optimizer='adam', loss='binary_crossentropy',
                                metrics=['binary_accuracy', 'mae'])
    multi_disease_model.summary()

    return multi_disease_model

if __name__ == "__main__":
    model = mobile_net([224, 224, 3], 8)
    plot_model(model, to_file='model.png', show_shapes=True)