import tensorflow as tf
import numpy as np

class ImageClassifier_MLP:
    def __init__(self, img_shape_x, img_shape_y, num_labels):
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.num_labels = num_labels
        self.classifier = None

    def fit(self, train_imgs, train_labels, num_epochs):
        self.classifier.fit(train_imgs, train_labels, epochs=num_epochs)

    def predict(self, test_imgs):
        predictions = self.classifier.predict(test_imgs)
        return predictions

    # class ImageClassifier_MLP의 member method
    def build_CNN_model(self):
        input_layer = tf.keras.Input(shape=[self.img_shape_x, self.img_shape_y,1,])

        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
                                              padding='valid', activation='relu')(input_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D((2,2))(hidden_layer)

        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1),
                                              padding='valid', activation='relu')(input_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D((2,2))(hidden_layer)

        hidden_layer = tf.keras.layers.Flatten()(hidden_layer)

        hidden_layer = tf.keras.layers.Dense(units=64, activation='relu')(hidden_layer)
        output = tf.keras.layers.Dense(units=self.num_labels, activation='softmax')(hidden_layer)
        

        classifier_model = tf.keras.Model(inputs=input_layer, outputs=output)
        classifier_model.summary()

        opt_alg = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss_cross_e = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier_model.compile(optimizer=opt_alg, loss=loss_cross_e, metrics=['accuracy'])
        self.classifier = classifier_model

    # class ImageClassifier_MLP의 static method
    @staticmethod
    def to_onehotvec_label(index_labels, dim):
        num_labels = len(index_labels)
        onehotvec_labes = np.zeros((num_labels, dim))
        for i, idx in enumerate(index_labels):
            onehotvec_labes[i][idx] = 1.0
        onehotvec_labes_tf = tf.convert_to_tensor(onehotvec_labes)
        return onehotvec_labes_tf