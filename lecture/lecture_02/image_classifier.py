import tensorflow as tf
import matplotlib.pyplot as plt
 # from ImageClassifier import ImageClassifier   ## 일단 주석 처리해 둠. ImageClassifier class만들고 나서 주석 해제

def run_classifier():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("Train data shape")
    print(train_images.shape)
    print("Train data labels")
    print(train_labels)
    print("Test data shape")
    print(test_images.shape)
    print("Test data labels")
    print(test_labels)

    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()

    ##### classifier train and predict - begin
    epochs = 10 # 1500

    mlp_classifier = ImageClassifier_MLP(img_shape_x=28, img_shape_y=28, num_labels=10)
    mlp_classifier.build_model()
    train_labels = ImageClassifier_MLP.to_onehotvec_label(train_labels, 10)
    mlp_classifier.fit(train_imgs=train_images, train_labels=train_labels, num_epochs=epochs)

    predicted_labels = mlp_classifier.predict(test_imgs=test_images)
    predicted_labels = tf.math.argmax(input=predicted_labels, axis=1)

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predicted_labels[i]])
    plt.show()
    ##### classifier train and predict - end

if __name__ == "__main__":
    # execute only if run as a script
    run_classifier()