import tensorflow as tf
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

##################### 과제 1 구현 #####################
def add_noise(x):
    mask = np.random.binomial(1, 0.5, size=x.shape)
    return x * mask

if __name__ == "__main__":

    print("Hi. I am an AutoEncoder Tester.")
    batch_size = 32
    num_epochs = 5

    data_loader = MNISTData()
    data_loader.load_data()
    x_train = data_loader.x_train
    input_output_dim = data_loader.in_out_dim

    auto_encoder = AutoEncoder()
    auto_encoder.build_model()
    load_path = "./model/denoising_ae.weights.h5"
    print("load model weights from %s" % load_path)
    auto_encoder.load_weights(load_path)

    # print for test
    num_test_items = 56
    test_data = data_loader.x_test[0:num_test_items, :]

    # 노이즈 테스트 데이터 추가
    test_data_noised = add_noise(test_data)
    test_label = data_loader.y_test[0:num_test_items]
    test_data_x_print = test_data_noised.reshape(num_test_items, data_loader.width, data_loader.height)

    ##################### 과제 2 출력 #####################
    print("const by codes")
    reconst_data = auto_encoder.en_decoder.predict(test_data_noised)
    reconst_data_x_print = reconst_data.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_print = tf.math.sigmoid(reconst_data_x_print)
    MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_print, test_label)

    num_test_items2 = 1000
    test_data2 = data_loader.x_test[:num_test_items2]
    test_label2 = data_loader.y_test[:num_test_items2]
    test_data_x_print2 = test_data2.reshape(num_test_items2, data_loader.width, data_loader.height)

    # 평균 부분 덮어씌우는 문제 해결 완료
    print("const by code means for each digit")
    avg_codes = np.zeros([10, 32])
    std_codes = np.zeros([10, 32])
    avg_add_cnt = np.zeros([10])

    latent_vecs = auto_encoder.encoder.predict(test_data2)

    for i, label in enumerate(test_label2):
        avg_codes[label] += latent_vecs[i]
        avg_add_cnt[label] += 1.0

    for i in range(10):
        if avg_add_cnt[i] > 0.1:
            avg_codes[i] /= avg_add_cnt[i]
            # std = np.std(latent_vecs[np.array(test_label) == i], axis=0)
            # print(f"Label {i} 평균 code:\n{avg_codes[i]}")
            # print(f"Label {i} 표준편차:\n{std}\n")
    
    for i in range(10):
        if avg_add_cnt[i] > 0:
            std_codes[i] = np.std(latent_vecs[np.array(test_label2) == i], axis=0)

    ##################### 과제 3 출력 #####################
    avg_code_tensor = tf.convert_to_tensor(avg_codes)
    reconst_data_by_vecs = auto_encoder.decoder.predict(avg_code_tensor)
    reconst_data_x_by_mean_print = reconst_data_by_vecs.reshape(10, data_loader.width, data_loader.height)
    label_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    MNISTData.print_10_images(reconst_data_x_by_mean_print, label_list)

    ##################### 과제 4 출력 #####################
    new_codes = []
    for i in range(10):
        for _ in range(5):  
            rand = np.random.uniform(-1, 1, size=32)
            new_code = avg_codes[i] + std_codes[i] * rand
            new_codes.append(new_code)

    new_codes = np.array(new_codes)

    new_code_tensor = tf.convert_to_tensor(new_codes)
    reconst_data_by_rand = auto_encoder.decoder.predict(new_code_tensor)
    reconst_data_x_by_rand = reconst_data_by_rand.reshape(50, data_loader.width, data_loader.height)
    label_list = [i for i in range(10) for _ in range(5)]
    MNISTData.print_50_images(reconst_data_x_by_rand, label_list)

    ##################### 과제 5 출력 #####################
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    avg_codes_2d = tsne.fit_transform(avg_codes) 

    plt.figure(figsize=(8, 6))
    for i in range(10):
        x, y = avg_codes_2d[i]
        plt.scatter(x, y, s=100)
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=12)

    plt.title("t-SNE Visualization of Avg Codes (per digit)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()