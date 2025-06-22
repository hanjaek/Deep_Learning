import tensorflow as tf
from MNISTData import MNISTData
from AutoEncoder import AutoEncoder
import numpy as np

# def add_noise(x):
#     mask = np.random.binomial(1, 0.5, size=x.shape)
#     return x * mask

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
    num_test_items = 1000
    test_data = data_loader.x_test[:num_test_items]
    test_label = data_loader.y_test[:num_test_items]
    test_data_x_print = test_data.reshape(num_test_items, data_loader.width, data_loader.height)

    # 노이즈 테스트 데이터 추가
    # test_data_noised = add_noise(test_data)
    # test_label = data_loader.y_test[0:num_test_items]
    # test_data_x_print = test_data_noised.reshape(num_test_items, data_loader.width, data_loader.height)

    print("const by codes")
    reconst_data = auto_encoder.en_decoder.predict(test_data)
    reconst_data_x_print = reconst_data.reshape(num_test_items, data_loader.width, data_loader.height)
    reconst_data_x_print = tf.math.sigmoid(reconst_data_x_print)
    # MNISTData.print_56_pair_images(test_data_x_print, reconst_data_x_print, test_label)

    # 평균 부분 덮어씌우는 문제 해결 완료
    print("const by code means for each digit")
    avg_codes = np.zeros([10, 32])
    std_codes = np.zeros([10, 32])
    avg_add_cnt = np.zeros([10])

    latent_vecs = auto_encoder.encoder.predict(test_data)

    for i, label in enumerate(test_label):
        avg_codes[label] += latent_vecs[i]
        avg_add_cnt[label] += 1.0

    for i in range(10):
        if avg_add_cnt[i] > 0.1:
            avg_codes[i] /= avg_add_cnt[i]
            
    for i in range(10):
        if avg_add_cnt[i] > 0:
            std_codes[i] = np.std(latent_vecs[np.array(test_label) == i], axis=0)

    new_codes = []
    for i in range(10):
        for _ in range(5):  # 각 숫자당 5개 생성
            rand = np.random.uniform(-1, 1, size=32)
            new_code = avg_codes[i] + std_codes[i] * rand  # element-wise 연산
            new_codes.append(new_code)

    new_codes = np.array(new_codes)

    new_code_tensor = tf.convert_to_tensor(new_codes)
    reconst_data_by_vecs = auto_encoder.decoder.predict(new_code_tensor)
    reconst_data_x_by_rand = reconst_data_by_vecs.reshape(50, data_loader.width, data_loader.height)
    label_list = [i for i in range(10) for _ in range(5)]
    MNISTData.print_50_images(reconst_data_x_by_rand, label_list)