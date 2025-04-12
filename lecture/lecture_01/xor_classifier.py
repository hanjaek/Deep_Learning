import tensorflow as tf
from MLP import MLP

def xor_classifier_example ():
    input_data = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]) # 2개의 입력값으로 구성된 4가지 조합
    input_data = tf.cast(input_data, tf.float32)
    xor_labels = tf.constant([0.0, 1.0, 1.0, 0.0]) # XOR의 정답 (0, 1, 1, 0)
    xor_labels = tf.cast(xor_labels, tf.float32)
    batch_size = 1
    epochs = 1500
    mlp_classifier = MLP(hidden_layer_conf=[4], num_output_nodes=1) # hidden_layer_conf=[4]: 은닉층 1개, 뉴런 수 4개, num_output_nodes=1: 출력층 뉴런 1개 (결과가 0 또는 1)

    mlp_classifier.build_model() # 모델 구조 생성
    mlp_classifier.fit(x=input_data, y=xor_labels, batch_size=batch_size, epochs=epochs)

    ######## MLP XOR prediciton
    prediction = mlp_classifier.predict(x=input_data, batch_size=batch_size) # 학습이 끝난 모델로 입력 데이터들에 대해 예측값을 얻음
    input_and_result = zip(input_data, prediction)
    print("====== MLP XOR classifier result =====")
    for x, y in input_and_result:
        if y > 0.5:
            print("%d XOR %d => %.2f => 1" % (x[0], x[1], y))
        else:
            print("%d XOR %d => %.2f => 0" % (x[0], x[1], y))

    """
        예측값 y가 0.5보다 크면 1로 판단, 아니면 0

        예: 0 XOR 1 => 0.97 => 1 이런 식으로 출력
    """
    
# Entry point
if __name__ == '__main__':
    xor_classifier_example()