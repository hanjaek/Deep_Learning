import tensorflow as tf
class MLP:
# "hidden_layer_conf" is the array indicates the number of layers (num_of_elements)
# and the number of elements in each layer.

    # hidden_layer_conf를 통해 은닉층 개수와 뉴런 수를 지정 가능.
    # 예: [4, 3] => 첫 번째 은닉층에 4개 뉴런, 두 번째 은닉층에 3개 뉴런

    # num_output_nodes를 통해 출력층 뉴런 개수를 설정.
    # num_output_nodes 예시(예: 분류라면 클래스 수, 회귀라면 1)
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        # 나중에 만들어질 모델을 담을 변수
        self.logic_op_model = None 

    def build_model(self):
        input_layer = tf.keras.Input(shape=[2, ]) # 입력층 생성: 입력은 2차원 (예: [x1, x2])
        hidden_layers = input_layer # 처음엔 입력층을 기준으로 layer 연결 시작
        if self.hidden_layer_conf is not None:
            for num_hidden_nodes in self.hidden_layer_conf:
                hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes,  # 이 은닉층에 들어갈 뉴런 수
                                                      activation=tf.keras.activations.sigmoid, # 활성화 함수는 sigmoid
                                                      use_bias=True)(hidden_layers)
        output = tf.keras.layers.Dense(units=self.num_output_nodes, # 출력층 뉴런 수
                                       activation=tf.keras.activations.sigmoid,
                                       use_bias=True)(hidden_layers) # 마지막 은닉층과 연결
        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output) # 모델 정의: 입력층부터 출력층까지 연결한 모델
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")


    """
        x: 입력 데이터 (예: [[0,0], [0,1], [1,0], [1,1]])

        y: 타겟 데이터 (예: [0, 1, 1, 0] for XOR)

        batch_size: 한 번에 학습할 샘플 수

        epochs: 전체 데이터를 몇 번 반복 학습할지
    """
    # 모델학습 수행
    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    # 입력 데이터를 받아 예측값 반환
    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction