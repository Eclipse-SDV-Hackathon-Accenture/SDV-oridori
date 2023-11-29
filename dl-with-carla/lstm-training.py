import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 하이퍼파라미터 설정
input_size = 5  # 입력 특성의 수 (속도, 스티어링, throttle, brake, lane detection)
hidden_size = 50  # LSTM 레이어의 히든 사이즈
output_size = 1  # 출력의 크기 (운전 점수)
num_layers = 2   # LSTM 레이어의 수
dropout_rate = 0.2  # 드롭아웃 비율
learning_rate = 0.001  # 학습률
batch_size = 64   # 배치 크기
sequence_length = 100  # 시퀀스 길이
epochs = 4200  # 학습 에포크 수
optimizer_type = 'adam'  # 최적화 알고리즘 타입: 'adam' 또는 'sgd'

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_sequence = lstm_out[:, -1, :]
        predictions = self.linear(last_sequence)
        return predictions

# 모델 초기화
model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_rate)

# 손실 함수와 최적화 알고리즘 정의
loss_function = nn.MSELoss()
if optimizer_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# CSV 파일 읽기 및 텐서 변환
can_df = pd.read_csv('carla_data.csv')
can_data = can_df.iloc[:, 1:6].values
can_data_tensor = torch.tensor(can_data, dtype=torch.float32)
print(can_data_tensor.size())
# 적절한 크기로 텐서 재구성
num_sequences = can_data_tensor.size(0) // sequence_length
if can_data_tensor.size(0) % sequence_length != 0:
    # 시퀀스 길이에 맞지 않는 데이터는 제외
    can_data_tensor = can_data_tensor[:num_sequences * sequence_length]

x_train = can_data_tensor.view(-1, sequence_length, input_size)
print(x_train.size())
# 점수가 저장된 CSV 파일 읽기
score_df = pd.read_csv('scores.csv')  # 헤더가 없다고 가정

# 데이터를 PyTorch 텐서로 변환
y_train = torch.tensor(score_df.values, dtype=torch.float32)
print(y_train.size())
# 학습
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = loss_function(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# 학습된 모델 저장
torch.save(model.state_dict(), 'trained_lstm_model.pth')