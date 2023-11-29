import pandas as pd
import torch
import torch.nn as nn

# LSTM 모델 클래스 정의
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

# 하이퍼파라미터 설정
input_size = 5  # 입력 특성의 수
hidden_size = 50  # LSTM 레이어의 히든 사이즈
output_size = 1  # 출력의 크기
num_layers = 2   # LSTM 레이어의 수
dropout_rate = 0.2  # 드롭아웃 비율
sequence_length = 100  # 시퀀스 길이

# 모델 파라미터 로드
model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout_rate)
model.load_state_dict(torch.load('trained_lstm_model.pth'))
model.eval()

# CSV 파일 읽기 및 테스트 데이터 준비
test_df = pd.read_csv('./carla-data-test/carla_data_seo.csv')
test_data = test_df.iloc[:, 1:6].values
test_tensor = torch.tensor(test_data, dtype=torch.float32)
tensor_array = torch.zeros([100, 5]) 
print(tensor_array)
print(test_tensor)

num_sequences = test_tensor.size(0) // sequence_length
if test_tensor.size(0) % sequence_length != 0:
    # 시퀀스 길이에 맞지 않는 데이터는 제외
    test_tensor = test_tensor[:num_sequences * sequence_length]

x_test = test_tensor.view(-1, sequence_length, input_size)

# 테스트 실행
with torch.no_grad():
    test_output = model(x_test)
    # 평균 점수 계산
    average_score = test_output.mean().item()
    print("Average Test Output:", average_score)

