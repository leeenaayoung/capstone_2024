import torch
import os
from model import TransformerModel
# from generation_m_model import GenerationModel 

def get_latest_opset_version():
    # PyTorch에서 지원하는 최대 ONNX opset 버전을 반환
    return 18  # 예시: 현재 최신 릴리즈에서 지원되는 최대 opset 버전 (수동 업데이트 필요)

# 분류 모델 변환
def convert_to_onnx(pth_model_path, onnx_model_path, input_dim, d_model, nhead, num_layers, num_classes, max_len=2000):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화 및 가중치 로드
    model = TransformerModel(input_dim, d_model, nhead, num_layers, num_classes, max_len=max_len).to(device)
    checkpoint = torch.load(pth_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()  # 추론 모드로 설정
    
    # 더미 입력 데이터 생성
    dummy_input = torch.randn(1, max_len, input_dim).to(device)

    # 최신 opset 버전 가져오기
    latest_opset_version = get_latest_opset_version()
    
    # ONNX 변환
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=latest_opset_version, 
        do_constant_folding=True,
        input_names=['input'],       # 입력 텐서 이름
        output_names=['output'],     # 출력 텐서 이름
        dynamic_axes={'input': {0: 'batch_size', 1: 'seq_len'}, 'output': {0: 'batch_size', 1: 'seq_len'}}  # 동적 배치 및 시퀀스 길이 지원
    )
    
    print(f"ONNX 모델이 {onnx_model_path}에 저장되었습니다.")


def main():
    pth_model_path = "best_classification_model.pth"  # 학습된 PyTorch 모델 경로
    onnx_model_path = "best_classification_model.onnx"  # 저장할 ONNX 파일 경로
    
    # Transformer 모델 하이퍼파라미터 설정
    input_dim = 21 
    d_model = 32
    nhead = 2
    num_layers = 3
    num_classes = 20    # 궤적 종류 개수
    max_len = 2000 

    if os.path.exists(pth_model_path):
        print("학습된 PyTorch 모델을 ONNX로 변환 중...")
        convert_to_onnx(pth_model_path, onnx_model_path, input_dim, d_model, nhead, num_layers, num_classes, max_len)
    else:
        print(f"Error: {pth_model_path} 파일이 존재하지 않습니다.")
        print("모델 학습을 먼저 실행해야 합니다.")

if __name__ == "__main__":
    main()

# 생성 모델 변환
# def convert_to_onnx(pth_model_path, onnx_model_path, input_size):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 모델 초기화 및 디바이스 이동
#     model = GenerationModel().model
#     model.to(device)
    
#     # 가중치 로드
#     checkpoint = torch.load(pth_model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # 더미 입력 데이터 생성
#     dummy_input = torch.randn(*input_size).to(device)
    
#     # 최신 opset 버전 가져오기
#     latest_opset_version = get_latest_opset_version()
    
#     # ONNX 변환
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_model_path,
#         export_params=True,
#         opset_version=latest_opset_version,  # 최신 opset 버전 사용
#         do_constant_folding=True,
#         input_names=['input'],
#         output_names=['output'],
#         dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size', 1: 'sequence_length'}}
#     )
    
#     print(f"ONNX 모델이 {onnx_model_path}에 저장되었습니다. Opset Version: {latest_opset_version}")

# def main():
#     pth_model_path = "best_classification_model.pth"  # 학습된 PyTorch 모델 경로
#     onnx_model_path = "best_classification_model.onnx"  # 저장할 ONNX 파일 경로
#     input_size = (1, 30, 4)  # 예: (batch_size=1, sequence_length=30, num_features=4)

#     if os.path.exists(pth_model_path):
#         print("학습된 PyTorch 모델을 ONNX로 변환 중...")
#         convert_to_onnx(pth_model_path, onnx_model_path, input_size)
#     else:
#         print(f"Error: {pth_model_path} 파일이 존재하지 않습니다.")
#         print("모델 학습을 먼저 실행해야 합니다.")

# if __name__ == "__main__":
#     main()
