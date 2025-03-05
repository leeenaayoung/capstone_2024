import os
import pandas as pd

# (1) evaluate.py 안의 TrajectoryEvaluator
from evaluate import TrajectoryEvaluator

# (2) utils.py 안의 전처리 함수
from utils import preprocess_trajectory_data

def main():
    # 1) 폴더 경로 설정
    #    - Windows 절대경로 (예: C:\Users\kdh03\Desktop\캡스톤\capstone_2024\data)
    base_dir = r"C:\\Users\\kdh03\Desktop\\캡스톤\\capstone_2024\\data"
    golden_dir = os.path.join(base_dir, "golden_sample")

    # 2) TrajectoryEvaluator 초기화
    evaluator = TrajectoryEvaluator()

    # 3) golden_sample 폴더 내 txt 파일 목록
    golden_files = [f for f in os.listdir(golden_dir) if f.endswith('.txt')]
    if not golden_files:
        raise ValueError("golden_sample 디렉토리에 txt 파일이 없습니다.")

    for filename in golden_files:
        file_path = os.path.join(golden_dir, filename)

        # ========== (A) CSV 읽기 ==========
        # CSV를 바로 DataFrame으로 읽음
        df = pd.read_csv(file_path, delimiter=',')

        # 만약 CSV 헤더가 없거나, 순서가 맞지 않으면 아래처럼 직접 매핑할 수도 있음:
        # df.columns = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
        #               'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']

        # ========== (B) utils.py 전처리 ========== 
        # 전처리 함수는 "data_list" (row 단위 리스트) 형태를 인자로 받으므로,
        # df를 원하는 순서의 컬럼으로 뽑아서 values.tolist()를 만듭니다.
        columns_needed = [
            'r','sequence','timestamp','deg','deg/sec','mA',
            'endpoint','grip/rotation','torque','force','ori','#'
        ]
        # 혹시 파일에 columns_needed가 전부 있는지 체크
        missing = [c for c in columns_needed if c not in df.columns]
        if missing:
            print(f"[Warning] {filename}에서 부족한 컬럼: {missing}")
            # 필요한 경우 continue로 넘어가거나, 오류 처리
            continue

        data_list = df[columns_needed].values.tolist()

        # 전처리(스케일러가 있다면 넣고, 없으면 None)
        # return_raw=True 로 하면 (스케일된 DF, 원본 DF) 튜플을 반환
        scaled_df = preprocess_trajectory_data(data_list, scaler=None, return_raw=True)

        # 여기서 'deg1','deg2','deg3','deg4'가 포함된 DataFrame이
        # raw_df 혹은 scaled_df에 생성되었을 것입니다.
        # 이후 evaluator에서 deg1,deg2,deg3,deg4 등을 참고해 평가합니다.

        # ========== (C) 궤적 타입 추정(파일명 기준) ==========
        # (예) line_d_l.txt → "line_d_l"를 trajectory_type으로 쓸 수도 있고,
        # 단순히 ".txt"만 제외하기 위해:
        trajectory_type = filename.replace(".txt", "")

        # ========== (D) 평가 함수 호출 ==========
        # evaluate_trajectory()는 "deg1~deg4" 칼럼을 찾아서 처리하므로
        # raw_df(스케일링 전)나 scaled_df 중 *어떤 것*을 전달할지 결정해야 합니다.
        # 보통 end-effector 계산은 "실제 각도"를 사용하는 게 맞기 때문에 raw_df 권장.
        try:
            result = evaluator.evaluate_trajectory(scaled_df, trajectory_type)
        except ValueError as e:
            print(f"\n[Error] {filename} => {e}")
            continue

        # ========== (E) 평가 결과 출력 ==========
        print(f"\n===== {filename} 평가 결과 =====")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            elif isinstance(value, tuple):
                # 튜플일 경우 각 원소만 반올림
                rounded_values = tuple(round(v, 4) for v in value)
                print(f"{metric}: {rounded_values}")
            else:
                print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
