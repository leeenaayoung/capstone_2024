import paho.mqtt.client as mqtt
import time
import pandas as pd
import socket

# MQTT 설정
broker_address = 'localhost'
mqtt_port = 1883
topic_receive = "ROBOT_DATA"   # 로봇에서 보내는 데이터 토픽

# TCP 설정 (Unity 서버의 IP와 포트에 맞게 수정)
unity_address = '127.0.0.1'  # 예시: 로컬에서 테스트
unity_port = 5005

# MQTT 클라이언트 생성
client = mqtt.Client()

# TCP 소켓 생성 및 Unity 서버에 연결
def create_tcp_connection(address, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((address, port))
    return sock

try:
    unity_sock = create_tcp_connection(unity_address, unity_port)
    print(f"Connected to Unity at {unity_address}:{unity_port}")
except Exception as e:
    print("Error connecting to Unity:", e)
    unity_sock = None

def preprocess_trajectory_data_single(data_list, scaler=None):
    """
    단일 데이터 전처리 함수.
    data_list: ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
                'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']
    """
    columns = ['r', 'sequence', 'timestamp', 'deg', 'deg/sec', 'mA',
               'endpoint', 'grip/rotation', 'torque', 'force', 'ori', '#']
    df_t = pd.DataFrame([data_list], columns=columns)
    
    # 'r' 값이 's'인 경우는 필터링 (예: 상태 메시지 등)
    if df_t.iloc[0]['r'] == 's':
        return None

    data_v = df_t.drop(['r', 'grip/rotation', '#'], axis=1)
    
    # 각 컬럼 split 처리: 슬래시("/") 기준으로 분리하여 새로운 컬럼 생성
    splits = {
        'endpoint': ['x_end', 'y_end', 'z_end'],
        'deg': ['deg1', 'deg2', 'deg3', 'deg4'],
        'deg/sec': ['degsec1', 'degsec2', 'degsec3', 'degsec4'],
        'torque': ['torque1', 'torque2', 'torque3'],
        'force': ['force1', 'force2', 'force3'],
        'ori': ['yaw', 'pitch', 'roll']
    }
    
    for col, new_cols in splits.items():
        v_split = data_v[col].astype(str).str.split('/')
        for idx, new_col in enumerate(new_cols):
            data_v[new_col] = v_split.str.get(idx)
    
    # 원본 컬럼 제거
    data_v = data_v.drop(['deg', 'deg/sec', 'mA', 'endpoint', 'torque', 'force', 'ori'], axis=1)
    
    # 숫자형 변환 (변환 실패 시 NaN은 0으로 채움)
    data_v = data_v.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # deg2와 deg4 값에서 90도 빼기 (필요에 따라 조정)
    data_v['deg2'] = data_v['deg2'] - 90
    data_v['deg4'] = data_v['deg4'] - 90
    
    # time 열 생성: timestamp - sequence - 1 (예시)
    data_v['time'] = data_v['timestamp'] - data_v['sequence'] - 1
    data_v = data_v.drop(['sequence', 'timestamp'], axis=1)
    
    return data_v

def on_connect(client, userdata, flags, rc):
    print("MQTT Connected with result code " + str(rc))
    client.subscribe(topic_receive)

def on_message(client, userdata, message):
    start_time = time.perf_counter()
    payload = message.payload.decode("utf-8").strip()
    # 들어오는 메시지가 콤마(,)로 구분된 문자열임을 가정
    data_list = payload.split(',')
    processed_data = preprocess_trajectory_data_single(data_list)
    if processed_data is not None:
        # 전처리된 DataFrame을 JSON 형식으로 변환하여 Unity에 전송
        result_json = processed_data.to_json(orient='records')
        if unity_sock:
            try:
                # TCP 소켓으로 전송 (바이트 인코딩 필요)
                unity_sock.sendall(result_json.encode('utf-8'))
                print("Sent data to Unity:", result_json)
            except Exception as e:
                print("Error sending data to Unity:", e)
        else:
            print("Unity socket is not available.")
    else:
        print("Message filtered out (r == 's')")
    
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000  # 밀리초 단위
    print(f"Processing and sending took {elapsed_time:.6f} ms")

client.on_connect = on_connect
client.on_message = on_message

# MQTT 브로커에 연결 및 네트워크 루프 실행 (별도 스레드로)
client.connect(broker_address, port=mqtt_port)
client.loop_start()

# 메인 루프: 20ms 간격으로 유지 (필요 시 추가 작업 가능)
try:
    while True:
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()
    if unity_sock:
        unity_sock.close()