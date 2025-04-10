#!/usr/bin/env python3
import os
import glob
import time
import socket
import pandas as pd
import paho.mqtt.client as mqtt
import keyboard

# --- 설정 부분 ---
# 실시간 저장 폴더
save_dir = os.path.join("capstone_2024", "trajectory")
os.makedirs(save_dir, exist_ok=True)

# 보간된 궤적 파일 폴더
gen_dir = os.path.join("capstone_2024", "generation_trajectory")

# MQTT 설정
broker_address = '192.168.0.25'
mqtt_port = 1883
topic_receive    = "85"    # 로봇 → 이 스크립트
topic_send_robot = "85/S"  # 이 스크립트 → 로봇

# Unity TCP 설정
unity_address = '127.0.0.1'
unity_port    = 5005

# 녹화 제어
recording   = False
file_handle = None
stop_time   = None  # stop_record 시각 저장

# MQTT 클라이언트 생성
client = mqtt.Client()

# Unity 소켓 연결
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

# --- 키 이벤트 콜백 ---
def start_record(e):
    global recording, file_handle
    if not recording:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"data_{ts}.txt"
        file_handle = open(os.path.join(save_dir, fname), 'w', encoding='utf-8')
        recording = True
        print(f">>> Recording started → {fname}")

def stop_record(e):
    global recording, file_handle, stop_time
    if recording:
        recording = False
        file_handle.close()
        print(">>> Recording stopped")
        stop_time = time.time()
        process_interpolated_trajectory()

keyboard.on_press_key('s', start_record)
keyboard.on_press_key('e', stop_record)

# --- 데이터 전처리 (기존) ---
def preprocess_trajectory_data_single(data_list):
    if len(data_list) != 12:
        print(f"Invalid data length: {len(data_list)}. Expected 12 elements.")
        return None

    cols = ['r','sequence','timestamp','deg','deg/sec','mA',
            'endpoint','grip/rotation','torque','force','ori','#']
    df = pd.DataFrame([data_list], columns=cols)
    if df.iloc[0]['r'] == 's':
        return None

    dv = df.drop(['r','grip/rotation','#'], axis=1)
    splits = {
        'endpoint': ['x_end','y_end','z_end'],
        'deg': ['deg1','deg2','deg3','deg4'],
        'deg/sec': ['degsec1','degsec2','degsec3','degsec4'],
        'torque': ['torque1','torque2','torque3'],
        'force': ['force1','force2','force3'],
        'ori': ['yaw','pitch','roll']
    }
    for col, names in splits.items():
        parts = dv[col].astype(str).str.split('/')
        for i,name in enumerate(names):
            dv[name] = parts.str.get(i)
    dv = dv.drop(list(splits.keys()) + ['mA'], axis=1)
    dv = dv.apply(pd.to_numeric, errors='coerce').fillna(0)
    dv['deg2'] -= 90
    dv['deg4'] -= 90
    dv['time'] = dv['timestamp'] - dv['sequence'] - 1
    dv = dv.drop(['sequence','timestamp'], axis=1)
    return dv

# --- MQTT 콜백 (기존) ---
def on_connect(client, userdata, flags, rc):
    print("MQTT Connected with result code", rc)
    client.subscribe(topic_receive)

def on_message(client, userdata, message):
    global recording, file_handle
    payload = message.payload.decode('utf-8').strip()
    data_list = payload.split(',')
    processed = preprocess_trajectory_data_single(data_list)
    if processed is not None:
        js = processed.to_json(orient='records')
        # Unity 전송
        if unity_sock:
            try:
                unity_sock.sendall(js.encode('utf-8'))
            except Exception as e:
                print("Error sending to Unity:", e)
        # 파일 기록
        if recording and file_handle:
            file_handle.write(js + '\n')
    else:
        print("Filtered out (r=='s')")

    # 처리 시간 출력 (선택)
    # elapsed = (time.perf_counter() - start) * 1000
    # print(f"Processing took {elapsed:.3f} ms")

client.on_connect = on_connect
client.on_message = on_message

# --- 보간 파일 대기용 폴링 함수 ---
def wait_for_new_file(dir_path, since, timeout=60, interval=0.5):
    """
    dir_path: 모니터할 폴더
    since:   stop_record 호출 시각 (timestamp)
    timeout: 최대 대기 시간(초)
    interval: 폴링 주기(초)
    """
    start = time.time()
    while time.time() - start < timeout:
        files = glob.glob(os.path.join(dir_path, "*.txt"))
        new = [f for f in files if os.path.getctime(f) > since]
        if new:
            return max(new, key=os.path.getctime)
        time.sleep(interval)
    return None

# --- 보간된 궤적 처리 함수 ---
def process_interpolated_trajectory():
    # 1) 새 파일이 생길 때까지 대기
    latest = wait_for_new_file(gen_dir, stop_time, timeout=60, interval=0.5)
    if not latest:
        print("[Error] 보간 파일 생성 대기 시간 초과")
        return
    print(f"[Interpolation] Found new file: {latest}")

    # 2) 첫 줄에서 deg 파싱
    with open(latest, 'r', encoding='utf-8') as f:
        _header    = f.readline()
        first_line = f.readline().strip()
    if not first_line:
        print("[Error] First data line is empty!")
        return

    parts = first_line.split(',')
    if len(parts) < 4:
        print(f"[Error] Unexpected format: {first_line}")
        return

    deg_vals = parts[3].split('/')
    try:
        deg_scaled = [int(float(v) * 10) for v in deg_vals]
    except ValueError:
        print(f"[Error] Cannot convert deg values: {deg_vals}")
        return
    deg_str = "/".join(str(v) for v in deg_scaled)
    msg = f"s,K,8,7,{deg_str},#"

    # 3) MQTT로 로봇에 발행
    res = client.publish(topic_send_robot, msg)
    if res.rc == mqtt.MQTT_ERR_SUCCESS:
        print(f"[Interpolation] Sent to robot: {msg}")
    else:
        print(f"[Error] MQTT publish failed: {res.rc}")

    # 4) 보간된 전체 궤적 Unity 전송
    with open(latest, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if unity_sock:
        for line in lines:
            try:
                unity_sock.sendall(line.strip().encode('utf-8'))
                time.sleep(0.005)  # 과부하 방지
            except Exception as e:
                print("Error sending to Unity:", e)
                break
        print(f"[Interpolation] Sent {len(lines)} lines to Unity")
    else:
        print("[Error] Unity socket not available")

# --- MQTT 시작 & 메인 루프 ---
client.connect(broker_address, mqtt_port)
client.loop_start()

try:
    while True:
        time.sleep(0.02)
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    client.disconnect()
    if unity_sock:
        unity_sock.close()
    if recording and file_handle:
        file_handle.close()