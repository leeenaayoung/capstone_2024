import paho.mqtt.client as mqtt
import time

broker_address = "localhost"
mqtt_port = 1883
topic = "ROBOT_DATA"

client = mqtt.Client()
client.connect(broker_address, mqtt_port)
client.loop_start()

file_path = r"C:\\Users\\kdh03\\Desktop\\캡스톤\\capstone_2024\\data\\golden_sample\\clock_big.txt"

with open(file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()
    if line:  # 빈 줄은 무시
        client.publish(topic, line)
        print(f"Published message: {line}")
        time.sleep(0.02)  # 20ms 간격 발행
