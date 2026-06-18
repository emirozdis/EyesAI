import cv2
import time
import subprocess
import requests
import serial
import logging
import signal
import sys
from PIL import Image
from picamera2 import Picamera2
from ultralytics import YOLO
import google.generativeai as genai

# ========= YAPILANDIRMA =========
LIDAR_PORT = "/dev/ttyAMA0"
BAUD_RATE = 115200
MIN_DISTANCE = 20
MAX_DISTANCE = 1200
MIN_STRENGTH = 2000
DEBOUNCE_SECONDS = 1
OUTPUT_FILE = "output.mp3"
TARGET_CLASSES = ["traffic light", "stop sign", "person", "dog", "cat", "car", "motorcycle", "bicycle"]

# Gerçek API anahtarlarınızla değiştirin
GEMINI_API_KEY = "**"
ELEVENLABS_API_KEY = "**"
VOICE_ID = "nPczCjzI2devNBz1zQrb"

# ========= BAŞLANGIÇ AYARLARI =========
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EYESAI")

# YOLOv8 Modelini yükle
model = YOLO("yolov8n.pt")
COCO_CLASSES = model.names

# Gemini yapılandırması
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={"max_output_tokens": 200, "response_mime_type": "text/plain"},
    system_instruction=(
        "Sen görme engelli bir kullanıcıya çevreyi tanımlayan yardımcı bir asistansın. "
        "Cevaplarını kısa (maksimum iki cümle), net ve spesifik tut. "
        "İnsanlar, araçlar, trafik işaretleri, evcil hayvanlar ve yürümeyi engelleyebilecek nesneleri belirt. "
        "Yön (sol/sağ/ön) ve yaklaşık mesafeyi (varsa) belirt."
    )
)

# ========= KAMERA BAŞLATMA =========
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(2)

running = True
last_trigger_time = 0
last_detection_time = 0  # Son algılama zamanını takip etmek için eklendi


# ========= YARDIMCI FONKSİYONLAR =========

def generate_audio(text: str):
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {"Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        with open(OUTPUT_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        subprocess.run(["mpg321", OUTPUT_FILE])
    except Exception as e:
        log.error(f"[Ses Hatası] {e}")


def analyze_frame(frame, distance):
    try:
        # Görüntü 4 kanallıysa (RGBA), BGR'ye dönüştür
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # BGR'den RGB'ye çevir
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Gemini için istem oluştur
        prompt = (
            f"Sen görme engelli bir kişiye çevreyi anlamasında yardımcı oluyorsun. "
            f"{distance} santimetre mesafede bir şey algılandı. "
            f"Sadece en önemli ve faydalı detayları tanımla — nesneler, insanlar, tabelalar veya yürümeyi etkileyebilecek herhangi bir şey dahil. "
            f"Tahmini konumu (sol/sağ/ön) ve mesafeyi belirt."
        )

        # Gemini modelinden yanıt al
        response = gemini_model.generate_content([prompt, img])
        description = response.text.strip()

        # Logla ve ses üret
        log.info(f"[Gemini AI] {description}")
        generate_audio(description)
        
    except Exception as e:
        log.error(f"[AI  Hatası] {e}")


def detect_objects(frame):
    try:
        # Görüntü 4 kanallıysa (RGBA), 3 kanala düşür
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # BGR'den RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO modeline gönder
        results = model(frame_rgb)
        for result in results:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for class_id in class_ids:
                if COCO_CLASSES[class_id] in TARGET_CLASSES:
                    return True
        return False
    except Exception as e:
        log.error(f"[Algılama Hatası] {e}")
        return False


def read_lidar():
    try:
        lidar = serial.Serial(LIDAR_PORT, BAUD_RATE, timeout=1)
        while running:
            if lidar.read(1) == b'\x59' and lidar.read(1) == b'\x59':
                packet = lidar.read(7)
                if len(packet) == 7:
                    distance = packet[0] + (packet[1] << 8)
                    strength = packet[2] + (packet[3] << 8)
                    if MIN_DISTANCE <= distance <= MAX_DISTANCE and strength >= MIN_STRENGTH:
                        return distance
    except Exception as e:
        log.error(f"[LIDAR Hatası] {e}")
    return None


# ========= İŞLEM DÖNGÜSÜ =========

def process_loop():
    global last_trigger_time, last_detection_time
    while running:
        distance = read_lidar()
        frame = picam2.capture_array()

        now = time.time()
        triggered = False

        if distance is not None and distance < 30:
            if now - last_trigger_time > DEBOUNCE_SECONDS:
                log.info(f"[LIDAR] {distance} santimetre mesafede engel algılandı.")
                triggered = True

        elif detect_objects(frame):
            if now - last_trigger_time > DEBOUNCE_SECONDS:
                log.info("[YOLO] İlgili nesne algılandı")
                triggered = True

        if triggered:
            last_trigger_time = now
            last_detection_time = now
            analyze_frame(frame, distance or 100)

        # Eğer 10 saniyedir algılama yoksa otomatik tetikle
        if now - last_detection_time >= 10:
            log.info("[Oto-tetikleme] 10 saniye hareketsizlik.")
            analyze_frame(frame, distance or 100)
            last_detection_time = now

        time.sleep(0.1)


# ========= KAPATMA İŞLEYİCİSİ =========

def shutdown_handler(signum, frame):
    global running
    log.info("shutting down...")
    running = False
    picam2.stop()
    sys.exit(0)


# ========= ASCII SANATI =========
ascii_art = """
 /$$$$$$$$ /$$     /$$ /$$$$$$$$  /$$$$$$         /$$$$$$  /$$$$$$
| $$_____/|  $$   /$$/| $$_____/ /$$__  $$       /$$__  $$|_  $$_/
| $$       \  $$ /$$/ | $$      | $$  \__/      | $$  \ $$  | $$  
| $$$$$     \  $$$$/  | $$$$$   |  $$$$$$       | $$$$$$$$  | $$  
| $$__/      \  $$/   | $$__/    \____  $$      | $$__  $$  | $$  
| $$          | $$    | $$       /$$  \ $$      | $$  | $$  | $$  
| $$$$$$$$    | $$    | $$$$$$$$|  $$$$$$/      | $$  | $$ /$$$$$$
|________/    |__/    |________/ \______/       |__/  |__/|______/
                                                                  
                                                                  
"""

print(ascii_art)
log.info("EYES AI running...")

# ========= ANA =========

if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    process_loop()
