import cv2
import requests
import time
import csv
import os
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# ----------- FUNCTION TO GET GPS LOCATION -----------
def get_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5).json()
        loc = res.get("loc", None)
        if loc:
            lat, lon = loc.split(",")
            return float(lat), float(lon)
    except:
        pass
    return None, None


LAT, LON = get_location()
if LAT is None:
    LAT, LON = 19.0760, 72.8777  # fallback

start_time = time.time()
DURATION = 40

last_detected_frame = -100
COOLDOWN = 30

CSV_FILE = "pothole_log.csv"

# ✅ NEW: Confidence Threshold (60%)
CONFIDENCE_THRESHOLD = 0.6

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "frame", "class", "confidence", "latitude", "longitude"])


client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="fcp24rpp22CI6vRzdeaC"
)

source = WebcamSource(resolution=(1280, 720))

config = StreamConfig(
    stream_output=["output_image"],
    data_output=["count_objects", "predictions"],
    processing_timeout=3600,
    requested_plan="webrtc-gpu-medium",
    requested_region="us"
)

session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize",
    workspace="manish3500",
    image_input="image",
    config=config
)

# ----------- FRAME HANDLER -----------
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)

    if time.time() - start_time > DURATION:
        print("⏹️ Stopping after 40 seconds...")
        session.close()
        return

    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()


# ----------- DETECTION HANDLER -----------
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    global last_detected_frame

    pred_block = data.get("predictions", {})
    predictions = pred_block.get("predictions", [])

    if not isinstance(predictions, list):
        return

    # ✅ FILTER predictions based on confidence
    filtered_predictions = [
        p for p in predictions
        if isinstance(p, dict) and p.get("confidence", 0) > CONFIDENCE_THRESHOLD
    ]

    for pred in filtered_predictions:
        class_name = str(pred.get("class", "")).lower()
        confidence = pred.get("confidence", 0)

        # DEBUG print (only high confidence now)
        print(f"Detected (HIGH CONF): {class_name} ({confidence})")

        if "pothole" in class_name:
            if metadata.frame_id - last_detected_frame > COOLDOWN:
                last_detected_frame = metadata.frame_id

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # ✅ SAVE TO CSV
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp,
                        metadata.frame_id,
                        class_name,
                        confidence,
                        LAT,
                        LON
                    ])

                print("\n🚧 POTHOLE DETECTED (HIGH CONFIDENCE) + SAVED ✅")
                print(f"Frame: {metadata.frame_id}")
                print(f"Confidence: {confidence}")
                print(f"GPS: {LAT}, {LON}\n")


session.run()
cv2.destroyAllWindows()