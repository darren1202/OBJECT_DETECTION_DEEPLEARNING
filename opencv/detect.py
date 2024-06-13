

""" runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import cv2
import os
import time
import json
from azure.iot.device import IoTHubDeviceClient, Message
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# Replace with your actual connection string
CONNECTION_STRING = ""HostName=FYP2.azure-devices.net;DeviceId=192.168.0.149;SharedAccessKey=Iw30t9g6775+2/fZrgBjj/a9ICyS7PZU1AIoTEIH+yk="

# Initialize the IoT Hub client
client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)

def send_to_iot_hub(message):
    try:
        client.send_message(message)
        print("Message sent to Azure IoT Hub")
    except Exception as e:
        print(f"Failed to send message to Azure IoT Hub: {e}")

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                                            default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Index of which video source to use (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--frame_rate', type=int, default=10,
                        help='Frame rate for video capture (default: 10)')
    parser.add_argument('--width', type=int, default=640,
                       help='Width of the video frame (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Height of the video frame (default: 480)')
    args = parser.parse_args()

    print(f'Loading {args.model} with {args.labels} labels.')
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.frame_rate)

    frame_count = 0
    total_inference_time = 0

    try:
              while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert the frame to RGB
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

            # Run inference
            inference_start_time = time.time()
            run_inference(interpreter, cv2_im_rgb.tobytes())
            inference_time = time.time() - inference_start_time
            total_inference_time += inference_time

            objs = get_objects(interpreter, args.threshold)[:args.top_k]
            cv2_im = append_objs_to_img(frame, inference_size, objs, labels)

            # Display inference time on the frame
           label = f'Inference time: {inference_time * 1000:.2f} ms'
            cv2_im = cv2.putText(cv2_im, label, (10, 30),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

            # Log detected objects to the console
            detected_objects = []
            for obj in objs:
                object_label = labels.get(obj.id, obj.id)
                detected_objects.append({
                    "id": obj.id,
                   "label": object_label,
                    "score": obj.score,
                    "bbox": {
                        "xmin": obj.bbox.xmin,
                        "ymin": obj.bbox.ymin,
                        "xmax": obj.bbox.xmax,
                        "ymax": obj.bbox.ymax
                    }
                })
                print(f'Detected object: {object_label} with score {obj.score:.2f}')
              
            # Prepare the message for IoT Hub
            message = {
                "frame_count": frame_count,
                "inference_time": inference_time,
                "objects_detected": detected_objects
            }

            # Convert message to JSON
            message_json = json.dumps(message)
            send_to_iot_hub(message_json)

            cv2.imshow('frame', cv2_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Update frame count
            frame_count += 1

            # Sleep to control frame rate
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 1 / args.frame_rate - elapsed_time)
            time.sleep(sleep_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        client.shutdown()

        if frame_count > 0:
            average_inference_time = total_inference_time / frame_count
            inference_rate = frame_count / total_inference_time
            print(f'Average inference time: {average_inference_time * 1000:.2f} ms')
            print(f'Inference rate: {inference_rate:.2f} FPS')

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = f'{percent}% {labels.get(obj.id, obj.id)}'

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im
  
  if __name__ == '__main__':
    main()







