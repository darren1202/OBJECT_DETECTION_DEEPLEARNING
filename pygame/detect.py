
""" run the detector in a Pygame camera stream."""

import argparse
import os
import sys
import time
import threading
from queue import Queue

import pygame
import pygame.camera
from pygame.locals import *

from azure.iot.device import IoTHubDeviceClient, Message

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

# Azure IoT Hub connection string
CONNECTION_STRING = "HostName=FYP2.azure-devices.net;DeviceId=192.168.0.149;SharedAccessKey=Iw30t9g6775+2/fZrgBjj/a9ICyS7PZU1AIoTEIH+yk="

def send_to_azure(result, labels):
    if result.id in labels:
        message_text = f"Detected: {labels[result.id]} with {result.score * 100:.2f}% confidence"
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        message = Message(message_text)
        client.send_message(message)
        client.disconnect()
        print(f"Sent to Azure: {message_text}")
    else:
        print(f"Warning: Detected object ID {result.id} not found in labels")

def capture_frames(camera, frame_queue, inference_size):
    while True:
        frame = camera.get_image()
        frame = pygame.transform.scale(frame, inference_size)
        frame_queue.put(frame)
        time.sleep(0.033)  # Capture at approximately 30 FPS

def process_frames(frame_queue, display_queue, interpreter, args, scale_x, scale_y, labels):
    red = pygame.Color(255, 0, 0)
    font = pygame.font.SysFont('Arial', 20)
    last_time = time.monotonic()

    while True:
        if not frame_queue.empty():
           frame = frame_queue.get()
            start_time = time.monotonic()
            run_inference(interpreter, frame.get_buffer().raw)
            results = get_objects(interpreter, args.threshold)[:args.top_k]
            stop_time = time.monotonic()
            inference_ms = (stop_time - start_time) * 1000.0
            fps = 1.0 / (stop_time - last_time)
            last_time = stop_time

            print(f"Detected {len(results)} objects")
            for result in results:
                print(f"Object ID: {result.id}, Score: {result.score}")
                bbox = result.bbox.scale(scale_x, scale_y)
                rect = pygame.Rect(bbox.xmin, bbox.ymin, bbox.width, bbox.height)
                pygame.draw.rect(frame, red, rect, 1)
                if result.id in labels:
                    label = f'{result.score * 100:.0f}% {labels[result.id]}'
                else:
                    label = f'{result.score * 100:.0f}% Unknown ID: {result.id}'
                    print(f"Warning: Detected object ID {result.id} not found in labels")
            text = font.render(label, True, red)
                frame.blit(text, (rect.x, rect.y - 20))  # Adjust position to place text above the rectangle
                send_to_azure(result, labels)

            annotate_text = f'Inference: {inference_ms:.2f}ms FPS: {fps:.1f}'
            annotate_text_surface = font.render(annotate_text, True, red)
            frame.blit(annotate_text_surface, (10, 10))
            display_queue.put(frame)

def main():
    cam_w, cam_h = 640, 480
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines() if l.strip() and not l.startswith('#'))
        labels = {int(k): v for k, v in pairs}

    print(f'Loading {args.model} with {args.labels} labels.')

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)

    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    inference_size = input_size(interpreter)

    camera = None
    for cam in camlist:
        try:
            camera = pygame.camera.Camera(cam, (cam_w, cam_h))
            camera.start()
            print(f'{str(cam)} opened')
            break
        except SystemError as e:
            print(f'Failed to open {str(cam)}: {str(e)}')
            camera = None
    if not camera:
        sys.stderr.write("\nERROR: Unable to open a camera.\n")
        sys.exit(1)

    try:
        display = pygame.display.set_mode((cam_w, cam_h), pygame.RESIZABLE)
    except pygame.error as e:
        sys.stderr.write(
            "\nERROR: Unable to open a display window. Make sure a monitor is "
            "connected and the DISPLAY environment variable is set. Example: \n"
            ">export DISPLAY=\":0\" \n")
        raise e

    frame_queue = Queue(maxsize=2)  # Limit the queue size to avoid memory issues
    display_queue = Queue(maxsize=2)
       
    scale_x, scale_y = cam_w / inference_size[0], cam_h / inference_size[1]

    capture_thread = threading.Thread(target=capture_frames, args=(camera, frame_queue, inference_size))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, display_queue, interpreter, args, scale_x, scale_y, labels))

    capture_thread.start()
    process_thread.start()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    raise KeyboardInterrupt

            if not display_queue.empty():
                frame = display_queue.get()
                frame = pygame.transform.scale(frame, (cam_w, cam_h))  # Scale to fill the display
                display.blit(frame, (0, 0))
                pygame.display.flip()
    except KeyboardInterrupt:
      print("Exiting...")
    finally:
        camera.stop()
        capture_thread.join()
        process_thread.join()

if __name__ == '__main__':
    main()




    

