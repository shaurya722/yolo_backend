from io import BytesIO
import tempfile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image, ImageDraw, ImageFont
from api.models import GeneratedVideo, YoloImage
from ultralytics import YOLO
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
from django.core.files.base import ContentFile
import cv2
import os


# Load the YOLO model
model_path = '/home/shaurya/Downloads/best.pt'
model = YOLO(model_path)



# Dictionary to store generated colors for each class index
class_color_map = {
    'motor_bike': (255, 0, 0),  # Red for bikes
    'with_helmet': (255, 255, 0),  # Yellow for helmets
    'without_helmet': (0, 0, 255),  # Blue for without helmet
}

class YoloPredict(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "Image file is required"}, status=status.HTTP_400_BAD_REQUEST)
    
            # Open image and make predictions
        try:
            image = Image.open(image_file)
            # output = model(image)  # Perform inference
            # results = Detections.from_ultralytics(output[0])
            results = model(image)  # Perform inference
            # annotator = BoxAnnotator()
            # annotated_image = annotator.annotate(scene=image, detections=results)

            # Extract the results (boxes, labels, and confidence scores)
            boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box coordinates
            labels = results[0].boxes.cls.cpu().numpy()  # Class labels
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

            class_names = model.names 
            print(class_names)
            print(boxes)
            print('labels :',labels)
            print(type(labels))
            print(labels[1])
            print(confidences)
            # Convert the image to a NumPy array (RGB)
            annotated_image = np.array(image)

            # Initialize ImageDraw object to draw on the PIL Image
            annotated_image = Image.fromarray(annotated_image)  # Convert back to PIL Image for drawing
            draw = ImageDraw.Draw(annotated_image)

            # Load a font (optional: for adding text to boxes)
            font = ImageFont.load_default()  # Ensure it's for PIL Image, not NumPy array

            # Loop through the detections and annotate the image
            for i in range(len(boxes)):
                x1, y1, w, h = boxes[i]
                x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
                label_idx = int(labels[i])  # Get the class index
                label_name = class_names[label_idx]  # Get the class name
                confidence = confidences[i]
                text = f"{label_name} {confidence:.2f}"  # Use class name instead of numeric index

                 # Assign a unique color to the label
                if label_name in class_color_map:
                    color = class_color_map[label_name]  # Get the predefined color for the class
                else:
                    color = (255, 255, 255) 
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label and confidence
                draw.text((x1, y1), text, fill=color, font=font)
            
            img_byte_arr = BytesIO()
            annotated_image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

                    # Convert to Django InMemoryUploadedFile for saving
            img_file = InMemoryUploadedFile(
                img_byte_arr, None, "annotated_image.png", 'image/png', img_byte_arr.getbuffer().nbytes, None
            )

            img = YoloImage.objects.create(img=img_file)
           
            return Response({'image_url': img.img.url}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class YoloPredictBikeDetectionVideo(APIView):
    def post(self, request, *args, **kwargs):
        video_file = request.FILES.get('video')
        if not video_file:
            return Response({"error": "Video file is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Save the uploaded video to a temporary file
            video_data = video_file.read()
            temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_file.write(video_data)
            temp_video_file.close()  # Close the file to allow OpenCV to access it

            # Open the video file using OpenCV
            video_capture = cv2.VideoCapture(temp_video_file.name)
            if not video_capture.isOpened():
                os.remove(temp_video_file.name)  # Clean up the temporary file
                return Response({"error": "Failed to open video file."}, status=status.HTTP_400_BAD_REQUEST)

            # Get video properties
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            # Prepare for saving the video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
            output_video_path = "/tmp/input_video.mp4"  # Output video path
            out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break  # End of video

                # Convert frame to PIL image for annotation
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Perform inference
                results = model(pil_image)  # Perform inference

                # Extract the results (boxes, labels, and confidence scores)
                boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box coordinates
                labels = results[0].boxes.cls.cpu().numpy()  # Class labels
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

                class_names = model.names  # Get class names

                # Convert the PIL image back to NumPy array (RGB)
                annotated_image = np.array(pil_image)
                annotated_image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(annotated_image)

                # Load a font for adding text
                font = ImageFont.load_default()

                # Loop through the detections and annotate the image
                for i in range(len(boxes)):
                    x1, y1, w, h = boxes[i]
                    x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)
                    label_idx = int(labels[i])  # Get the class index
                    label_name = class_names[label_idx]  # Get the class name
                    confidence = confidences[i]
                    text = f"{label_name} {confidence:.2f}"

                    # Assign color based on the class
                    if label_name in class_color_map:
                        color = class_color_map[label_name]
                    else:
                        color = (255, 255, 255)  # Default to white

                    # Draw bounding box and label
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1, y1), text, fill=color, font=font)

                # Convert annotated image back to BGR for OpenCV
                annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

                # Write the annotated frame to the output video
                out_video.write(annotated_frame)

            # Release video capture and writer objects
            video_capture.release()
            out_video.release()

            # Clean up the temporary file
            os.remove(temp_video_file.name)

            # Create an InMemoryUploadedFile for the annotated video
            with open(output_video_path, 'rb') as f:
                video_content = ContentFile(f.read())
                video_file = InMemoryUploadedFile(video_content, None, "annotated_video.mp4", 'video/mp4', video_content.size, None)

            # Save the video file to the model
            video_instance = GeneratedVideo.objects.create(video=video_file)

            return Response({'video_url': video_instance.video.url}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

