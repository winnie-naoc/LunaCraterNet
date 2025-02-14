import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os


class YOLOv11:
    """YOLOv11 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
        """
        Initialize the YOLOv11 model for object detection.

        Args:
            onnx_model (str): Path to the ONNX model.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression (NMS).
        """
        # Load the ONNX model and specify the providers (CUDA and CPU)
        self.onnx_model = onnx_model
        self.session = ort.InferenceSession(self.onnx_model,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        self.input_width = self.model_inputs[0].shape[2]
        self.input_height = self.model_inputs[0].shape[3]

        # Load the class names from the COCO dataset (80 classes)
        self.classes = ['keng']
        # Generate a color palette for the classes (random colors for each class)
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, image):
        """
        Preprocess the input image before inference.

        Args:
            image: The input image to be preprocessed.

        Returns:
            Preprocessed image data in the expected format for the model.
        """
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape expected by the model
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing by 255.0 to bring pixel values between 0 and 1
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape (batch size 1)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Post-process the model's output to get final bounding boxes, scores, and class IDs.

        Args:
            input_image: The original input image.
            output: The model's raw output.

        Returns:
            The input image with bounding boxes and labels drawn.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array (number of detections)
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        img_height, img_width = input_image.shape[:2]
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the bounding box and label on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the image with detections drawn
        return input_image

    def main(self, input_image):
        """
        Run the full pipeline: preprocess, inference, postprocess.

        Args:
            input_image: The image to run detection on.

        Returns:
            The processed image with detections.
        """
        # Preprocess the image
        img_data = self.preprocess(input_image)

        # Run inference on the preprocessed image
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        # Postprocess the output and return the final image
        return self.postprocess(input_image, outputs)


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./runs/train/exp2/weights/best.onnx", help="Input your ONNX model.")
    parser.add_argument("--image", type=str, default="/DATA01/yolov11/test_images/D117_15_20_628__3_1.png", help="Path to your image file.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU threshold for NMS.")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image.")
    args = parser.parse_args()

    # Initialize the YOLOv11 model with the ONNX model and thresholds
    model = YOLOv11(args.model, args.conf_thres, args.iou_thres)

    # Load the input image
    image = cv2.imread(args.image)

    # Run object detection on the input image
    output_image = model.main(image)

    # Save the processed image
    cv2.imwrite(args.output, output_image)
