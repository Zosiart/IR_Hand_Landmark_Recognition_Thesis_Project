import cv2
import numpy as np

from src.colorization.Zhang import create_colorized_pictures


class PipelineManager:
    """
    Class to manage and execute transformation pipelines.
    """

    def __init__(self, base_output_path="../../resources/stylized-pictures"):
        """
        Initialize the PipelineManager with a base output directory.
        :param base_output_path: Base path to save transformed images.
        """
        self.base_output_path = base_output_path

    def remove_temperature_boxes(self, image_path, image_name):
        """
        Remove temperature boxes (red and green regions) from an image.
        :param image_path: Path to the input image.
        :param image_name: Name of the image for output file naming.
        :return: Path to the processed image with no boxes.
        """
        image_cv = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])

        mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_combined = mask_red1 | mask_red2 | mask_green

        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_combined, kernel, iterations=1)
        image_no_boxes = cv2.inpaint(image_cv, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        smoothed_image = cv2.GaussianBlur(image_no_boxes, (5, 5), 0)

        output_path = f"{self.base_output_path}/no_boxes/{image_name}_no_boxes.png"
        cv2.imwrite(output_path, smoothed_image)
        return output_path

    def execute_first_pipeline(self, image_path, image_name, colorization_model='siggraph17'):
        """
        Execute the first transformation pipeline.
        :param image_path: Path to the input image.
        :param image_name: Name of the image for output file naming.
        :param colorization_model: Model to use for colorization ('siggraph17' or 'eccv16').
        :return: Path to the transformed image.
        """
        no_boxes_image_path = self.remove_temperature_boxes(image_path, image_name)
        inverted_image = cv2.bitwise_not(cv2.imread(no_boxes_image_path))
        inverted_image_path = f"{self.base_output_path}/inverted/{image_name}_inverted.png"
        cv2.imwrite(inverted_image_path, inverted_image)

        if colorization_model == 'eccv16':
            return self.colorize_zhang_eccv16(inverted_image_path, image_name)
        return self.colorize_zhang_siggraph17(inverted_image_path, image_name)

    def execute_second_pipeline(self, image_path, image_name):
        """
        Execute the second transformation pipeline with no visible fingers.
        :param image_path: Path to the input image.
        :param image_name: Name of the image for output file naming.
        :return: Path to the transformed image.
        """
        no_boxes_image_path = self.remove_temperature_boxes(image_path, image_name)
        image = cv2.imread(no_boxes_image_path, cv2.IMREAD_GRAYSCALE)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        clahe_output_path = f"{self.base_output_path}/clahe/{image_name}_clahe.jpg"
        cv2.imwrite(clahe_output_path, enhanced_image)

        sharpened_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        sharpened_image_path = f"{self.base_output_path}/not_detected/sharpened/{image_name}_sharpened.png"
        cv2.imwrite(sharpened_image_path, sharpened_image)

        return sharpened_image_path

    @staticmethod
    def colorize_zhang_siggraph17(image_path, image_name):
        """
        Colorize the image using Zhang's SIGGRAPH17 model.
        :param image_path: Path to the input image.
        :param image_name: Name of the image for output file naming.
        :return: Path to the colorized image.
        """
        print(f"Colorizing {image_name} using Zhang SIGGRAPH17 model...")
        create_colorized_pictures(model='siggraph17', img_path=image_path, save_prefix=image_name)
        return f'../../resources/stylized-pictures/siggraph17/{image_name}_siggraph17.png'

    @staticmethod
    def colorize_zhang_eccv16(image_path, image_name):
        """
        Colorize the image using Zhang's ECCV16 model.
        :param image_path: Path to the input image.
        :param image_name: Name of the image for output file naming.
        :return: Path to the colorized image.
        """
        print(f"Colorizing {image_name} using Zhang ECCV16 model...")
        create_colorized_pictures(model='eccv16', img_path=image_path, save_prefix=image_name)
        return f'../../resources/stylized-pictures/eccv16/{image_name}_eccv16.png'
