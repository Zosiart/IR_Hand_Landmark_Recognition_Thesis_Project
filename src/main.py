import json
import os
import asyncio
from dotenv import load_dotenv
import cv2


from src.evaluation.LandmarkMerger import LandmarkMerger
from src.evaluation.PCKCalculator import PCKCalculator
from src.evaluation.recognizer import Recognizer
from src.pipelines.PipelineManager import PipelineManager

IR_IMAGE_DIRECTORY = '../resources/evaluation_dataset/IR'
IR_GROUND_TRUTH = '../resources/evaluation_dataset/IR_annotations'

RGB_IMAGE_DIRECTORY = '../resources/evaluation_dataset/RGB'
RGB_GROUND_TRUTH = '../resources/evaluation_dataset/RGB_annotation'

def rotate_landmarks_90_counterclockwise(landmarks):
    rotated_landmarks = []
    for group in landmarks:
        rotated_group = []
        for point in group:
            x, y = point['x'], point['y']
            rotated_group.append({'x': y, 'y': 1 - x})
        rotated_landmarks.append(rotated_group)
    return rotated_landmarks

def main():

    # Load environment variables from the .env file
    load_dotenv()
    # Initialize the PCKCalculator
    pck_calculator = PCKCalculator(threshold=0.05)

    # Schedule upper and lower bound calculations concurrently
    pck_calculator.calculate_pck_bound(
            image_directory=RGB_IMAGE_DIRECTORY,
            ground_truth_directory=RGB_GROUND_TRUTH,
            bound_type="upper"
    )

    pck_calculator.calculate_pck_bound(
            image_directory=IR_IMAGE_DIRECTORY,
            ground_truth_directory=IR_GROUND_TRUTH,
            bound_type="lower"
    )

    total_pck = {"Left": [], "Right": []}

    pipeline_manager = PipelineManager('../resources/stylized-pictures')

    for file_name in os.listdir(IR_GROUND_TRUTH):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(IR_GROUND_TRUTH, file_name)

            with open(json_file_path, 'r') as f:
                ground_truth_data = json.load(f)

            for entry in ground_truth_data:

                image_name = entry["image"]
                ground_truth_landmarks = entry["landmarks"]
                ground_truth_landmarks = rotate_landmarks_90_counterclockwise(ground_truth_landmarks)
                image_path = f"{IR_IMAGE_DIRECTORY}/{image_name}"

                image = cv2.imread(image_path)
                rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rotated_path = f"../resources/stylized-pictures/rotated/{image_name}_rotated.png"
                cv2.imwrite(rotated_path, rotated)
                image_path = rotated_path


                first_pipeline_image_path = pipeline_manager.execute_first_pipeline(image_path, image_name)


                second_pipeline_image_path = pipeline_manager.execute_second_pipeline(image_path, image_name)

                recognizer = Recognizer(os.getenv("MODEL_PATH"))
                first_results = recognizer.recognize_landmarks_gestures(first_pipeline_image_path)
                second_results = recognizer.recognize_landmarks_gestures(second_pipeline_image_path)

                landmark_merger = LandmarkMerger(first_results, second_results)
                landmark_merger.merge_landmarks()

                final_pck = pck_calculator.calculate_pck(ground_truth_landmarks, landmark_merger.final_landmarks)

                total_pck["Left"].append(final_pck["Left"])
                total_pck["Right"].append(final_pck["Right"])

    final_val = PCKCalculator.calculate_final_pck(total_pck)

    print(final_val)




if __name__ == "__main__":
    main()
