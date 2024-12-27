import numpy as np




def calculate_euclidean_distance(ground_truth, prediction):
    return np.sqrt((float(ground_truth["x"]) - prediction.x)**2 + float((ground_truth["y"]) - prediction.y)**2)

def calculate_hand_distances(ground_truth_hands, predicted_hands, handedness_info):
    """
    Calculate Euclidean distances for each hand based on handedness. Apply penalty for missing hands.

    Args:
        ground_truth_hands: List of lists of ground truth landmarks (left hand, right hand).
        predicted_hands: List of lists of predicted landmarks from MediaPipe.
        handedness_info: List of handedness classifications for each predicted hand
                         (e.g., [{"category_name": "Left"}, {"category_name": "Right"}]).

    Returns:
        distances: A dictionary with distances for "Left" and "Right" hands. If no match is found, returns None for that hand.
    """
    # Initialize distances with None (indicating no match)
    distances = {"Left": None, "Right": None}

    # Map ground truth indices (0: left, 1: right)
    ground_truth_indices = {"Left": 0, "Right": 1}

    # If no hands are detected, return None (penalty will be applied later)
    if len(predicted_hands) == 0:
        distances["Left"] = distances["Right"] = [bad_score_penalty]
        return distances

    # Calculate distances for each hand
    for pred_hand, handedness in zip(predicted_hands, handedness_info):
        pred_label = handedness[0].category_name  # Get "Left" or "Right"
        if pred_label in ground_truth_indices:
            gt_index = ground_truth_indices[pred_label]
            # Calculate distances for the corresponding hand
            distances[pred_label] = [
                calculate_euclidean_distance(gt_landmark, pred_landmark)
                for gt_landmark, pred_landmark in zip(ground_truth_hands[gt_index], pred_hand)
            ]

    # If fewer hands are predicted than expected, apply penalty for missing hands
    if distances["Left"] is None:
        distances["Left"] = [bad_score_penalty]
    if distances["Right"] is None:
        distances["Right"] = [bad_score_penalty]

    return distances


def assess_accuracy(image_path, ground_truth_hands):
    # Load the image using MediaPipe's Image utility
    image = mp.Image.create_from_file(image_path)

    # Run the HandLandmarker on the image
    results = landmark_detector.detect(image)

    if not results.hand_landmarks or len(results.hand_landmarks) == 0:
        # No hands detected, apply penalty for both hands
        print(f"No hands detected for {image_path}")
        return {"Left": bad_score_penalty, "Right": bad_score_penalty}

    # Calculate distances for each hand
    distances = calculate_hand_distances(
        ground_truth_hands,
        results.hand_landmarks,
        results.handedness
    )

    # Calculate average distances, penalizing missing hands
    scores = {}
    for hand in ["Left", "Right"]:
        if distances[hand] is not None:
            scores[hand] = sum(distances[hand]) / len(distances[hand])  # Average distance
        else:
            scores[hand] = bad_score_penalty  # Penalize if no prediction

    return scores
