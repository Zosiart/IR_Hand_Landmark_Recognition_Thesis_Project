import cv2

# Open the video file
cap = cv2.VideoCapture('../../resources/hand-pictures/before-water.mp4')

# Get video properties (frame width, height, and FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the contrast and brightness values
contrast = 1.0  # Try values between 1 and 3 for strong contrast
brightness = -90  # Try values between -100 and 50 for moderate brightness

# Create a VideoWriter object to save the modified video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('../../resources/stylized-pictures/before-water-adjusted.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Adjust contrast and brightness using addWeighted
    adjusted_frame = cv2.addWeighted(frame, contrast, frame, 0, brightness)

    # Write the frame to the output video
    out.write(adjusted_frame)

# Release everything when done
cap.release()
out.release()

# Optionally, display the video during processing (can be commented out)
# cv2.imshow('Adjusted Video', adjusted_frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# cv2.destroyAllWindows()
