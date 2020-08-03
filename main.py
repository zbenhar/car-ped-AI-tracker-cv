
import cv2

# Load pre-trained data on car rear ends (haar cascade algorithm)
car_tracker = cv2.CascadeClassifier('car_detector.xml')
pedestrian_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Get video footage for system recognition
video = cv2.VideoCapture('pedestrians-cars-video.mp4')

# Now, iterate over the frames of video
while True:
    # Read current frame
    read_successful, frame = video.read()

    if read_successful:
        # Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)  # no need for scale factor & minNeighbors; simple program
    # Detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    # Draw rectangles around cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Draw rectangles around pedestrians
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Display image with cars and pedestrians spotted
    cv2.imshow('', frame)

    # wait for a key press
    key = cv2.waitKey(1)

    # stop if 'Q' or 'q' is pressed
    if key == 81 or key == 113:  # using ascii values
        break

# Release VideoCapture object
video.release()


print('Everything looks cool')