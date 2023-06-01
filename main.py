import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

# Maze creation function
def create_maze(size=(10, 10)):
    shape = ((size[0] * 2) - 1, (size[1] * 2) - 1)
    maze = np.zeros(shape)

    # Generate a grid graph
    G = nx.grid_2d_graph(size[0]-1, size[1])
    maze[::2, ::2] = 1

    visited_nodes = {(0, 0)}
    stack = [(0, 0)]

    while stack:
        node = stack[-1]
        neighbors = [(node[0] + dx, node[1] + dy) for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
        neighbors = [(x, y) for x, y in neighbors if (x, y) in G.nodes and (x, y) not in visited_nodes]

        if neighbors:
            neighbor = neighbors[np.random.randint(len(neighbors))]
            maze[node[0] + neighbor[0], node[1] + neighbor[1]] = 1
            visited_nodes.add(neighbor)
            stack.append(neighbor)
        else:
            stack.pop()
    

    return maze


# Define the maze - replace this with your maze generator
maze = create_maze(size=(10, 10))

# MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# MediaPipe drawing utility
mp_drawing = mp.solutions.drawing_utils

# Window name
window_name = "Hand Tracking"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Read the first frame to get the size
ret, frame = cap.read()

# Initialize the maze
maze = create_maze(size=(frame.shape[0]//80, frame.shape[1]//80))

# Scale the maze up to the size of the video frame
maze = cv2.resize((maze*255).astype('uint8'), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

# Initialize the game state
game_started = False
hit_count = 0
last_check = time.time()
red_frame_start_time = -1  # Time when the frame started to turn red
index_finger_track = []  # Initialize the finger track
score_displayed = False  # Flag to indicate if score is displayed

while cap.isOpened():
    # Capture the frame
    ret, frame = cap.read()
    if not ret:
        print("Unable to acquire webcam feed.")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)

    # Draw the hand annotations
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x, index_finger_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            # Check if the index finger is at the start position
            if not game_started and index_finger_x < 30 and index_finger_y < 30:
                game_started = True
                hit_count = 0  # Reset hit count
                index_finger_track = []  # Reset the finger track
                maze = create_maze(size=(frame.shape[0]//80, frame.shape[1]//80))  # Create a new maze
                maze = cv2.resize((maze*255).astype('uint8'), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)  # Scale up the maze
                score_displayed = False  # Reset the score display flag

            if game_started:
                index_finger_track.append((index_finger_x, index_finger_y))

                # Check if the index finger is touching a wall
                if 0 <= index_finger_y < maze.shape[0] and 0 <= index_finger_x < maze.shape[1]:
                    if maze[index_finger_y, index_finger_x] == 0 and time.time() - last_check > 0.3:
                        hit_count += 1
                        last_check = time.time()
                        red_frame_start_time = time.time()

                # Check if the index finger is at the end position
                if index_finger_x > frame.shape[1] - 40 and index_finger_y < 40:
                    score = max(0, 100 - hit_count)
                    print(f"You reached the end! Your score is {score}.")
                    game_started = False
                    score_displayed = True

    # Draw the track of the index finger tip
    for i in range(1, len(index_finger_track)):
        cv2.line(frame, index_finger_track[i-1], index_finger_track[i], (0, 0, 255), 2)

    # If the frame should be red, add a red overlay
    if time.time() - red_frame_start_time < 0.2:
        frame = cv2.addWeighted(frame, 0.5, np.full_like(frame, (0, 0, 255)), 0.5, 0)

    # Display the maze over the frame
    #frame = cv2.addWeighted(frame, 0.8, cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR), 0.2, 0)
    # Display the maze over the frame
    yellow_color = (0, 128, 128)  # 薄い黄色 (BGR形式)
    yellow_maze = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    yellow_maze[np.where((yellow_maze == [255, 255, 255]).all(axis=2))] = yellow_color
    frame = cv2.addWeighted(frame, 0.8, yellow_maze, 0.2, 0)


    # Draw the start and end points
    cv2.putText(frame, "S", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, "G", (maze.shape[1] - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
    #cv2.circle(frame, (15, 15), 10, (0, 255, 0), -1)  # Start is green
    #cv2.circle(frame, (frame.shape[1]-15, 15), 10, (0, 0, 255), -1)  # End is red
    

    # Display the score if the game has ended
    if score_displayed:
        cv2.putText(frame, f"Your score is {score}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow(window_name, frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Destroy the window
cv2.destroyAllWindows()
