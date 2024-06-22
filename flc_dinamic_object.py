from controller import Robot
import cv2
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import csv

# create the Robot instance.
robot = Robot()

def run_robot(robot):
    timestep = int(robot.getBasicTimeStep())
    
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    ds = []
    dsNames = ['left infrared sensor','front left infrared sensor', 'front infrared sensor',
                'front right infrared sensor', 'right infrared sensor']
    for i in range(5):
        ds.append(robot.getDevice(dsNames[i]))
        ds[i].enable(timestep)

    wheels = []
    wheels_names = ['left wheel motor', 'right wheel motor']
    for i in range(2):
        wheels.append(robot.getDevice(wheels_names[i]))
        wheels[i].setPosition(float('inf'))
        wheels[i].setVelocity(0.0)
    
    # Define fuzzy variables
    error = ctrl.Antecedent(np.arange(-376, 377, 1), 'error')
    delta_error = ctrl.Antecedent(np.arange(-376, 377, 1), 'delta_error')
    left_speed = ctrl.Consequent(np.arange(-12, 19, 1), 'left_speed')
    right_speed = ctrl.Consequent(np.arange(-12, 19, 1), 'right_speed')
    
    # Define fuzzy sets for error
    error['negative'] = fuzz.trapmf(error.universe, [-376, -376, -100, 0])
    error['zero'] = fuzz.trimf(error.universe, [-50, 0, 50])
    error['positive'] = fuzz.trapmf(error.universe, [0, 100, 376, 376])
    
    # Define fuzzy sets for delta_error
    delta_error['negative'] = fuzz.trapmf(delta_error.universe, [-376, -376, -80, 0])
    delta_error['zero'] = fuzz.trimf(delta_error.universe, [-30, 0, 30])
    delta_error['positive'] = fuzz.trapmf(delta_error.universe, [0, 80, 376, 376])

    # Define fuzzy sets for speeds
    left_speed['n_med'] = fuzz.trimf(left_speed.universe, [-12, -12, -4])
    left_speed['n_low'] = fuzz.trimf(left_speed.universe, [-6, 0, 0])
    left_speed['slow'] = fuzz.trimf(left_speed.universe, [0, 0, 6])
    left_speed['medium'] = fuzz.trimf(left_speed.universe, [4, 9, 14])
    left_speed['fast'] = fuzz.trimf(left_speed.universe, [10, 18, 18])
    
    right_speed['n_med'] = fuzz.trimf(right_speed.universe, [-12, -12, -4])
    right_speed['n_low'] = fuzz.trimf(right_speed.universe, [-6, 0, 0])
    right_speed['slow'] = fuzz.trimf(right_speed.universe, [0, 0, 6])
    right_speed['medium'] = fuzz.trimf(right_speed.universe, [4, 9, 14])
    right_speed['fast'] = fuzz.trimf(right_speed.universe, [10, 18, 18])
    
    # Define fuzzy rules
    rule1 = ctrl.Rule(error['negative'] & delta_error['negative'], (left_speed['fast'], right_speed['n_med']))
    rule2 = ctrl.Rule(error['negative'] & delta_error['zero'], (left_speed['fast'], right_speed['n_low']))
    rule3 = ctrl.Rule(error['negative'] & delta_error['positive'], (left_speed['fast'], right_speed['slow']))

    rule4 = ctrl.Rule(error['zero'] & delta_error['negative'], (left_speed['fast'], right_speed['medium']))
    rule5 = ctrl.Rule(error['zero'] & delta_error['zero'], (left_speed['medium'], right_speed['medium']))
    rule6 = ctrl.Rule(error['zero'] & delta_error['positive'], (left_speed['medium'], right_speed['fast']))

    rule7 = ctrl.Rule(error['positive'] & delta_error['negative'], (left_speed['slow'], right_speed['fast']))
    rule8 = ctrl.Rule(error['positive'] & delta_error['zero'], (left_speed['n_low'], right_speed['fast']))
    rule9 = ctrl.Rule(error['positive'] & delta_error['positive'], (left_speed['n_med'], right_speed['fast']))

    # Create control system and simulation
    speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    speed = ctrl.ControlSystemSimulation(speed_ctrl)

    previous_error = 0
    
    with open('hasilSimulasi.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Object', 'Error', 'Delta Error', 'Left Motor', 'Right Motor'])

        while robot.step(timestep) != -1:
            # Get image from the camera
            img = camera.getImage()
            if img:
                # Convert the image to a format that OpenCV can use
                img = np.frombuffer(img, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Convert BGR image to HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Define range of blue color in HSV
                lower_blue = np.array([110, 50, 50])
                upper_blue = np.array([130, 255, 255])
                
                # Create masks for blue color
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours and find the largest one
                largest_contour = None
                max_area = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        largest_contour = contour
                
                if largest_contour is not None:
                    # Compute the center of the largest contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        object_x = int(M["m10"] / M["m00"])
                        object_y = int(M["m01"] / M["m00"])
                        cv2.circle(img, (object_x, object_y), 5, (0, 255, 0), -1)
                        
                        # Get the image width
                        height, width, _ = img.shape
                        img_center_x = width // 2
                        
                        # Draw the vertical center line
                        cv2.line(img, (img_center_x, 0), (img_center_x, height), (0, 255, 0), 1)
                        
                        # Calculate the error between the center of the object and the center of the image
                        error_value = img_center_x - object_x
                        
                        # Calculate delta error
                        delta_error_value = error_value - previous_error
                        previous_error = error_value
                        
                        # Set the fuzzy inputs
                        speed.input['error'] = error_value
                        speed.input['delta_error'] = delta_error_value
                        
                        # Compute the fuzzy output
                        speed.compute()
                        
                        # Get the fuzzy output for wheel speeds
                        l_speed = speed.output['left_speed']
                        r_speed = speed.output['right_speed']
                        
                        current_time = robot.getTime()
                        writer.writerow([current_time, object_x, error_value, delta_error_value, l_speed, r_speed])
                        
                        if ds[2].getValue() >= 250:
                            l_speed = 0
                            r_speed = 0
                else:
                    # If no object is detected, use obstacle avoidance logic
                    if ds[0].getValue() >= 200:
                        l_speed = 4
                        r_speed = -4
                    elif ds[1].getValue() >= 200:
                        l_speed = 4
                        r_speed = -4
                    elif ds[3].getValue() >= 200:
                        l_speed = -4
                        r_speed = 4
                    elif ds[4].getValue() >= 200:
                        l_speed = -4
                        r_speed = 4
                    else:
                        l_speed = 8
                        r_speed = 8
                
                img_resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5) # Resize the image to half its original size
                
                # Display the image with the center line and detected object
                cv2.imshow("Camera View", img_resized)
                cv2.waitKey(1)
            
            # Set the wheel speeds
            wheels[0].setVelocity(l_speed)
            wheels[1].setVelocity(r_speed)

if __name__ == "__main__":
    run_robot(robot)