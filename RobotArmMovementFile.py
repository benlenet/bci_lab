import xarm
import time
import keyboard
import socket

# Setup xArm Controller
arm = xarm.Controller('USB')
# Setup MATLAB file

EEGThought = None

# Servo 1 range 0 - 650
# Servo 2 range 0 - 1000
# Servo 3 range 10 - 990
# Servo 4 range


# Initialize all six servos
speed = 1500
neutral = 500


def Initialize():
    arm.servoOff()
    arm.setPosition(1, neutral, speed, True)
    arm.setPosition(2, neutral, speed, True)
    arm.setPosition(3, neutral, speed, True)
    arm.setPosition(4, neutral, speed, True)
    arm.setPosition(5, neutral, speed, True)
    arm.setPosition(6, neutral, speed, True)

min_position = 100   # Set the minimum valid position for the servo
max_position_gripper = 650  # Set the maximum valid position for the gripper servo
max_position = 900

currentKey = None
currentThought = None


# Set to manual for keyboard operation
control = "manual"

if control == "manual":
    speed = 1500
    print("The arm uses the following controls:")
    print("Q: Closes gripper")
    print("W: Open gripper")
    print("E: Twists gripper left")
    print("R: Twists gripper right")
    print("T: Moves arm up slightly")
    print("Y: Moves arm down slightly")
    print("U: Moves arm up more")
    print("I: Moves arm down more")
    print("O: Moves arm up a lot")
    print("P: Moves arm down a lot")
    print("J: Rotates arm left")
    print("K: Rotates arm right")
    print("S: Stops movement")
    print("+: Increases speed")
    print("-: Decreases speed")
    print("A: Return to Neutral Position")
    print("esc: Exits program")
else:
    speed = 2000
    print("Working off of the EEG headset")

    # Set up a UDP socket on port 6000 (match MATLAB's port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 60000))


print("Initializing...")
Initialize()
print("Initialization complete")

while True:
    if (control == "manual"):
        if keyboard.is_pressed("Q") and currentKey != "Q":
            currentKey = "Q"
            arm.setPosition(1, max_position_gripper, speed, False)
            print("Closing the gripper")


        if keyboard.is_pressed("W") and currentKey != "W":
            currentKey = "W"
            arm.setPosition(1, min_position, speed, False)
            print("Opening the gripper")


        if keyboard.is_pressed("E") and currentKey != "E":
            currentKey = "E"
            arm.setPosition(2, max_position, speed, False)
            print("Twisting left")


        if keyboard.is_pressed("R") and currentKey != "R":
            currentKey = "R"
            arm.setPosition(2, min_position, speed, False)
            print("Twisting right")


        if keyboard.is_pressed("T") and currentKey != "T":
            currentKey = "T"
            arm.setPosition(3, max_position, speed, False)
            print("Tilting up")


        if keyboard.is_pressed("Y") and currentKey != "Y":
            currentKey = "Y"
            arm.setPosition(3, min_position, speed, False)
            print("Tilting down")


        if keyboard.is_pressed("U") and currentKey != "U":
            currentKey = "U"
            arm.setPosition(4, min_position, speed, False)
            print("Lifting up")


        if keyboard.is_pressed("I") and currentKey != "I":
            currentKey = "I"
            arm.setPosition(4, max_position, speed, False)
            print("Lifting down")


        if keyboard.is_pressed("O") and currentKey != "O":
            currentKey = "O"
            arm.setPosition(5, max_position, speed, False)
            print("Heaving up")


        if keyboard.is_pressed("P") and currentKey != "P":
            currentKey = "P"
            arm.setPosition(5, min_position, speed, False)
            print("Heaving down")


        if keyboard.is_pressed("J") and currentKey != "J":
            currentKey = "J"
            arm.setPosition(6, max_position, speed, False)
            print("Rotating left")


        if keyboard.is_pressed("K") and currentKey != "K":
            currentKey = "K"
            arm.setPosition(6, min_position, speed, False)
            print("Rotating right")


        if keyboard.is_pressed("+"):
            speed -= 100
            time.sleep(0.1)
            print("Current speed is: ", speed)


        if (keyboard.is_pressed("-")):
            speed += 100
            time.sleep(0.1)
            print("Current speed is: ", speed)


        if keyboard.is_pressed("S"):
            currentKey = None
            arm.servoOff()
            CurrentPosition = arm.getPosition(1, False)
            arm.setPosition(1, CurrentPosition, speed, False)
            print("Stopping gripper")


            CurrentPosition = arm.getPosition(2, False)
            arm.setPosition(2, CurrentPosition, speed, False)
            print("Stopping twister")


            CurrentPosition = arm.getPosition(3, False)
            arm.setPosition(3, CurrentPosition, speed, False)
            print("Stopping movement up")


            CurrentPosition = arm.getPosition(4, False)
            arm.servoOff(4)
            arm.setPosition(4, CurrentPosition, 100, True)
            print("Stopping more movement up")


            CurrentPosition = arm.getPosition(5, False)
            arm.servoOff(5)
            arm.setPosition(5, CurrentPosition, 100, True)
            print("Stopping most movement up")


            CurrentPosition = arm.getPosition(6, False)
            arm.setPosition(6, CurrentPosition, speed, False)
            print("Stopped rotation")


            print("Arm stopped moving")
        if (keyboard.is_pressed("A")) and currentKey != "A":
            arm.setPosition(1, neutral, speed, True)
            arm.setPosition(2, neutral, speed, True)
            arm.setPosition(3, neutral, speed, True)
            arm.setPosition(4, neutral, speed, True)
            arm.setPosition(5, neutral, speed, True)
            arm.setPosition(6, neutral, speed, True)
    
        if (keyboard.is_pressed("esc")):
            currentKey = all
            print("Exiting program. Please do not press anything")
            speed = 1000
            Initialize()
            print("Goodbye")
            arm.servoOff()
            break

    else:
        data, addr = sock.recvfrom(1024)
        EEGThought = data.decode("utf-8")
        if  EEGThought == "CloseThoughts" and currentThought != "CloseThoughts":
            currentThought = "CloseThoughts"
            arm.setPosition(1, max_position_gripper, speed, False)
            print("Closing the gripper")


        if EEGThought == "OpenThoughts" and currentThought != "OpenThoughts":
            currentThought = "OpenThoughts"
            arm.setPosition(1, min_position, speed, False)
            print("Opening the gripper")


        if EEGThought == "TwistLeftThoughts" and currentThought != "TwistLeftThoughts":
            currentThought = "TwistLeftThoughts"
            arm.setPosition(2, max_position, speed, False)
            print("Twisting left")


        if EEGThought == "TwistRightThoughts" and currentThought != "TwistRightThoughts":
            currentThought = "TwistRightThoughts"
            arm.setPosition(2, min_position, speed, False)
            print("Twisting right")


        if EEGThought == "SmallUpThoughts" and currentThought != "SmallUpThoughts":
            currentThought = "SmallUpThoughts"
            arm.setPosition(3, max_position, speed, False)
            print("Tilting up")


        if EEGThought == "SmallDownThoughts" and currentThought != "SmallDownThoughts":
            currentThought = "SmallDownThoughts"
            arm.setPosition(3, min_position, speed, False)
            print("Tilting down")


        if EEGThought == "UpThoughts" and currentThought != "UpThoughts":
            currentThought = "UpThoughts"
            arm.setPosition(4, min_position, speed, False)
            print("Lifting up")


        if EEGThought == "DownThoughts" and currentThought != "DownThoughts":
            currentThought = "DownThoughts"
            arm.setPosition(4, max_position, speed, False)
            print("Lifting down")


        if EEGThought == "BigUpThoughts" and currentThought != "BigUpThoughts":
            currentThought = "BigUpThoughts"
            arm.setPosition(5, max_position, speed, False)
            print("Heaving up")


        if EEGThought == "BigDownThoughts" and currentThought != "BigDownThoughts":
            currentThought = "BigDownThoughts"
            arm.setPosition(5, min_position, speed, False)
            print("Heaving down")


        if EEGThought == "LeftThoughts" and currentThought != "LeftThoughts":
            currentThought = "LeftThoughts"
            arm.setPosition(6, max_position, speed, False)
            print("Rotating left")


        if EEGThought == "RightThoughts" and currentThought != "RightThoughts":
            currentThought = "RightThoughts"
            arm.setPosition(6, min_position, speed, False)
            print("Rotating right")


        if EEGThought == "SlowDownThoughts":
            speed += 100
            time.sleep(0.1)
            print("Current speed is: ", speed)


        if EEGThought == "SpeedUpThoughts":
            speed -= 100
            time.sleep(0.1)
            print("Current speed is: ", speed)


        if EEGThought == "StopThoughts":
            currentKey = None
            arm.servoOff()
            CurrentPosition = arm.getPosition(1, False)
            arm.setPosition(1, CurrentPosition, speed, False)
            print("Stopping gripper")


            CurrentPosition = arm.getPosition(2, False)
            arm.setPosition(2, CurrentPosition, speed, False)
            print("Stopping twister")


            CurrentPosition = arm.getPosition(3, False)
            arm.setPosition(3, CurrentPosition, speed, False)
            print("Stopping movement up")


            CurrentPosition = arm.getPosition(4, False)
            arm.servoOff(4)
            arm.setPosition(4, CurrentPosition, 100, True)
            print("Stopping more movement up")


            CurrentPosition = arm.getPosition(5, False)
            arm.servoOff(5)
            arm.setPosition(5, CurrentPosition, 100, True)
            print("Stopping most movement up")


            CurrentPosition = arm.getPosition(6, False)
            arm.setPosition(6, CurrentPosition, speed, False)
            print("Stopped rotation")


            print("Arm stopped moving")

        if (keyboard.is_pressed("esc")):
            currentKey = all
            currentThought = all
            print("Exiting program. Please do not press anything")
            speed = 1000
            Initialize()
            print("Goodbye")
            arm.servoOff()
            break