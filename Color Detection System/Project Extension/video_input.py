# Importing required libraries
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.messagebox
import cv2

# Reading dataset
index = ["color", "color_name", "hex", "R", "G", "B"]
color = pd.read_csv("colors.csv", names=index, header=None)
print(color.head(10))

# Creating TKinter GUI

# Setting up the main window
win = tk.Tk()
win.title('Select A Picture')
win.geometry('220x45')
win.maxsize(300, 50)

# Creating a label
name = ttk.Label(win, text='Open A File')
name.grid(row=4, column=3, pady=10, padx=5)

# Defining variables
r = g = b = 0
hovered = True
img = []
white_font = (255, 255, 255)
black_font = (0, 0, 0)
frame = []
# red_mask, green_mask, blue_mask = np.ones((3, 1), "uint8")
# res_blue, res_green, res_red = np.ones((3, 1), "uint8")
kernel = []


# Function which will execute at mouse event
def detect_color(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, hovered, frame, red_mask, green_mask, blue_mask, kernel, res_blue, res_green, res_red
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        b = frame[:, :, :1]
        g = frame[:, :, 1:2]
        r = frame[:, :, 2:]
        #
        # Set range for red color and
        # define mask
        red_lower = np.array([136, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(frame, red_lower, red_upper)

        # Set range for green color and
        # define mask
        green_lower = np.array([25, 52, 72], np.uint8)
        green_upper = np.array([102, 255, 255], np.uint8)
        green_mask = cv2.inRange(frame, green_lower, green_upper)

        # Set range for blue color and
        # define mask
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(frame, blue_lower, blue_upper)

        kernel = np.ones((5, 5), "uint8")

        # For red color
        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

        # For green color
        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernel)
        res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        # print(res_red)
        # print(type(res_red))
        # # g = frame[:, :, 1:2]
        # # r = frame[:, :, 2:]

        r = np.mean(r)
        b = np.mean(b)
        g = np.mean(g)


# Function to determine the color
def get_Color_Name(B, G, R):
    minimum = 10000
    for i in range(len(color)):
        d = abs(B - int(color.loc[i, "B"])) + abs(G - int(color.loc[i, "G"])) + abs(R - int(color.loc[i, "R"]))
        if d <= minimum:
            minimum = d
            clr_name = color.loc[i, "color_name"]
    return clr_name


# Function to open a dialog box for browsing images when button is pressed
def dialog_win():
    global img, hovered, frame, red_mask, green_mask, blue_mask, kernel, res_blue, res_green, res_red

    video = cv2.VideoCapture(askopenfilename(initialdir="/", title="Select A File", filetype=(("mp4", "*.mp4"), ("flv", "*.flv"), ("avi", "*.avi"))))

    # setting up the window
    cv2.namedWindow("Color Detection")
    cv2.setMouseCallback("Color Detection", detect_color)

    while True:

        _, frame = video.read()
        frame = cv2.resize(frame, (900, 500))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Displaying the window
        cv2.imshow("Color Detection", frame)

        if hovered:
            cv2.rectangle(frame, (30, 30), (1000, 50), (r, g, b), -1)
            text = get_Color_Name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            if r + g + b >= 500:
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, black_font, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, white_font, 1, cv2.LINE_AA)
            cv2.imshow("Color Detection", frame)

        # Checking if Esc key is pressed then breaking out of the loop & destroying all the windows
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Checking if CLOSE ('X') is clicked on the cv2 window & if true then displaying pop-up
        if cv2.getWindowProperty("Color Detection", cv2.WND_PROP_VISIBLE) < 1:
            tkinter.messagebox.showinfo("Tip", "Press Esc key to close the window")

    cv2.destroyAllWindows()


# Creating a button
Import_button = ttk.Button(win, text='Browse an image', command=dialog_win)
Import_button.grid(row=4, column=4, pady=10, padx=5)

win.mainloop()


# Check when the video ends & close
#  OR
# Play video on loop
