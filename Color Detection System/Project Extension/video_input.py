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
name = ttk.Label(win, text='Open A Media')
name.grid(row=4, column=3, pady=10, padx=5)

# Defining global variables
r = g = b = 0
clicked = False
frame = []
frame_counter = 0

# Defining local variables
white_font = (255, 255, 255)
black_font = (0, 0, 0)


# Function which will execute at mouse event
def detect_color(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, clicked, frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   #COLOR_RGB2BGR
        clicked = True

        b = frame[:, :, :1]
        g = frame[:, :, 1:2]
        r = frame[:, :, 2:]

        r = np.mean(r)
        b = np.mean(b)
        g = np.mean(g)


# Function to determine the color
def get_Color_Name(R, G, B):
    minimum = 10000
    for i in range(len(color)):
        d = abs(R - int(color.loc[i, "R"])) + abs(G - int(color.loc[i, "G"])) + abs(B - int(color.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            clr_name = color.loc[i, "color_name"]
    return clr_name


# Function to open a dialog box for browsing images when button is pressed
def dialog_win():
    global clicked, frame, frame_counter

    video = cv2.VideoCapture(askopenfilename(initialdir="/", title="Select A File", filetype=(("mp4", "*.mp4"), ("flv", "*.flv"), ("avi", "*.avi"))))

    # setting up the window
    cv2.namedWindow("Color Detection")
    cv2.setMouseCallback("Color Detection", detect_color)

    while True:

        _, frame = video.read()

        frame_counter = frame_counter + 1

        #  Playing video on loop
        if frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0  # Or whatever as long as it is the same as next line
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Displaying the window
        cv2.imshow("Color Detection", frame)

        if clicked:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (30, 30), (1000, 50), (r, g, b), -1)
            text = get_Color_Name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            if r + g + b >= 500:
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, black_font, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, white_font, 1, cv2.LINE_AA)
            cv2.imshow("Color Detection", frame)

        # Checking if Esc key is pressed then breaking out of the loop & destroying all the windows
        if cv2.waitKey(1) & 0xFF == 27:

            # Setting to false so that previous rectangle is not displayed on a next video the user selects
            clicked = False
            break

        # Checking if CLOSE ('X') is clicked on the cv2 window & if true then displaying pop-up
        if cv2.getWindowProperty("Color Detection", cv2.WND_PROP_VISIBLE) < 1:
            tkinter.messagebox.showinfo("Tip", "Press Esc key to close the window")

    cv2.destroyAllWindows()


# Creating a button
Import_button = ttk.Button(win, text='Browse a Video', command=dialog_win)
Import_button.grid(row=4, column=4, pady=10, padx=5)

win.mainloop()


# Check when the video ends & close
#  OR
# Play video on loop
