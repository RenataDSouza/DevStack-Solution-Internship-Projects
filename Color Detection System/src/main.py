# Importing required libraries
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import pandas as pd
import tkinter as tk
import tkinter.messagebox
import cv2

# Reading dataset
col_name = ["color", "color_name", "hex", "R", "G", "B"]
color = pd.read_csv("colors.csv", names=col_name, header=None)
print(color.head(10))
print("Data type of the column values: ", color.dtypes)

# Creating TKinter GUI

# Setting up the main window
win = tk.Tk()
win.title('Select A Picture')
win.geometry('220x45')
win.maxsize(300, 50)

# Creating a label
name = ttk.Label(win, text='Open A File')
name.grid(row=4, column=3, pady=10, padx=5)

# Defining global variables
r = g = b = 0
img = []

# Defining color of the font
white_font = (255, 255, 255)
black_font = (0, 0, 0)


# Function which will execute at mouse event
def detect_color(event, x, y, flag, param):
    global b, g, r, img

    if event == cv2.EVENT_MOUSEMOVE:

        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

        cv2.rectangle(img, (15, 15), (700, 50), (b, g, r), -1)
        text = get_Color_Name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

        if r + g + b >= 500:
            cv2.putText(img, text, (40, 40), cv2.FONT_HERSHEY_PLAIN, 1.3, black_font, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, text, (40, 40), cv2.FONT_HERSHEY_PLAIN, 1.3, white_font, 1, cv2.LINE_AA)


# Function to determine the color name
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
    global img

    img = cv2.resize(cv2.imread(
        askopenfilename(initialdir="/", title="Select A File", filetype=(("jpeg", "*.jpg"), ("png", "*.png")))),
                     (600, 600))

    # setting up the window
    cv2.namedWindow("Color Detection")
    cv2.setMouseCallback("Color Detection", detect_color)

    while True:
        # Displaying the window
        cv2.imshow("Color Detection", img)

        # Checking if Esc key is pressed then breaking out of the loop & destroying all the windows
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Checking if CLOSE ('X') is clicked on the cv2 window & if true then displaying pop-up
        if cv2.getWindowProperty("Color Detection", cv2.WND_PROP_VISIBLE) == 0:
            tkinter.messagebox.showinfo("Tip", "Press Esc key to close the window")

    cv2.destroyAllWindows()


# Creating a button
Import_button = ttk.Button(win, text='Browse an image', command=dialog_win)
Import_button.grid(row=4, column=4, pady=10, padx=5)

win.mainloop()
