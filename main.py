import tkinter as tk
from tkinter import filedialog
import cv2
from Detector import Detector
import Tracker


def select_input_file():
    filepath = filedialog.askopenfilename()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, filepath)


def select_output_file():
    filepath = filedialog.asksaveasfilename(defaultextension=".mp4")
    output_entry.delete(0, tk.END)
    output_entry.insert(0, filepath)


def process_video():
    input_path = input_entry.get()
    output_path = output_entry.get()
    use_density = density_var.get()

    detector = Detector()
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Read each frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Process each frame
        tracked_objects = Tracker.update_tracker(detector, frame, None, use_density)
        # Live show each processed frame
        cv2.imshow('Tracked Persons', tracked_objects)
        out.write(tracked_objects)

        # Press 'q' to quit anytime
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    status_label.config(text="Processing Completed!")


root = tk.Tk()
root.title("Video Processing App")

# Input file selection
tk.Label(root, text="Input Video File:").pack()
input_entry = tk.Entry(root, width=50)
input_entry.pack()
tk.Button(root, text="Browse", command=select_input_file).pack()

# Output file selection
tk.Label(root, text="Output File:").pack()
output_entry = tk.Entry(root, width=50)
output_entry.pack()
tk.Button(root, text="Browse", command=select_output_file).pack()

# Checkbox for density visualization
density_var = tk.IntVar()
density_checkbox = tk.Checkbutton(root, text="Use Density Visualization", variable=density_var)
density_checkbox.pack()

# Process button
tk.Button(root, text="Start Processing", command=process_video).pack()

# Notation on UI
note_label = tk.Label(root, text="Note: please wait for 'Processing Completed!' in order to get the full processed video. You can also press 'q' any time to stop the processing.")
note_label.pack()

# Status label
status_label = tk.Label(root, text="")
status_label.pack()

root.mainloop()
