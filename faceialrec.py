import os
import cv2
import csv
import time
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import face_recognition

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_LOGS_DIR = "attendance_logs"
MIN_FACE_SIZE = 100

class AttendanceSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition Attendance")

        self.cap = None
        self.known_encodings = []
        self.known_names = []
        self.attended = set()
        self.expected = []
        self.log_file = None

        self.setup_ui()
        self.setup_dirs()
        self.load_faces()

    def setup_ui(self):
        self.video_frame = tk.Label(self.master)
        self.video_frame.pack()

        self.status_label = tk.Label(self.master, text="", fg="green", font=("Arial", 12))
        self.status_label.pack()

        tk.Button(self.master, text="Register", command=self.register_face).pack(pady=5)
        tk.Button(self.master, text="Start Attendance", command=self.start_attendance).pack(pady=5)
        tk.Button(self.master, text="Quit", command=self.master.quit).pack(pady=5)

    def setup_dirs(self):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        os.makedirs(ATTENDANCE_LOGS_DIR, exist_ok=True)

    def load_faces(self):
        self.known_encodings.clear()
        self.known_names.clear()
        for file in os.listdir(KNOWN_FACES_DIR):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(KNOWN_FACES_DIR, file)
                image = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(image)
                if enc:
                    self.known_encodings.append(enc[0])
                    self.known_names.append(os.path.splitext(file)[0].replace("_", " "))

    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()

    def register_face(self):
        name = simpledialog.askstring("Name", "Enter full name:")
        if not name:
            return
        filename = name.replace(" ", "_") + ".jpg"
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        if not self.initialize_camera():
            messagebox.showerror("Error", "Camera not found")
            return

        messagebox.showinfo("Instructions", "Align your face and press SPACE to capture")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            for (t, r, b, l) in locs:
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.imshow("Register", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == 32 and len(locs) == 1:
                face = frame[t:b, l:r]
                if face.shape[0] >= MIN_FACE_SIZE:
                    cv2.imwrite(filepath, face)
                    messagebox.showinfo("Saved", f"{name} registered successfully")
                    self.load_faces()
                    break
        self.cap.release()
        cv2.destroyAllWindows()

    def start_attendance(self):
        if not self.known_names:
            messagebox.showwarning("No Faces", "Register at least one face.")
            return
        if not self.initialize_camera():
            messagebox.showerror("Error", "Could not open camera")
            return

        present = False
        for _ in range(30):
            ret, frame = self.cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb, face_recognition.face_locations(rgb))
            if any(face_recognition.compare_faces(self.known_encodings, enc) for enc in encs):
                present = True
                break

        if not present:
            messagebox.showwarning("No Match", "No known face detected.")
            self.cap.release()
            return

        self.attended.clear()
        self.expected = self.known_names.copy()
        self.log_file = os.path.join(ATTENDANCE_LOGS_DIR, f"attendance_{datetime.now().date()}.csv")
        self.track_faces()

    def track_faces(self):
        ret, frame = self.cap.read()
        if not ret:
            self.master.after(10, self.track_faces)
            return

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        for enc in encs:
            matches = face_recognition.compare_faces(self.known_encodings, enc)
            dist = face_recognition.face_distance(self.known_encodings, enc)
            best_idx = np.argmin(dist)
            if matches[best_idx]:
                name = self.known_names[best_idx]
                now = datetime.now().strftime("%H:%M:%S")
                if name not in self.attended:
                    self.attended.add(name)
                    self.expected.remove(name)
                    self.record_attendance(name)
                    self.status_label.config(text=f"{name} is present at {now}")
                    self.master.after(5000, lambda: self.status_label.config(text=""))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

        self.master.after(10, self.track_faces)

    def record_attendance(self, name):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if os.stat(self.log_file).st_size == 0:
                writer.writerow(["Name", "Timestamp", "Status"])
            writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Present"])

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()

