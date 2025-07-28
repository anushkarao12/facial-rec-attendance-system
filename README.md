# 🧠 Facial Recognition Attendance System

A Python-based facial recognition attendance system that automatically detects and recognizes faces through a webcam and marks attendance by saving data into log files. This project uses OpenCV and the `face_recognition` library for real-time face detection and recognition.

---

## 📁 Project Structure
facial-rec-attendance-system/
- facealrec2.0.py # Main script to run the facial recognition attendance system
- facial rec.rar # Compressed version of the entire project (for backup or sharing)
- attendance_logs/ # Folder to store CSV files of daily attendance
- known_faces/ # Folder containing known face images (used for recognition)

---

## 🧩 Required Modules

Install the following dependencies before running the project:

```bash
pip install cmake
pip install face_recognition
pip install opencv-python
pip install numpy
```

---

## 🚀 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/anushkarao12/facial-rec-attendance-system.git
cd facial-rec-attendance-system
```
2. Ensure your webcam is working and you have sample face images in the known_faces/ folder (with filenames as person names).

3. Run the Python script:
```bash
python facealrec.py
```
4. When a known face is detected, attendance is logged in the attendance_logs/ folder in a .csv file with the current date.

---

## 🧠 How It Works

- Loads images of known faces from the known_faces/ folder.
- Initializes webcam and captures live video.
- Uses face_recognition to detect and identify faces.
- If a face matches one in the dataset, its name is logged with the current time and date.
- Avoids duplicate attendance entries during the same session.
 
---

## 📌 Use Cases

- Classroom or office attendance
- Security check-in systems
- Touchless identification systems

---

## 📷 Future Enhancements

- Email/SMS notifications on recognition
- Integration with cloud storage or database
- Multi-camera support

---

## ⚠️ Disclaimer

This project is for educational purposes. Accuracy depends on lighting, camera quality, and the dataset used. Always ensure consent is obtained when using facial recognition.






