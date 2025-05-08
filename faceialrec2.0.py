import face_recognition
import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime

# Configuration
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_LOGS_DIR = "attendance_logs"
MIN_FACE_SIZE = 100  # Minimum face size in pixels

def setup_directories():
    """Create required directories if they don't exist"""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_LOGS_DIR, exist_ok=True)

def load_known_faces():
    """Load all known faces from the directory"""
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    name = os.path.splitext(filename)[0].replace("_", " ")
                    known_face_names.append(name)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return known_face_encodings, known_face_names

def initialize_camera():
    """Initialize camera with error handling"""
    max_attempts = 3
    for attempt in range(max_attempts):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Test frame capture
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap
            cap.release()
        time.sleep(1)
    print("Error: Could not initialize camera after multiple attempts")
    return None

def process_frame(frame, scale_factor=0.25):
    """Process frame with error handling"""
    if frame is None or frame.size == 0:
        return None, None
    
    try:
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        return small_frame, rgb_small_frame
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return None, None

def register_new_face():
    """Improved face registration with robust error handling"""
    print("\n=== Face Registration ===")
    name = input("Enter person's full name: ").strip()
    if not name:
        print("Registration cancelled - no name provided.")
        return False
    
    filename = name.replace(" ", "_") + ".jpg"
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    
    cap = initialize_camera()
    if cap is None:
        return False
    
    print("\nInstructions:")
    print("1. Face the camera directly with good lighting")
    print("2. Press SPACE to capture when face is aligned")
    print("3. Press ESC to cancel")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Could not read frame - retrying...")
                time.sleep(0.1)
                continue
            
            # Process frame for display
            display_frame = frame.copy()
            
            # Convert to RGB for face detection
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 1:
                    top, right, bottom, left = face_locations[0]
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    status = "Ready to capture"
                else:
                    status = "Align your face (one person)"
                
                cv2.putText(display_frame, status, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Face detection error: {str(e)}")
                continue
            
            cv2.imshow("Register Face", display_frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("Registration cancelled.")
                return False
            elif key == 32:  # SPACE
                if len(face_locations) != 1:
                    print("Please position exactly one face in frame")
                    continue
                
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                
                # Validate face image
                if (face_image.size == 0 or 
                    face_image.shape[0] < MIN_FACE_SIZE or 
                    face_image.shape[1] < MIN_FACE_SIZE):
                    print("Face too small - move closer")
                    continue
                
                try:
                    if cv2.imwrite(filepath, face_image):
                        print(f"\nSuccessfully registered: {name}")
                        return True
                    else:
                        print("Failed to save image")
                except Exception as e:
                    print(f"Error saving image: {str(e)}")
                    return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return False

def main():
    setup_directories()
    known_face_encodings, known_face_names = load_known_faces()
    
    if not known_face_names:
        print("No known faces found. Please register at least one face.")
        if not register_new_face():
            return
    
    cap = initialize_camera()
    if cap is None:
        return
    
    expected_attendees = known_face_names.copy()
    log_file = os.path.join(ATTENDANCE_LOGS_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    attended = set()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Could not read frame - retrying...")
                time.sleep(0.1)
                continue
            
            small_frame, rgb_small_frame = process_frame(frame)
            if small_frame is None:
                continue
            
            # Face recognition
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    # Display recognition
                    cv2.putText(frame, f"{name} - Present", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Log attendance
                    if name in expected_attendees and name not in attended:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            if os.stat(log_file).st_size == 0:
                                writer.writerow(["Name", "Timestamp", "Status"])
                            writer.writerow([name, current_time, "Present"])
                        attended.add(name)
                        expected_attendees.remove(name)
            
            # Display status
            status = f"Remaining: {len(expected_attendees)}" if expected_attendees else "All attended"
            controls = "Press: [R] Register | [Q] Quit"
            
            cv2.putText(frame, controls, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, status, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Attendance System", frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 13:  # Q or Enter
                break
            elif key & 0xFF == ord('r'):  # R
                if register_new_face():
                    known_face_encodings, known_face_names = load_known_faces()
                    new_name = known_face_names[-1]
                    if new_name not in expected_attendees:
                        expected_attendees.append(new_name)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAttendance logged to: {log_file}")
        if expected_attendees:
            print("Absentees:", ", ".join(expected_attendees))

if __name__ == "__main__":
    main()
