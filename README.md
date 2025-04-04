# Gesture-Based Sound and Media Control

## ⭐ Overview
This Python program allows you to control your system's volume using hand gestures and switch media using eye blinks. It utilizes a webcam to track hand and face movements, making it an interactive and touchless experience.

## 🎯 Features
- **🎵 Volume Control:** Adjust the system volume by changing the distance between your thumb and index finger.
- **⏭ Media Control:** Blink twice rapidly to skip to the next track or video.
- **📷 Real-time Tracking:** Uses OpenCV and MediaPipe for accurate hand and face detection.

## 🛠 Technologies Used
- *Python*
- *OpenCV*
- *MediaPipe*
- *PyAutoGUI*
- *Pycaw (for audio control)*

## 📌 Installation
1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/gesture-sound-control.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd gesture-sound-control
   ```
3. **Install the required dependencies:**
   ```bash
   pip install opencv-python numpy mediapipe pyautogui comtypes pycaw
   ```

## 🚀 Usage
1. **Run the script:**
   ```bash
   python main.py
   ```
2. **Ensure your webcam is working properly.**
3. **Perform the following gestures:**
   - ✋ Move your *thumb* and *index finger* closer or farther apart to change the volume.
   - 👀 Blink *twice quickly* to skip to the next media track.

## ⌨️ Key Bindings
- Press `q` to *exit* the program.

## ⚠️ Troubleshooting
- If the **webcam** does not start, ensure it is not being used by another application.
- If the **volume control** is unresponsive, check your system's audio settings.

## 🔥 Future Improvements
- ➕ Add support for *pausing and playing* media.
- 🔆 Implement *brightness control* using gestures.
- 🎯 Enhance *eye blink detection* accuracy.


## 👤 Author
Developed by *Aryan Dhiman*
email : *aryandhiman003@gmail.com*

