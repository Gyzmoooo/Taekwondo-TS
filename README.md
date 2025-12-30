# Taekwondo TS
### Train hard, kick smart!

**Taekwondo Training-System** is the world’s first data collection system designed specifically for Taekwondo, allowing athletes to monitor and improve their fighting skills through objective statistical data. The first prototype of the system, whose code is inside this repository, system utilizes wearable "Smartstraps" and a ML model to identify kicks.

```markdown
$\color{red}{\text{Please note that this repo is public for display of the work done. This project is not really meant to be repreducible, as it is an early-prototype version of a potential future product.}}$
```

## 🏗️ System Architecture
The system consists of four wearable Smartstraps and a central computer.

### 1. Hardware (Smartstraps)
Each strap is housed in a 3D-printed ABS box and contains:
* **Microcontroller**: ESP-32-C3-SUPER-MINI, chosen for its compact size and Wi-Fi/Bluetooth capabilities.
* **Sensor**: MPU-6050 Inertial Measurement Unit (IMU) with a 3-axis accelerometer and 3-axis gyroscope.
* **Power**: 1000mAh battery providing approximately 5 hours of continuous use.
* **Configuration**: One "Master" strap (equipped with a physical trigger button) and three "Slave" straps.

### 2. Software Stack
* **Firmware**: Written in C++ (`Master.ino` and `Slave.ino`).
* **Desktop Application**: Written in Python (`main.py`) using Pandas for data processing and PyQt6 for the GUI.
* **Data Processing**: Uses Signal Magnitude Vector (SMV) analysis to distinguish between "Kick" and "Still" movements.

## ⚙️ How It Works
1.  **Recording**: The athlete presses the button on the Master Smartstrap to begin data collection.
2.  **Transmission**: After combat, a second button press stops the recording and sends all sensor data to the PC via HTTP.
3.  **Classification**: The AI processes a 480-point input vector (4 sensors × 6 data points × 20 measurements) for each identified kick.
4.  **Review**: Results are displayed in the dashboard for immediate feedback.

## 🛠️ Installation & Setup
1.  **Repository**: Clone this repo to your local machine.
2.  **Hardware**: Flash the ESP32-C3 microcontrollers with the provided `.ino` files found in the firmware directory.
3.  **Enviroment**
    * Python 3.13.7
    * Conda

    ### Setup with Conda
    ```bash
    conda env create -f environment.yml
    conda activate TKDconda
    ```
4.  **Run**: Launch the system by running:
    ```bash
    python main.py
    ```

## 👥 The Team
Developed by:
* **Giulio Gismondi**: Project manager, software developer, and AI model training.
* **Simone Gismondi**: Hardware construction, 3D design, soldering, and PCB design.

---
