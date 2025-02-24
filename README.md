# Battery Charge Optimization using TinyML on ESP32

## Overview
This project explores the discovery of human behavioral tendencies using a lightweight neural network running on an ESP32. The goal is to optimize battery charging schedules by learning when a user is least active, thereby extending battery longevity. While the system does interact with battery charging hardware (a nod to electrical engineering), the primary focus is on the neural network code and its ability to detect periodic patterns in user behavior.

## Features
- **Incremental Learning:** Continuously learns from daily battery usage data over a 4-week rolling window.
- **Cyclical Data Encoding:** Uses sine and cosine transformations to represent the day of the week, ensuring that similar days (e.g., all Mondays) are treated alike.
- **Per-Hour Neural Networks:** Implements individual neural networks for each hour, providing fine-grained predictions for battery usage.
- **Real-Time Recommendations:** Predicts battery usage for the upcoming day and recommends optimal charging times during periods of low usage.
- **ESP32 Compatible:** Designed to run efficiently on an ESP32, leveraging its capability for embedded TinyML applications.

## How It Works
- **Data Collection:** Every hour, the system records battery usage data. This data is stored over 28 days, allowing the network to learn from multiple weeks of behavior.
- **Neural Network Architecture:**  
  - Each hour of the day has its own neural network.
  - The network uses two inputs, representing the sine and cosine values of the weekday angle, to capture the 7-day cyclical pattern.
  - A simple feedforward neural network with one hidden layer and ReLU activation is used for predictions.
- **Training and Prediction:**  
  - The network is trained daily with historical data, updating its weights using gradient descent.
  - Predictions for the next day are made by feeding in the cyclically encoded weekday values. If predicted usage is below a certain threshold, the system recommends starting battery charge at the preceding hour.

## Code Structure
- **Data Handling:**  
  - `simulateDay()` collects hourly battery usage via Serial input.
  - A rolling data array stores usage for up to 28 days.
- **Neural Network Functions:**  
  - `forwardNN_hour()`: Performs a forward pass using sine and cosine encoded inputs.
  - `trainNN_hour()`: Implements gradient descent updates for training.
  - `predictNN_hour()`: Provides predictions based on current network weights.
- **Training and Prediction Loop:**  
  - `trainNetworksForDay()` trains each hour-specific network with historical data.
  - `runPredictionsForNextDay()` predicts battery usage for the next day and identifies optimal charging times.

## Setup Instructions
1. **Environment Setup:**  
   - Install the [Arduino IDE](https://www.arduino.cc/en/software) and configure ESP32 board support.
2. **Code Deployment:**  
   - Copy the provided code into a new Arduino project.
   - Connect your ESP32 and select the appropriate board and COM port.
3. **Running the Project:**  
   - Upload the code to your ESP32.
   - Open the Serial Monitor at 115200 baud.
   - Input hourly battery usage data as prompted and observe predictions and charging recommendations.

## Future Enhancements
- **Sensor Integration:** Automate data collection with actual battery usage sensors.
- **Model Complexity:** Experiment with deeper or alternative network architectures for improved predictions.
- **Behavioral Insights:** Further analyze human activity patterns to refine charging strategies.

## Conclusion
This project demonstrates how TinyML on an ESP32 can be used to uncover human tendencies through simple neural networks. By predicting low-usage periods and recommending optimal charging times, it provides an innovative way to extend battery life while bridging the gap between software intelligence and hardware functionality.
