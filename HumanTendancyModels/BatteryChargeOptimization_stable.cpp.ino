#define DAYS_TO_STORE 28
#define HOURS_PER_DAY 24
#define USAGE_THRESHOLD 1  // threshold for deciding if battery usage is significant
#define NUM_NEURONS 3      // number of neurons in the hidden layer

// Rolling window: usageData[d][h] holds usage for day d at hour h
int usageData[DAYS_TO_STORE][HOURS_PER_DAY];
int currentDay = 0;   // index for the current day (0 to 6)
int numDays = 0;      // how many days of data have been recorded (max 7)
int totalDaysSimulated = 0;  // total days simulated (for labeling output)

// --- Neural Network Parameters for Each Hour ---
// We create one network per hour. For hour h, we store parameters in the [h] index.
double nn_weights_input[HOURS_PER_DAY][NUM_NEURONS];   // weights from input to hidden layer
double nn_biases_hidden[HOURS_PER_DAY][NUM_NEURONS];     // biases for hidden layer neurons
double nn_weights_hidden[HOURS_PER_DAY][NUM_NEURONS];    // weights from hidden layer to output
double nn_bias_output[HOURS_PER_DAY];                    // bias for the output neuron

double learning_rate = 0.01;  // learning rate for batch updates

// --- Activation Functions ---
double relu(double x) {
  return (x > 0) ? x : 0;
}
double relu_derivative(double x) {
  return (x > 0) ? 1.0 : 0.0;
}

// --- Forward Pass for a Given Hour's Network ---
// 'hourIndex' selects which network to use.
// 'input' will be the day index (as a double); you could normalize if desired.
double forwardNN_hour(int hourIndex, double input, double hidden[NUM_NEURONS]) {
  for (int i = 0; i < NUM_NEURONS; i++) {
    double sum = nn_weights_input[hourIndex][i] * input + nn_biases_hidden[hourIndex][i];
    hidden[i] = relu(sum);
  }
  double output = nn_bias_output[hourIndex];
  for (int i = 0; i < NUM_NEURONS; i++) {
    output += nn_weights_hidden[hourIndex][i] * hidden[i];
  }
  return output;
}

// --- Training Function for a Given Hour's Network ---
// Performs one gradient descent update on one training sample.
void trainNN_hour(int hourIndex, double input, double target) {
  double hidden[NUM_NEURONS];
  double prediction = forwardNN_hour(hourIndex, input, hidden);
  double error = prediction - target;  // error = (predicted - actual)
  
  // Backpropagation for the output layer.
  for (int i = 0; i < NUM_NEURONS; i++) {
    double grad_output = error * hidden[i];
    nn_weights_hidden[hourIndex][i] -= learning_rate * grad_output;
  }
  nn_bias_output[hourIndex] -= learning_rate * error;
  
  // Backpropagation for the hidden layer.
  for (int i = 0; i < NUM_NEURONS; i++) {
    double sum = nn_weights_input[hourIndex][i] * input + nn_biases_hidden[hourIndex][i];
    double d_activation = relu_derivative(sum);
    double error_hidden = error * nn_weights_hidden[hourIndex][i] * d_activation;
    nn_weights_input[hourIndex][i] -= learning_rate * error_hidden * input;
    nn_biases_hidden[hourIndex][i]  -= learning_rate * error_hidden;
  }
}

// --- Convenience: Prediction Function for a Given Hour's Network ---
double predictNN_hour(int hourIndex, double input) {
  double hidden[NUM_NEURONS];
  double prediction = forwardNN_hour(hourIndex, input, hidden);
  // Ensure we don't predict negative usage.
  return (prediction < 0) ? 0 : prediction;
}

// --- Setup ---
void setup() {
  Serial.begin(115200);
  // Initialize usageData to zeros.
  for (int d = 0; d < DAYS_TO_STORE; d++) {
    for (int h = 0; h < HOURS_PER_DAY; h++) {
      usageData[d][h] = 0;
    }
  }
  
  // Initialize the neural networks for each hour with small random weights.
  randomSeed(analogRead(0));
  for (int h = 0; h < HOURS_PER_DAY; h++) {
    for (int i = 0; i < NUM_NEURONS; i++) {
      nn_weights_input[h][i] = ((double)random(-100, 100)) / 10000.0;
      nn_biases_hidden[h][i] = 0.0;
      nn_weights_hidden[h][i] = ((double)random(-100, 100)) / 10000.0;
    }
    nn_bias_output[h] = 0.0;
  }
  
  Serial.println("ESP32 Battery Usage & Incremental TinyML Neural Network (Per-Hour) Debug Test Code");
  Serial.println("====================================================================================");
  Serial.println("For each hour, input an integer representing the battery usage count.");
  Serial.println();
}

// --- simulateDay() ---
// For each hour of the day, prompt the user for usage and store it.
void simulateDay() {
  Serial.println("Simulating a new day...");
  for (int hour = 0; hour < HOURS_PER_DAY; hour++) {
    Serial.print("Hour ");
    Serial.print(hour);
    Serial.println(": Please enter battery usage count (integer):");
    
    // Wait until data is available via Serial Monitor.
    while (Serial.available() == 0) {
      delay(100);
    }
    
    int usage = Serial.parseInt();
    // Clear any extra characters from Serial.
    while (Serial.available() > 0) { Serial.read(); }
    
    // Store the usage in the rolling data array.
    usageData[currentDay][hour] = usage;
    
    Serial.print("Recorded usage for hour ");
    Serial.print(hour);
    Serial.print(": ");
    Serial.println(usage);
    Serial.println();
  }
  
  // Update rolling window indices.
  currentDay = (currentDay + 1) % DAYS_TO_STORE;
  if (numDays < DAYS_TO_STORE) {
    numDays++;
  }
  totalDaysSimulated++;
}

// --- trainNetworksForDay() ---
// After a day is simulated, for each hour, train its dedicated network
// using the historical data (from all stored days) for that hour.
void trainNetworksForDay() {
  // For each hour (0-23)
  for (int hour = 0; hour < HOURS_PER_DAY; hour++) {
    // For each stored day, treat the day index as the input (e.g., 0,1,...)
    // and the recorded usage as the target.
    // You might normalize the day index if needed.
    for (int d = 0; d < numDays; d++) {
      double input = (double)d;  // day index as feature
      double target = (double)usageData[d][hour];
      // Perform one training update.
      trainNN_hour(hour, input, target);
    }
  }
}

// --- runPredictionsForNextDay() ---
// Using the trained per-hour networks, predict the usage for each hour
// for the next day (using day index = numDays as the predicted input).
// Also calculates the gap between non-consecutive significant usage hours.
void runPredictionsForNextDay() {
  Serial.println("Predicted battery usage for each hour (for next day):");
  
  // Array to store the hours that have significant predicted usage.
  int significantHours[HOURS_PER_DAY];
  int count = 0;
  
  // Loop over each hour and predict usage.
  for (int hour = 0; hour < HOURS_PER_DAY; hour++) {
    double predictedUsage = predictNN_hour(hour, (double)numDays);
    Serial.print("Hour ");
    Serial.print(hour);
    Serial.print(": Predicted usage = ");
    Serial.print(predictedUsage, 4);
    
    if (predictedUsage >= USAGE_THRESHOLD) {
      // Record the hour if usage is significant.
      significantHours[count++] = hour;
      int chargeHour = (hour - 1 >= 0) ? hour - 1 : 23; // wrap-around if hour 0
      Serial.print(" -> Battery expected to be used. Recommend starting charge at hour ");
      Serial.println(chargeHour);
    } else {
      Serial.println(" -> No significant usage predicted.");
    }
  }
  
  // Now, calculate the gaps between non-consecutive groups.
  if (count > 0) {
    Serial.println();
    Serial.println("slope future reference data:");
    
    // Initialize the first block.
    int blockStart = significantHours[0];
    int blockEnd = significantHours[0];
    
    // Iterate through the stored significant hours.
    for (int i = 1; i < count; i++) {
      if (significantHours[i] == blockEnd + 1) {
        // If consecutive, extend the current block.
        blockEnd = significantHours[i];
      } else {
        // Not consecutive, so calculate the gap from the end of the previous block
        // to the beginning of the next block.
        int gap = significantHours[i] - blockEnd;
        // Only print if the gap is greater than 1 hour.
        if (gap > 1) {
          Serial.print("hour ");
          Serial.print(blockEnd);
          Serial.print(" to ");
          Serial.print(significantHours[i]);
          Serial.print(" = ");
          Serial.print(gap);
          Serial.println(" hours");
        }
        // Start a new block.
        blockStart = significantHours[i];
        blockEnd = significantHours[i];
      }
    }
  }
  
  Serial.println();
}

void loop() {
  // Simulate one day (24 hours)
  simulateDay();
  
  // Batch train each hour's network using all stored historical data.
  trainNetworksForDay();
  
  // Print a summary for the day.
  Serial.print("Day ");
  Serial.print(totalDaysSimulated);
  Serial.println(" Summary:");
  runPredictionsForNextDay();
  Serial.println("--------------------------------------------------");
  Serial.println();
  
  // Wait a bit before starting the next day's simulation.
  delay(3000);
  // Loop continuously without resetting the board.
}
