// --- Definitions and Parameters ---
#define DAYS_TO_STORE 28
#define HOURS_PER_DAY 24
#define USAGE_THRESHOLD 1  // threshold for deciding if battery usage is significant
#define NUM_NEURONS 3      // number of neurons in the hidden layer
#define NUM_INPUTS 2       // using sin and cos inputs for weekday encoding
#define PI 3.14159265358979323846

// --- Data Storage ---
int usageData[DAYS_TO_STORE][HOURS_PER_DAY]; // usageData[d][h] holds usage for day d at hour h
int currentDay = 0;      // index for the current day (0 to 27)
int numDays = 0;         // how many days of data have been recorded (max 28)
int totalDaysSimulated = 0;  // total days simulated (for labeling output)

// --- Neural Network Parameters for Each Hour ---
// One network per hour.
double nn_weights_input[HOURS_PER_DAY][NUM_INPUTS][NUM_NEURONS];  // weights from inputs to hidden layer
double nn_biases_hidden[HOURS_PER_DAY][NUM_NEURONS];                // biases for hidden layer neurons
double nn_weights_hidden[HOURS_PER_DAY][NUM_NEURONS];               // weights from hidden layer to output
double nn_bias_output[HOURS_PER_DAY];                               // bias for the output neuron

double learning_rate = 0.01;  // learning rate for batch updates

// --- Activation Functions ---
double relu(double x) {
  return (x > 0) ? x : 0;
}
double relu_derivative(double x) {
  return (x > 0) ? 1.0 : 0.0;
}

// --- Forward Pass for a Given Hour's Network ---
// Uses two inputs: input1 (sin value) and input2 (cos value) for the weekday.
double forwardNN_hour(int hourIndex, double input1, double input2, double hidden[NUM_NEURONS]) {
  for (int i = 0; i < NUM_NEURONS; i++) {
    double sum = nn_weights_input[hourIndex][0][i] * input1 
               + nn_weights_input[hourIndex][1][i] * input2 
               + nn_biases_hidden[hourIndex][i];
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
void trainNN_hour(int hourIndex, double input1, double input2, double target) {
  double hidden[NUM_NEURONS];
  double prediction = forwardNN_hour(hourIndex, input1, input2, hidden);
  double error = prediction - target;  // error = (predicted - actual)
  
  // Backpropagation for the output layer.
  for (int i = 0; i < NUM_NEURONS; i++) {
    double grad_output = error * hidden[i];
    nn_weights_hidden[hourIndex][i] -= learning_rate * grad_output;
  }
  nn_bias_output[hourIndex] -= learning_rate * error;
  
  // Backpropagation for the hidden layer.
  for (int i = 0; i < NUM_NEURONS; i++) {
    double sum = nn_weights_input[hourIndex][0][i] * input1 
               + nn_weights_input[hourIndex][1][i] * input2 
               + nn_biases_hidden[hourIndex][i];
    double d_activation = relu_derivative(sum);
    double error_hidden = error * nn_weights_hidden[hourIndex][i] * d_activation;
    // Update weights for both inputs.
    nn_weights_input[hourIndex][0][i] -= learning_rate * error_hidden * input1;
    nn_weights_input[hourIndex][1][i] -= learning_rate * error_hidden * input2;
    nn_biases_hidden[hourIndex][i]  -= learning_rate * error_hidden;
  }
}

// --- Prediction Function for a Given Hour's Network ---
double predictNN_hour(int hourIndex, double input1, double input2) {
  double hidden[NUM_NEURONS];
  double prediction = forwardNN_hour(hourIndex, input1, input2, hidden);
  // Ensure we don't predict negative usage.
  return (prediction < 0) ? 0 : prediction;
}

// --- Setup Function ---
void setup() {
  Serial.begin(115200);
  
  // Initialize usageData to zeros.
  for (int d = 0; d < DAYS_TO_STORE; d++) {
    for (int h = 0; h < HOURS_PER_DAY; h++) {
      usageData[d][h] = 0;
    }
  }
  
  // Initialize neural network weights and biases with small random values.
  randomSeed(analogRead(0));
  for (int h = 0; h < HOURS_PER_DAY; h++) {
    for (int j = 0; j < NUM_INPUTS; j++) {
      for (int i = 0; i < NUM_NEURONS; i++) {
        nn_weights_input[h][j][i] = ((double)random(-100, 100)) / 10000.0;
      }
    }
    for (int i = 0; i < NUM_NEURONS; i++) {
      nn_biases_hidden[h][i] = 0.0;
      nn_weights_hidden[h][i] = ((double)random(-100, 100)) / 10000.0;
    }
    nn_bias_output[h] = 0.0;
  }
  
  Serial.println("ESP32 Battery Usage & Incremental TinyML Neural Network (Per-Hour)");
  Serial.println("with Weekly Sine/Cosine Encoding");
  Serial.println("==================================================================================");
  Serial.println("For each hour, input an integer representing the battery usage count.");
  Serial.println();
}

// --- simulateDay() ---
// Prompts the user for battery usage count for each hour of the day.
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
// Trains each hour's network using all stored historical data for that hour.
// Uses sine and cosine encoding of the weekday for each training sample.
void trainNetworksForDay() {
  for (int hour = 0; hour < HOURS_PER_DAY; hour++) {
    for (int d = 0; d < numDays; d++) {
      // Compute the cyclic weekday angle.
      double angle = 2 * PI * (d % 7) / 7.0;
      double weekdaySin = sin(angle);
      double weekdayCos = cos(angle);
      double target = (double)usageData[d][hour];
      // Perform one training update.
      trainNN_hour(hour, weekdaySin, weekdayCos, target);
    }
  }
}

// --- runPredictionsForNextDay() ---
// Uses the trained networks to predict battery usage for the next day.
// Applies sine and cosine encoding for the predicted weekday.
void runPredictionsForNextDay() {
  Serial.println("Predicted battery usage for each hour (for next day):");
  
  int significantHours[HOURS_PER_DAY]; // To record hours with significant predicted usage
  int count = 0;
  
  // Compute the weekday encoding for the next day.
  double angle = 2 * PI * (numDays % 7) / 7.0;
  double weekdaySin = sin(angle);
  double weekdayCos = cos(angle);
  
  // Predict usage for each hour.
  for (int hour = 0; hour < HOURS_PER_DAY; hour++) {
    double predictedUsage = predictNN_hour(hour, weekdaySin, weekdayCos);
    Serial.print("Hour ");
    Serial.print(hour);
    Serial.print(": Predicted usage = ");
    Serial.print(predictedUsage, 4);
    
    if (predictedUsage >= USAGE_THRESHOLD) {
      significantHours[count++] = hour;
      int chargeHour = (hour - 1 >= 0) ? hour - 1 : 23; // Wrap-around if hour 0
      Serial.print(" -> Battery expected to be used. Recommend starting charge at hour ");
      Serial.println(chargeHour);
    } else {
      Serial.println(" -> No significant usage predicted.");
    }
  }
  
  // Calculate gaps between non-consecutive significant usage hours.
  if (count > 0) {
    Serial.println();
    Serial.println("slope future reference data:");
    
    int blockStart = significantHours[0];
    int blockEnd = significantHours[0];
    
    for (int i = 1; i < count; i++) {
      if (significantHours[i] == blockEnd + 1) {
        // Extend current block if hours are consecutive.
        blockEnd = significantHours[i];
      } else {
        int gap = significantHours[i] - blockEnd;
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

// --- Main Loop ---
void loop() {
  simulateDay();
  trainNetworksForDay();
  
  Serial.print("Day ");
  Serial.print(totalDaysSimulated);
  Serial.println(" Summary:");
  runPredictionsForNextDay();
  Serial.println("--------------------------------------------------");
  Serial.println();
  
  // Wait before starting the next days simulation
  delay(3000);
}
