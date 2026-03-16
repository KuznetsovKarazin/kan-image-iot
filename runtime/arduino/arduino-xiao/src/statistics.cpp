#include <Arduino.h>

#include "statistics.hpp"

uint32_t TP = 0;
uint32_t TN = 0;
uint32_t FP = 0;
uint32_t FN = 0;

void updateConfusionMatrix(int realLabel, int predLabel) {
    if (realLabel == 1 && predLabel == 1) TP++;
    else if (realLabel == 0 && predLabel == 0) TN++;
    else if (realLabel == 0 && predLabel == 1) FP++;
    else if (realLabel == 1 && predLabel == 0) FN++;
}

float getAccuracy() {
    uint32_t total = TP + TN + FP + FN;
    if (total == 0) return 0;
    return (float(TP + TN) / float(total))*100.0;
}

void printConfusionMatrix() {
    Serial.println("\n=== CONFUSION MATRIX ===");
    Serial.printf("TP: %u\n", TP);
    Serial.printf("TN: %u\n", TN);
    Serial.printf("FP: %u\n", FP);
    Serial.printf("FN: %u\n", FN);

    Serial.printf("\nAccuracy: %.4f\n", getAccuracy());
}

bool checkEndDataset(int samples) {    
    if (TP + TN + FP + FN >= samples) {
        return true;
    }
    return false;
}
