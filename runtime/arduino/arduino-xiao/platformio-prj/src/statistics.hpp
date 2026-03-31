#pragma once

void updateConfusionMatrix(int realLabel, int predLabel);
float getAccuracy();
void printConfusionMatrix();
bool checkEndDataset(int samples);
