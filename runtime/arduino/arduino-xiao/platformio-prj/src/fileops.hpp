#pragma once
#include <SD.h>

void initDirectories();
bool getNextFromDir(File &dir, char* outName);
bool getNextImage(char* outName, int &label);
