#include <Arduino.h>
#include <SD.h>
#include <SPI.h>

#include "fileops.hpp"

#define MAX_FILENAME_LEN 64

File dirPerson;
File dirNoPerson;

bool dirsInitialized = false;

char currentFilename[MAX_FILENAME_LEN];
int currentLabel = -1;   // 1 = person, 0 = no_person

void initDirectories() {
    dirPerson = SD.open("/val/person");
    dirNoPerson = SD.open("/val/no_person");

    if (!dirPerson || !dirPerson.isDirectory()) {
        Serial.println("[ERR] /val/person not valid");
        return;
    }
    if (!dirNoPerson || !dirNoPerson.isDirectory()) {
        Serial.println("[ERR] /val/no_person not valid");
        return;
    }

    dirsInitialized = true;
    Serial.println("[OK] Directories initialized");
}

bool getNextFromDir(File &dir, char* outName) {
    while (true) {
        File f = dir.openNextFile();

        // End of directory entries → restart
        if (!f) {
            dir.rewindDirectory();
            f = dir.openNextFile();
            if (!f) return false; // empty directory
        }

        if (!f.isDirectory()) {
            String name = f.name();
            name.toLowerCase();
            if (name.endsWith(".jpg") || name.endsWith(".jpeg")) {
                strcpy(outName, f.name());
                f.close();
                return true;
            }
        }

        f.close();
    }
}

bool getNextImage(char* outName, int &label) {
    static bool toggle = false;

    if (!dirsInitialized) return false;

    if (toggle) {
        // PERSON → label 1
        if (getNextFromDir(dirPerson, outName)) {
            label = 1;
            toggle = !toggle;
            return true;
        }
    } else {
        // NO_PERSON → label 0
        if (getNextFromDir(dirNoPerson, outName)) {
            label = 0;
            toggle = !toggle;
            return true;
        }
    }

    return false;
}

