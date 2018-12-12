#pragma once

#include <gpucf/log.h>

#include <fstream>
#include <vector>


template<typename T, class Parser>
class DataReader {

public:
    DataReader(const std::string &fName) { readFrom(fName); }

    const std::vector<T> &get() { return data; }

private:
    std::vector<T> data;   

    void readFrom(const std::string &fName) {
        Parser parser;
        std::ifstream infile(fName); 

        int lineNr = 1;
        for (std::string line; 
                std::getline(infile, line); 
                lineNr++) {
            bool ok = parser(line, &data);
            
            ASSERT(ok) << "\n\n Failed to load file " << fName 
                << ". Error in line " << lineNr;  
        }
    }

};

// vim: set ts=4 sw=4 sts=4 expandtab:
