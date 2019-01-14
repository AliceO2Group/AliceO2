#pragma once

#include <gpucf/log.h>
#include "ParsingError.h"

#include <fstream>
#include <vector>


template<typename T, class Parser>
class DataReader 
{

public:
    DataReader(const std::string &fName) 
    { 
        readFrom(fName); 
    }

    const std::vector<T> &get() 
    { 
        return data; 
    }

private:
    std::vector<T> data;   

    void readFrom(const std::string &fName) 
    {
        log::Info() << "Reading file " << fName;

        Parser parser;
        std::ifstream infile(fName); 

        size_t lineNr = 1;
        for (std::string line; 
                std::getline(infile, line); 
                lineNr++) 
        {
            bool ok = parser(line, &data);

            if (!ok) 
            {
                throw ParsingError(fName, lineNr);
            }
            
        }
    }

};

// vim: set ts=4 sw=4 sts=4 expandtab:
