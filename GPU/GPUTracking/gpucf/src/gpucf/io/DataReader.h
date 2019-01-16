#pragma once

#include <gpucf/log.h>
#include <gpucf/Timer.h>
#include "ParsingError.h"

#include <nonstd/optional.hpp>

#include <cmath>
#include <fstream>
#include <thread>
#include <vector>


namespace gpucf
{

template<typename T, class Parser>
class DataReader 
{

public:
    DataReader(const std::string &fName, size_t numWorkers=4) 
    { 
        readFrom(fName, numWorkers); 
    }

    const std::vector<T> &get() 
    { 
        return data; 
    }

private:
    std::vector<T> data;   

    void readFrom(const std::string &fName, size_t numWorkers)
    {
        log::Info() << "Reading file " << fName 
                    << " with " << numWorkers << " threads...";

        std::ifstream infile(fName); 

        Timer splitTimer;

        splitTimer.begin();
        std::vector<std::string> lines = splitLines(infile);
        splitTimer.stop();
        log::Debug() << "Splitting into lines took " 
                    << splitTimer.elapsedTime() << "secs";

        std::vector<std::thread>    workers(numWorkers);
        std::vector<std::vector<T>> partialResults(numWorkers);
        std::vector<nonstd::optional<size_t>> lineErr(numWorkers);

        Timer parseTimer;

        parseTimer.begin();
        for (size_t w = 0; w < numWorkers; w++)
        {
            workers[w] = std::thread( parserWorker,
                    lines, w, numWorkers, &partialResults[w], &lineErr[w]);
        }

        data.reserve(lines.size());
        for (size_t w = 0; w < numWorkers; w++)
        {
            workers[w].join();
            
            if (lineErr[w])
            {
                throw ParsingError(fName, *lineErr[w]);
            }

            data.insert(data.end(), 
                    partialResults[w].begin(), partialResults[w].end());
        }
        parseTimer.stop();

        log::Debug() << "Parsing the file took " << parseTimer.elapsedTime() << "secs";
    }

    std::vector<std::string> splitLines(std::ifstream &infile)
    {
        std::vector<std::string> lines;
        for (std::string line;
             std::getline(infile, line);)
        {
            lines.push_back(line);
        }

        return lines;
        
    }

    static void parserWorker(
            const std::vector<std::string> &lines, 
            int myId,
            int numWorker,
            std::vector<T> *tgt,
            nonstd::optional<size_t> *err)
    {
        size_t numItems = lines.size();
        size_t itemsPerWorker = std::ceil(float(numItems) / float(numWorker));

        size_t myStart = myId * itemsPerWorker;
        size_t myEnd   = std::min(myStart + itemsPerWorker, numItems);

        Parser parser;

        for (size_t i = myStart; i < myEnd; i++)
        {
            bool ok = parser(lines[i], tgt);
            
            if (!ok)
            {
                *err = i;
                return;
            }
        }
        
    }

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
