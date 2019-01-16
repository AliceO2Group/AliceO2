#pragma once

#include "DataReader.h"
#include "DigitParser.h"

#include <args/args.hxx>


namespace gpucf
{

class DigitReader : public DataReader<Digit, DigitParser>
{

public:
    class Flags
    {

    public:
        args::ValueFlag<std::string> infile;
        args::ValueFlag<int> workers;

        Flags(args::Group &required, args::Group &optional)
            : infile(required, "File", "File of digits.",
                    {'d', "digits"})
            , workers(optional, "N", 
                "Number of workers that parse the digit file. (default=4)",
                {'w', "workers"}, 4)
        {
        }
        
    };

    DigitReader(const std::string &file, size_t numWorkers) 
        : DataReader<Digit, DigitParser>(file, numWorkers)
    {
    }

    DigitReader(Flags &flags)
        : DigitReader(args::get(flags.infile), args::get(flags.workers))
    {
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
