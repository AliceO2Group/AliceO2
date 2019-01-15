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

        Flags(args::Group &required, args::Group &)
            : infile(required, "digitFile", "File of digits.",
                    {'d', "digits"})
        {
        }
        
    };

    DigitReader(const std::string &file) 
        : DataReader<Digit, DigitParser>(file)
    {
    }

    DigitReader(Flags &flags)
        : DigitReader(args::get(flags.infile))
    {
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
