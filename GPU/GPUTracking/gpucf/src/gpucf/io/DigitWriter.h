#pragma once

#include <gpucf/common/Digit.h>


namespace gpucf
{

class DigitWriter
{
    
public:
    DigitWriter(const std::string &out)
        : fName(out)
    {
    }

    void write(const std::vector<Digit> &);

private:
    std::string fName;

};
 
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
