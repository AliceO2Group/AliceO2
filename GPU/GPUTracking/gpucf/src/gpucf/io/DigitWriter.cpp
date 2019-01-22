#include "DigitWriter.h"

#include <gpucf/log.h>

#include <fstream>


using namespace gpucf;


void DigitWriter::write(const std::vector<Digit> &digits)
{
    log::Info() << "Writing peaks to file " << fName << ".";
    std::ofstream out(fName);

    for (const Digit &d : digits)
    {
        out << serialize(d) << "\n"; 
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
