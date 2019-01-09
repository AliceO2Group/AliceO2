#include "Digit.h"

#include <ostream>


std::ostream &operator<<(std::ostream &os, const Digit &d) 
{
    return os << "Digit{ "
       << "charge: " << d.charge << ", "
       << "cru: " << d.cru << ", "
       << "row: " << d.row << ", "
       << "pad: " << d.pad << ", "
       << "time: " << d.time
       << " }";
}

// vim: set ts=4 sw=4 sts=4 expandtab:
