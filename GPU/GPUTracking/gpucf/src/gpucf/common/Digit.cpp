#include "Digit.h"

#include <ostream>
#include <sstream>


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

std::string serialize(const Digit &d)
{
    std::stringstream ss; 

    ss << "Digit: "
       << "charge = " << d.charge << ", "
       << "cru = " << d.cru << ", "
       << "row = " << d.row << ", "
       << "pad = " << d.pad << ", "
       << "time = " << d.time;

    return ss.str();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
