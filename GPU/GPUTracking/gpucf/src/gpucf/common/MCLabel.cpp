#include "MCLabel.h"

#include <ostream>


using namespace gpucf;


std::ostream &gpucf::operator<<(std::ostream &o, const MCLabel &l)
{
    return o << "{ event: " << l.event << ", track: " << l.track << " }";
}

// vim: set ts=4 sw=4 sts=4 expandtab:

