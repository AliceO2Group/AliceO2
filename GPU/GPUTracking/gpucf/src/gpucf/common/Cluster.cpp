#include "Cluster.h"

#include <ostream>


std::ostream &operator<<(std::ostream &os, const Cluster &c)
{
    return os << "Cluster{ "
        << "cru: " << c.cru << ", "
        << "row: " << c.row << ", "
        << "Q: "   << c.Q   << ", "
        << "QMax: " << c.QMax << ", "
        << "padMean: " << c.padMean << ", "
        << "timeMean: " << c.timeMean << ", "
        << "padMean: " << c.padMean << ", "
        << "timeMean: " << c.timeMean
        << "}";
}

// vim: set ts=4 sw=4 sts=4 expandtab:

