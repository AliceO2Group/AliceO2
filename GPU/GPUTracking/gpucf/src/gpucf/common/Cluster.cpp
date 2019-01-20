#include "Cluster.h"

#include <gpucf/common/float.h>

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
        << "padSigma: " << c.padSigma << ", "
        << "timeSigma: " << c.timeSigma
        << "}";
}

bool operator==(const Cluster &c1, const Cluster &c2)
{
    return c1.cru == c2.cru
             && c1.row == c2.row
             && c1.Q == c2.Q
             && c1.QMax == c2.QMax
             && FLOAT_EQ(c1.padMean, c2.padMean)
             && FLOAT_EQ(c1.timeMean, c2.timeMean)
             && FLOAT_EQ(c1.padSigma, c2.padSigma)
             && FLOAT_EQ(c1.timeSigma, c2.timeSigma);
}

// vim: set ts=4 sw=4 sts=4 expandtab:

