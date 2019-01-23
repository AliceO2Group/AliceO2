#include "Cluster.h"

#include <gpucf/common/float.h>

#include <ostream>
#include <sstream>


using namespace gpucf;


static_assert(sizeof(FloatCluster) == FLOAT_CLUSTER_SIZE);
static_assert(sizeof(Cluster) == sizeof(FloatCluster));
static_assert(sizeof(HalfCluster) == HALF_CLUSTER_SIZE);


Cluster::Cluster()
{
}

Cluster::Cluster(int _cru, int _row, float q, float qmax, float _padMean,
                 float _timeMean, float _padSigma, float _timeSigma)
{
    cru = _cru;
    row = _row;
    Q   = q;
    QMax = qmax;
    padMean = _padMean;
    timeMean = _timeMean;
    padSigma = _padSigma;
    timeSigma = _timeSigma;
}


bool Cluster::operator==(const Cluster &c2) const
{
    return cru == c2.cru
             && row == c2.row
             && FLOAT_EQ(Q, c2.Q)
             && FLOAT_EQ(QMax, c2.QMax)
             && FLOAT_EQ(padMean, c2.padMean)
             && FLOAT_EQ(timeMean, c2.timeMean)
             && FLOAT_EQ(padSigma, c2.padSigma)
             && FLOAT_EQ(timeSigma, c2.timeSigma);
}

Object Cluster::serialize() const
{
    Object obj("Cluster");

    SET_FIELD(obj, cru);
    SET_FIELD(obj, row);
    SET_FIELD(obj, Q);
    SET_FIELD(obj, QMax);
    SET_FIELD(obj, padMean);
    SET_FIELD(obj, timeMean);
    SET_FIELD(obj, padSigma);
    SET_FIELD(obj, timeSigma);

    return obj;
}

void Cluster::deserialize(const Object &obj)
{
    GET_INT(obj, cru);
    GET_INT(obj, row);
    GET_FLOAT(obj, Q);
    GET_FLOAT(obj, QMax);
    GET_FLOAT(obj, padMean);
    GET_FLOAT(obj, timeMean);
    GET_FLOAT(obj, padSigma);
    GET_FLOAT(obj, timeSigma);
}

std::ostream &gpucf::operator<<(std::ostream &os, const Cluster &c)
{
    return os << c.serialize();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
