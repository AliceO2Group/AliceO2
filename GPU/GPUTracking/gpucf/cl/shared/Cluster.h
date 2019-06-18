#if !defined(SHARED_CLUSTER_H)
#    define  SHARED_CLUSTER_H

#include "types.h"


typedef struct FloatCluster_s
{
    SHARED_FLOAT Q;
    SHARED_FLOAT QMax;
    SHARED_FLOAT padMean;
    SHARED_FLOAT timeMean;
    SHARED_FLOAT padSigma;
    SHARED_FLOAT timeSigma;
    SHARED_INT cru;
    SHARED_INT row;
} FloatCluster;

#define FLOAT_CLUSTER_SIZE 32


typedef struct HalfCluster_s
{
    SHARED_HALF Q;
    SHARED_HALF QMax;
    SHARED_HALF padMean;
    SHARED_HALF timeMean;
    SHARED_HALF padSigma;
    SHARED_HALF timeSigma;
    SHARED_UCHAR cru;
    SHARED_UCHAR row;
} HalfCluster;

#define HALF_CLUSTER_SIZE 14


#endif //!defined(SHARED_CLUSTER_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
