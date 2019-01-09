#if !defined(SHARED_CLUSTER_H)
#    define  SHARED_CLUSTER_H

#include "types.h"

typedef struct Cluster_s
{
    SHARED_INT cru;
    SHARED_INT row;
    SHARED_INT Q;
    SHARED_INT QMax;
    SHARED_FLOAT padMean;
    SHARED_FLOAT timeMean;
    SHARED_FLOAT padSigma;
    SHARED_FLOAT timeSigma;
} Cluster;

#endif //!defined(SHARED_CLUSTER_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
