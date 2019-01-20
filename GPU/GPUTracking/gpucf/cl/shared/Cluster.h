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

#if IS_CL_HOST
    Cluster_s()
    {
    }

    Cluster_s(int _cru, int _row, int _Q, int _QMax, 
            float _padMean, float _timeMean, float _padSigma, float _timeSigma)
        : cru(_cru)
        , row(_row)
        , Q(_Q)
        , QMax(_QMax)
        , padMean(_padMean)
        , timeMean(_timeMean)
        , padSigma(_padSigma)
        , timeSigma(_timeSigma)
    {
    }
#endif

} Cluster;


#endif //!defined(SHARED_CLUSTER_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
