#include "ClusterFinder.h"


using namespace gpucf;


ClusterFinder::ClusterFinder(
        ClusterFinderConfig cfg,
        size_t digitnum,
        ClEnv env)
    : queue(env.getContext(), env.getDevice(), CL_QUEUE_PROFILING_ENABLE)
    , state(cfg, digitnum, env.getContext(), env.getDevice())
    , compactPeaks(env, digitnum)
    , computeCluster(env.getProgram())
    , countPeaks(env.getProgram())
    , fillChargeMap(env.getProgram())
    , findPeaks(env.getProgram())
    , resetMaps(env.getProgram())
{
}

// vim: set ts=4 sw=4 sts=4 expandtab:
