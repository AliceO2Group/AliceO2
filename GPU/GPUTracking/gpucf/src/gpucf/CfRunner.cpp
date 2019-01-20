#include "CfRunner.h"

#include <gpucf/ClusterChecker.h>
#include <gpucf/GPUClusterFinder.h>


using namespace gpucf;


void CfRunner::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags     = std::make_unique<ClEnv::Flags>(required, optional); 
    digitFlags   = std::make_unique<DigitReader::Flags>(required, optional);
    clusterFlags = std::make_unique<ClusterReader::Flags>(required, optional);
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DigitReader dreader(*digitFlags);
    ClusterReader creader(*clusterFlags);

    GPUClusterFinder cf;
    std::vector<Cluster> clusters = cf.run(env, dreader.get());

    ClusterChecker checker;
    checker.verify(clusters, creader.get());

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
