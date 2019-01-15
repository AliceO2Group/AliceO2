#include "CfRunner.h"

#include <gpucf/ClusterChecker.h>
#include <gpucf/GPUClusterFinder.h>


using namespace gpucf;


void CfRunner::setupFlags(args::Group &required, args::Group &optional)
{
    envFlags = std::make_unique<ClEnv::Flags>(required, optional); 
    digitFlags = std::make_unique<DigitReader::Flags>(required, optional);
}

int CfRunner::mainImpl()
{
    ClEnv env(*envFlags); 

    DigitReader reader(*digitFlags);

    GPUClusterFinder cf;
    std::vector<Cluster> clusters = cf.run(env, reader.get());

    ClusterChecker checker;
    checker.verify(clusters);

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
