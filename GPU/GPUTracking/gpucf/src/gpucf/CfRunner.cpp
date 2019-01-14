#include "CfRunner.h"

#include <gpucf/GPUClusterFinder.h>


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

    cf.run(env, reader.get());

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
