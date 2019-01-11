#include "ClCanary.h"

#include <gpucf/VectorAdd.h>
#include <gpucf/log.h>


void ClCanary::setupFlags(args::Group &required, args::Group &optional) 
{
    envFlags = std::make_unique<ClEnv::Flags>(required, optional); 
}

int ClCanary::mainImpl() 
{
    ASSERT(envFlags != nullptr);

    ClEnv env(*envFlags);
    VectorAdd canary;

    bool ok = canary.run(env);

    if (!ok) 
    {
        log::Info() << "OpenCL not working as expected.";
        return 1;
    }

    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
