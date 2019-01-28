#include "GPUAlgorithm.h"

#include <gpucf/common/log.h>


using namespace gpucf;


void GPUAlgorithm::setup(ClEnv &env, const DataSet &data)
{
    setupImpl(env, data); 
    isSetup = true;
}

GPUAlgorithm::Result GPUAlgorithm::run()
{
    ASSERT(isSetup); 
    return runImpl();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
