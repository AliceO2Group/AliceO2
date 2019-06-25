#include "FillChargeMap.h"


using namespace gpucf;


FillChargeMap::FillChargeMap(cl::Program prg)
    : Kernel1D("fillChargeMap", prg)
{
}

void FillChargeMap::call(ClusterFinderState &state)

// vim: set ts=4 sw=4 sts=4 expandtab:
