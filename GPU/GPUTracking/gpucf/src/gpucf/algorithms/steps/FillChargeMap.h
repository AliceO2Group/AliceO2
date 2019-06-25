#pragma once

#include <gpucf/common/Kernel1D.h>



namespace gpucf
{

class FillChargeMap : protected Kernel1D
{

public:
    DECL_KERNEL(FillChargeMap, "fillChargeMap");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        kernel.setArg(0, state.digits);    
        kernel.setArg(1, state.chargeMap);    
        Kernel1D::call(0, state.digitnum, 64, queue);
    }

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

