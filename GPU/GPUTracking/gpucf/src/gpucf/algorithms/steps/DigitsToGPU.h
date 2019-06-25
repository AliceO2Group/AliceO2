#pragma once

#include <gpucf/common/buffer.h>
#include <gpucf/common/Digit.h>

#include <nonstd/span.hpp>


namespace gpucf
{
    
class DigitsToGPU
{

public:

    void call(
            ClusterFinderState &state, 
            nonstd::span<const Digit> digits, 
            cl::CommandQueue queue)
    {
        state.digitnum = digits.size(); 

        gpucpy(digits, state.digits, digits.size(), queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
