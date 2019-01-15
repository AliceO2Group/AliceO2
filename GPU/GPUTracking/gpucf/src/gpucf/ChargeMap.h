#pragma once

#include <gpucf/cl.h>
#include <gpucf/common/Digit.h>


namespace gpucf
{

class ChargeMap
{

public:
    ChargeMap(cl::Context, cl::Program, const std::vector<Digit> &);

    cl::Buffer get() const
    {
        return *chargeMap;
    }

    void enqueueFill(cl::CommandQueue, cl::Buffer, cl::NDRange, cl::NDRange);

private:
    static size_t getNumOfRows(const std::vector<Digit> &);

    size_t chargeMapBytes;
    std::unique_ptr<cl::Buffer> chargeMap;
    
    std::unique_ptr<cl::Kernel> digitsToChargeMap;

    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
