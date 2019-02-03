#pragma once

#include <CL/cl2.hpp>

#include <vector>


namespace gpucf
{

class ClEnv;


class StreamCompaction
{
public:
    void setup(ClEnv &, size_t); 

    int  enqueue(
            cl::CommandQueue, 
            cl::Buffer, 
            cl::Buffer, 
            cl::NDRange, 
            cl::NDRange);

private:
    cl::Kernel inclusiveScanStart;
    cl::Kernel inclusiveScanStep;
    cl::Kernel compactArr;

    cl::Buffer newIdx;
    cl::Buffer offsetBuf;

    std::vector<int> offsets;

    size_t digitNum = 0;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
