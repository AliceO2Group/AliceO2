#pragma once

#include <gpucf/common/Event.h>
#include <gpucf/common/Measurements.h>

#include <CL/cl2.hpp>

#include <vector>


namespace gpucf
{

class ClEnv;


class StreamCompaction
{
public:
    void setup(ClEnv &, size_t); 

    void setDigitNum(size_t);

    int  enqueue(
            cl::CommandQueue,
            cl::Buffer,
            cl::Buffer,
            cl::Buffer,
            bool debug=false);

    std::vector<std::vector<int>> getNewIdxDump() const;

    Step asStep(const std::string &) const;

private:
    cl::Context context;

    cl::Kernel nativeScanUp;
    size_t     scanUpWorkGroupSize;

    cl::Kernel nativeScanTop;
    size_t     scanTopWorkGroupSize;

    cl::Kernel nativeScanDown;
    cl::Kernel compactArr;

    std::vector<cl::Buffer> incrBufs;
    std::vector<size_t>     incrBufSizes;

    std::vector<std::vector<int>> sumsDump;

    std::vector<Event> scanEvents;
    Event compactArrEv;

    size_t digitNum = 0;

    cl::Event *addScanEvent();

    void dumpBuffer(cl::CommandQueue, cl::Buffer, size_t);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
