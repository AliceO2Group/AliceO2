#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/gpu/StreamCompaction.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class ClEnv;


class SCFuzzer
{
public:
    SCFuzzer(ClEnv &);

    bool run(size_t);

private:
    StreamCompaction streamCompaction;

    cl::Context context;
    cl::Device  device;

    cl::CommandQueue queue;

    cl::Buffer digitsInBuf;
    cl::Buffer digitsOutBuf;
    cl::Buffer predicateBuf;

    void setup(ClEnv &);

    void dumpResult(const std::vector<std::vector<int>> &);

    bool repeatTest(size_t, size_t);
    bool runTest(size_t);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
