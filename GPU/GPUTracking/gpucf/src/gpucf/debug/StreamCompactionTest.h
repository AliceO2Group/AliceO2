#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/gpu/StreamCompaction.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class ClEnv;


class StreamCompactionTest
{
public:
    StreamCompactionTest(ClEnv &);

    bool run();

private:
    static constexpr size_t N = 20;

    StreamCompaction streamCompaction;

    cl::CommandQueue queue;

    cl::Buffer digitsBuf;
    cl::Buffer predicateBuf;

    std::array<unsigned char, N> predicate;
    std::array<Digit        , N> digits;

    void setup(ClEnv &);

    void dumpResult(const std::vector<std::vector<int>> &);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
