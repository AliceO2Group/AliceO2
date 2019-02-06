#include "StreamCompactionTest.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>

#include <array>


using namespace gpucf;


StreamCompactionTest::StreamCompactionTest(ClEnv &env)
{
    setup(env);
}

void StreamCompactionTest::setup(ClEnv &env)
{
    streamCompaction.setup(env, N);

    digitsBuf = cl::Buffer(
                    env.getContext(),
                    CL_MEM_READ_WRITE,
                    sizeof(Digit) * N);

    predicateBuf = cl::Buffer(
                    env.getContext(),
                    CL_MEM_READ_WRITE,
                    sizeof(cl_uchar) * N);

    static_assert(sizeof(cl_uchar) == sizeof(unsigned char));
    predicate.fill(1);

    queue = cl::CommandQueue(env.getContext(), env.getDevice());
}


bool StreamCompactionTest::run()
{
    queue.enqueueWriteBuffer(
            predicateBuf,
            CL_FALSE,
            0,
            sizeof(cl_uchar) * N,
            predicate.data());

    int res = streamCompaction.enqueue(queue, digitsBuf, predicateBuf, true);

    auto dump = streamCompaction.getNewIdxDump();
    if (res != N)
    {
        log::Error() << "StreamCompaction: got " << res 
            << " elements, but expected " << N;

        dumpResult(dump);

        return false;
    }

    std::vector<int> newIdx = dump.back();

    if (newIdx.size() != N)
    {
        log::Error() << "StreamCompaction: result buffer has wrong size.";
        return false;
    }
    
    for (size_t i = 0; i < newIdx.size(); i++)
    {
        if (newIdx[i] != static_cast<int>(i+1))
        {
            log::Error() << "StreamCompaction: found wrong indice."; 
            dumpResult(dump);
            return false;
        }
    }

    // TODO: test if array is actually compacted...

    log::Success() << "StreamCompactionTest: OK";

    return true;
}

void StreamCompactionTest::dumpResult(const std::vector<std::vector<int>> &dump)
{
    log::Error() << "Result dump: ";
    for (std::vector<int> interim : dump)
    {
        for (int idx : interim)
        {
            log::Error() << idx;
        }
        log::Error();
    }

    log::Error() << "Input predicate dump: ";
    for (unsigned char p : predicate)
    {
        log::Error() << int(p);
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
