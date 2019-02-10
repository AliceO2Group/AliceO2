#include "SCFuzzer.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>

#include <algorithm>


using namespace gpucf;


static_assert(sizeof(cl_int) == sizeof(int));


SCFuzzer::SCFuzzer(ClEnv &env)
{
    setup(env);
}

void SCFuzzer::setup(ClEnv &env)
{
    context = env.getContext();
    device  = env.getDevice();

    streamCompaction.setup(env, 10);
}

bool SCFuzzer::run(size_t runs)
{
    static const std::vector<size_t> digitNums = 
    {
        10,
        100,
        514,
    };

    for (size_t digitNum : digitNums)
    {
        if (repeatTest(digitNum, runs))
        {
            log::Debug() << "Test for size " << digitNum << ": OK";
        }
        else
        {
            log::Error() << "Test for size " << digitNum << ": Failed";
            return false;
        }
    }

    log::Success() << "StreamCompactionTest: OK";

    return true;
}

bool SCFuzzer::repeatTest(size_t digitNum, size_t runs)
{
    for (size_t i = 0; i < runs; i++)
    {
        if (!runTest(digitNum))
        {
            return false; 
        }
    }
    return true;
}

bool SCFuzzer::runTest(size_t N)
{
    streamCompaction.setDigitNum(N);

    std::vector<int> predicate(N);
    std::fill(predicate.begin(), predicate.end(), 1);

    std::vector<Digit> digitsIn(N);
    std::vector<Digit> digitsOut(N);

    size_t digitBytes     = sizeof(Digit) * N;
    size_t predicateBytes = sizeof(cl_int) * N;

    digitsInBuf = cl::Buffer(
                    context,
                    CL_MEM_READ_WRITE,
                    digitBytes);

    digitsOutBuf = cl::Buffer(
                    context,
                    CL_MEM_READ_WRITE,
                    digitBytes);

    predicateBuf = cl::Buffer(
                    context,
                    CL_MEM_READ_WRITE,
                    predicateBytes);

    queue = cl::CommandQueue(context, device);

    cl_command_queue_properties info;
    queue.getInfo(CL_QUEUE_PROPERTIES, &info);
    ASSERT(!(info & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE));

    queue.enqueueWriteBuffer(
            predicateBuf,
            CL_FALSE,
            0,
            predicateBytes,
            predicate.data());

    int res = streamCompaction.enqueue(
            queue,
            digitsInBuf,
            digitsOutBuf,
            predicateBuf,
            true);

    auto dump = streamCompaction.getNewIdxDump();
    if (res != static_cast<int>(N))
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

    return true;
}

void SCFuzzer::dumpResult(const std::vector<std::vector<int>> &dump)
{
    log::Error() << "Result dump: ";
    for (std::vector<int> interim : dump)
    {
        constexpr size_t maxPrintOut = 1000;
        size_t c = maxPrintOut;
        for (int idx : interim)
        {
            log::Error() << idx;
            c--;
            if (c == 0)
            {
                log::Error() << "...";
                break;
            }
        }
        log::Error();
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
