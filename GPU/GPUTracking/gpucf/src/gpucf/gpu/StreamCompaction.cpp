#include "StreamCompaction.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>

#include <cmath>


using namespace gpucf;


void StreamCompaction::setup(ClEnv &env, size_t digitNum)
{
    cl::Program scprg = env.buildFromSrc("streamCompaction.cl");  

    inclusiveScanStep  = cl::Kernel(scprg, "inclusiveScanStep");
    compactArr         = cl::Kernel(scprg, "compactArr");

    context = env.getContext();

    setDigitNum(digitNum);
}

void StreamCompaction::setDigitNum(size_t digitNum)
{
    this->digitNum = digitNum;

    newIdxBufIn = cl::Buffer(context, 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    newIdxBufOut = cl::Buffer(context, 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    inclusiveScanStep.setArg(0, newIdxBufIn);
    inclusiveScanStep.setArg(1, newIdxBufOut);

    offsets.clear();
    for (int i = 0; i < std::ceil(log2(digitNum)); i++)
    {
        ASSERT(i < int(sizeof(int)) * 8);
        offsets.push_back(1 << i);
    }
}

int StreamCompaction::enqueue(
        cl::CommandQueue queue,
        cl::Buffer digits,
        cl::Buffer digitsOut,
        cl::Buffer predicate,
        bool debug)
{
    log::Debug() << "Copying buffer to setup prefix sum";

    if (debug)
    {
        newIdxDump.clear();
    }

    queue.enqueueCopyBuffer(
            predicate,
            newIdxBufIn,
            0,
            0,
            sizeof(cl_int) * digitNum);
    /* queue.finish(); */

    queue.enqueueCopyBuffer(
            newIdxBufIn,
            newIdxBufOut,
            0,
            0,
            sizeof(cl_int) * digitNum);
    /* queue.finish(); */

    log::Debug() << "Starting prefix sum";

    cl::NDRange local(64);
    for (int offset : offsets)
    {
        log::Debug() << "Prefix sum step: offset = " << offset;
        cl::NDRange global(digitNum-offset);
        cl::NDRange itemOffset(offset);

        /* queue.finish(); */

        queue.enqueueNDRangeKernel(
                inclusiveScanStep,
                itemOffset,
                global,
                local);
        /* queue.finish(); */

        queue.enqueueCopyBuffer(
                newIdxBufOut,
                newIdxBufIn,
                0,
                0,
                sizeof(cl_int) * digitNum);
        /* queue.finish(); */

        if (debug)
        {
            newIdxDump.emplace_back(digitNum);
            queue.enqueueReadBuffer(
                    newIdxBufOut,
                    CL_FALSE,
                    0,
                    digitNum * sizeof(cl_int),
                    newIdxDump.back().data());
        }

        /* queue.finish(); */
    }

    compactArr.setArg(0, digits);
    compactArr.setArg(1, digitsOut);
    compactArr.setArg(2, predicate);
    compactArr.setArg(3, newIdxBufOut);
    queue.enqueueNDRangeKernel(
            compactArr,
            cl::NullRange,
            cl::NDRange(digitNum),
            local);
    /* queue.finish(); */

    cl_int newDigitNum;

    // Read the last element from index buffer to get the new number of digits
    queue.enqueueReadBuffer(
            newIdxBufOut, 
            CL_TRUE, 
            (digitNum-1) * sizeof(cl_int),
            sizeof(cl_int),
            &newDigitNum);

    return newDigitNum;
}

std::vector<std::vector<int>> StreamCompaction::getNewIdxDump() const
{
    return newIdxDump;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
