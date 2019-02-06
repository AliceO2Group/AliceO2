#include "StreamCompaction.h"

#include <gpucf/ClEnv.h>

#include <cmath>


using namespace gpucf;


void StreamCompaction::setup(ClEnv &env, size_t digitNum)
{
    this->digitNum = digitNum;

    cl::Program scprg = env.buildFromSrc("streamCompaction.cl");  

    inclusiveScanStart = cl::Kernel(scprg, "inclusiveScanStart");
    inclusiveScanStep  = cl::Kernel(scprg, "inclusiveScanStep");
    compactArr         = cl::Kernel(scprg, "compactArr");

    newIdxBufIn = cl::Buffer(env.getContext(), 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    newIdxBufOut = cl::Buffer(env.getContext(), 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    for (int i = 0; i < std::ceil(log2(digitNum)); i++)
    {
        offsets.push_back(1 << i);
    }
}

int StreamCompaction::enqueue(
        cl::CommandQueue queue,
        cl::Buffer digits,
        cl::Buffer predicate,
        bool debug)
{
    cl::NDRange global(digitNum);
    cl::NDRange local(64);

    inclusiveScanStart.setArg(0, predicate);
    inclusiveScanStart.setArg(1, newIdxBufIn);
    queue.enqueueNDRangeKernel(
            inclusiveScanStart,
            cl::NullRange,
            global,
            local);

    /* queue.enqueueFillBuffer(newIdxBufOut, 0, 0, sizeof(cl_int) * digitNum); */
    queue.enqueueCopyBuffer(
            newIdxBufIn,
            newIdxBufOut,
            0,
            0,
            sizeof(cl_int) * digitNum);

    for (int offset : offsets)
    {
        cl::NDRange itemOffset(offset);

        inclusiveScanStep.setArg(0, newIdxBufIn);
        inclusiveScanStep.setArg(1, newIdxBufOut);
        queue.enqueueNDRangeKernel(
                inclusiveScanStep,
                itemOffset,
                global,
                local);

        queue.enqueueCopyBuffer(
                newIdxBufOut,
                newIdxBufIn,
                0,
                0,
                sizeof(cl_int) * digitNum);

        if (debug)
        {
            newIdxDump.emplace_back(digitNum);
            queue.enqueueReadBuffer(
                    newIdxBufOut,
                    CL_TRUE,
                    0,
                    digitNum * sizeof(cl_int),
                    newIdxDump.back().data());
        }
    }

    compactArr.setArg(0, digits);
    compactArr.setArg(1, predicate);
    compactArr.setArg(2, newIdxBufOut);
    queue.enqueueNDRangeKernel(
            compactArr,
            cl::NullRange,
            global,
            local);

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
