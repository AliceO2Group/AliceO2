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

    newIdx = cl::Buffer(env.getContext(), 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    offsetBuf = cl::Buffer(env.getContext(),
                        CL_MEM_READ_ONLY,
                        sizeof(cl_int));

    for (int i = 0; i < std::ceil(log2(digitNum)) - 1; i++)
    {
        offsets.push_back(1 << i);
    }
}

int StreamCompaction::enqueue(
        cl::CommandQueue queue,
        cl::Buffer digits,
        cl::Buffer predicate,
        cl::NDRange global,
        cl::NDRange local)
{
    inclusiveScanStart.setArg(0, predicate);
    inclusiveScanStart.setArg(1, newIdx);

    queue.enqueueNDRangeKernel(
            inclusiveScanStart,
            cl::NullRange,
            global,
            local);

    for (const int &offset : offsets)
    {
        queue.enqueueWriteBuffer(
                offsetBuf,
                CL_FALSE,
                0,
                sizeof(cl_int),
                &offset);

        cl::NDRange itemOffset(offset);
        // TODO: set kernel args!!
        queue.enqueueNDRangeKernel(
                inclusiveScanStep,
                itemOffset,
                global,
                local);
    }

    compactArr.setArg(0, digits);
    compactArr.setArg(1, predicate);
    compactArr.setArg(2, newIdx);
    queue.enqueueNDRangeKernel(
            compactArr,
            cl::NullRange,
            global,
            local);

    cl_int newDigitNum;

    // Read the last element from index buffer to get the new number of digits
    queue.enqueueReadBuffer(
            newIdx, 
            CL_TRUE, 
            (digitNum-1) * sizeof(cl_int),
            sizeof(cl_int),
            &newDigitNum);

    return newDigitNum;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
