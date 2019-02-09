#include "StreamCompaction.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>

#include <cmath>


using namespace gpucf;


void StreamCompaction::setup(ClEnv &env, size_t digitNum)
{
    cl::Program scprg = env.buildFromSrc("streamCompaction.cl");  

    nativeScanUp   = cl::Kernel(scprg, "nativeScanUp");
    nativeScanUp.getWorkGroupInfo(
            env.getDevice(),
            CL_KERNEL_WORK_GROUP_SIZE, 
            &scanUpWorkGroupSize);

    nativeScanTop  = cl::Kernel(scprg, "nativeScanTop");
    nativeScanTop.getWorkGroupInfo(
            env.getDevice(),
            CL_KERNEL_WORK_GROUP_SIZE,
            &scanTopWorkGroupSize);

    nativeScanDown = cl::Kernel(scprg, "nativeScanDown");
    
    compactArr = cl::Kernel(scprg, "compactArr");

    context = env.getContext();

    setDigitNum(digitNum);
}

void StreamCompaction::setDigitNum(size_t digitNum)
{
    this->digitNum = digitNum;

    sumsBuf = cl::Buffer(context, 
                        CL_MEM_READ_WRITE, 
                        sizeof(cl_int) * digitNum);

    size_t d = std::ceil(digitNum / float(scanUpWorkGroupSize));
    while (d > scanTopWorkGroupSize)
    {
        incrBufs.emplace_back(
                context,
                CL_MEM_READ_WRITE,
                sizeof(cl_int) * d);
        incrBufSizes.push_back(d);
        d = std::ceil(d / float(scanUpWorkGroupSize));    
    }

    incrBufs.emplace_back(
            context,
            CL_MEM_READ_WRITE,
            sizeof(cl_int) * d);
    incrBufSizes.push_back(d);
}

int StreamCompaction::enqueue(
        cl::CommandQueue queue,
        cl::Buffer digits,
        cl::Buffer digitsOut,
        cl::Buffer predicate,
        bool debug)
{
    if (debug)
    {
        log::Info() << "StreamCompaction debug on";
        sumsDump.clear();
    }

    scanEvents.clear();

    queue.enqueueCopyBuffer(
            predicate,
            sumsBuf,
            0,
            0,
            sizeof(cl_int) * digitNum,
            nullptr,
            addScanEvent());


    for (size_t i = 0; i < incrBufs.size(); i++)
    {
        nativeScanUp.setArg(0, (i == 0) ? sumsBuf : incrBufs[i-1]);
        nativeScanUp.setArg(1, incrBufs[i]);

        cl::NDRange workItems((i == 0) ? digitNum : incrBufSizes[i-1]);
        queue.enqueueNDRangeKernel(
                nativeScanUp,
                cl::NullRange,
                workItems,
                cl::NDRange(scanUpWorkGroupSize),
                nullptr,
                addScanEvent());
    }

    nativeScanTop.setArg(0, incrBufs.back());
    queue.enqueueNDRangeKernel(
            nativeScanTop,
            cl::NullRange,
            cl::NDRange(scanTopWorkGroupSize),
            cl::NDRange(scanTopWorkGroupSize),
            nullptr,
            addScanEvent());

    for (size_t i = incrBufs.size()-1;; i--)
    {
        nativeScanDown.setArg(0, (i == 0) ? sumsBuf : incrBufs[i-1]);
        nativeScanDown.setArg(1, incrBufs[i]);

        cl::NDRange workItems((i == 0) ? digitNum : incrBufSizes[i-1]);
        queue.enqueueNDRangeKernel(
                nativeScanDown,
                cl::NullRange,
                workItems,
                cl::NullRange,
                nullptr,
                addScanEvent());

        if (i == 0)
        {
            break;
        }
    }

    compactArr.setArg(0, digits);
    compactArr.setArg(1, digitsOut);
    compactArr.setArg(2, predicate);
    compactArr.setArg(3, sumsBuf);
    queue.enqueueNDRangeKernel(
            compactArr,
            cl::NullRange,
            cl::NDRange(digitNum),
            cl::NullRange,
            nullptr,
            compactArrEv.get());

    cl_int newDigitNum;

    // Read the last element from index buffer to get the new number of digits
    queue.enqueueReadBuffer(
            sumsBuf,
            CL_TRUE, 
            (digitNum-1) * sizeof(cl_int),
            sizeof(cl_int),
            &newDigitNum,
            nullptr,
            addScanEvent());

    return newDigitNum;
}

std::vector<std::vector<int>> StreamCompaction::getNewIdxDump() const
{
    return sumsDump;
}

float StreamCompaction::executionTimeMs() const
{
    return scanTimeMs() + compactionTimeMs();
}

float StreamCompaction::scanTimeMs() const
{
    float time = 0;

    for (const Event &ev : scanEvents)
    {
        time += ev.executionTimeMs();
    }

    return time;
}

float StreamCompaction::compactionTimeMs() const
{
    return compactArrEv.executionTimeMs();
}

cl::Event *StreamCompaction::addScanEvent()
{
    scanEvents.emplace_back();
    return scanEvents.back().get();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
