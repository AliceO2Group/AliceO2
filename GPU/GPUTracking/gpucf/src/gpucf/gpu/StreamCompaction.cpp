#include "StreamCompaction.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>

#include <cmath>


using namespace gpucf;


StreamCompaction::Worker::Worker(
        cl::Program prg, 
        cl::Device device, 
        DeviceMemory mem)
    : nativeScanUp(prg, "nativeScanUp")
    , nativeScanTop(prg, "nativeScanTop")
    , nativeScanDown(prg, "nativeScanDown")
    , compactArr(prg, "compactArr")
    , mem(mem)
{
    nativeScanUp.getWorkGroupInfo(
            device,
            CL_KERNEL_WORK_GROUP_SIZE, 
            &scanUpWorkGroupSize);

    log::Debug() << "scanUpWorkGroupSize = " << scanUpWorkGroupSize;

    nativeScanTop.getWorkGroupInfo(
            device,
            CL_KERNEL_WORK_GROUP_SIZE,
            &scanTopWorkGroupSize);

    size_t scanDownWorkGroupSize;
    nativeScanDown.getWorkGroupInfo(
            device,
            CL_KERNEL_WORK_GROUP_SIZE,
            &scanDownWorkGroupSize);

    ASSERT(scanDownWorkGroupSize == scanUpWorkGroupSize);

    log::Debug() << "scanTopWorkGroupSize = " << scanTopWorkGroupSize;
}


void StreamCompaction::setup(ClEnv &env, size_t workernum, size_t digitnum)
{
    prg = env.buildFromSrc("streamCompaction.cl");  

    context = env.getContext();
    device = env.getDevice();

    setDigitNum(digitnum, workernum);
}

void StreamCompaction::setDigitNum(size_t digitnum, size_t workernum)
{
    this->digitNum = digitnum;

    mems = std::stack<DeviceMemory>();

    cl::Kernel scanUp(*prg, "nativeScanUp");
    size_t scanUpWorkGroupSize;
    scanUp.getWorkGroupInfo(
            device,
            CL_KERNEL_WORK_GROUP_SIZE, 
            &scanUpWorkGroupSize);

    cl::Kernel scanTop(*prg, "nativeScanTop");
    size_t scanTopWorkGroupSize;
    scanTop.getWorkGroupInfo(
            device,
            CL_KERNEL_WORK_GROUP_SIZE,
            &scanTopWorkGroupSize);

    for (size_t i = 0; i < workernum; i++)
    {
        DeviceMemory mem;

        size_t d = digitNum;
        for (; d > scanTopWorkGroupSize; 
               d = std::ceil(d / float(scanUpWorkGroupSize)) )
        {
            mem.incrBufs.emplace_back(
                    context,
                    CL_MEM_READ_WRITE,
                    sizeof(cl_int) * d);
            mem.incrBufSizes.push_back(d);
        }

        mem.incrBufs.emplace_back(
                context,
                CL_MEM_READ_WRITE,
                sizeof(cl_int) * d);
        mem.incrBufSizes.push_back(d);

        mems.push(mem);
    }
}

size_t StreamCompaction::Worker::run(
        const Fragment &range,
        cl::CommandQueue queue,
        cl::Buffer digits,
        cl::Buffer digitsOut,
        cl::Buffer predicate,
        bool debug)
{
    if (debug)
    {
        log::Info() << "StreamCompaction debug on";
    }

    sumsDump.clear();
    scanEvents.clear();

    ASSERT(!mem.incrBufs.empty());
    ASSERT(!mem.incrBufSizes.empty());
    ASSERT(mem.incrBufs.size() == mem.incrBufSizes.size());

    size_t digitnum = range.backlog + range.items;
    size_t offset = range.start;

    queue.enqueueCopyBuffer(
            predicate,
            mem.incrBufs.front(),
            sizeof(cl_int) * offset,
            sizeof(cl_int) * offset,
            sizeof(cl_int) * digitnum,
            nullptr,
            addScanEvent());

    std::vector<size_t> offsets;
    std::vector<size_t> digitnums;

    size_t stepnum = this->stepnum(range);
    DBG(stepnum);
    ASSERT(stepnum <= mem.incrBufs.size());
    for (size_t i = 1; i < stepnum; i++)
    {
        digitnums.push_back(digitnum);
        offsets.push_back(offset);

        ASSERT(mem.incrBufSizes[i-1] >= digitnum);

        nativeScanUp.setArg(0, mem.incrBufs[i-1]);
        nativeScanUp.setArg(1, mem.incrBufs[i]);
        queue.enqueueNDRangeKernel(
                nativeScanUp,
                cl::NDRange(offset),
                cl::NDRange(digitnum),
                cl::NDRange(scanUpWorkGroupSize),
                nullptr,
                addScanEvent());

        if (debug)
        {
            dumpBuffer(queue, mem.incrBufs[i-1], mem.incrBufSizes[i-1]);
        }

        offset = 0;
        digitnum /= scanUpWorkGroupSize;
    }

    if (debug)
    {
        dumpBuffer(queue, mem.incrBufs[stepnum-1], mem.incrBufSizes[stepnum-1]);
    }

    ASSERT(digitnum <= scanTopWorkGroupSize);

    nativeScanTop.setArg(0, mem.incrBufs[stepnum-1]);
    queue.enqueueNDRangeKernel(
            nativeScanTop,
            cl::NDRange(offset),
            cl::NDRange(digitnum),
            cl::NDRange(scanTopWorkGroupSize),
            nullptr,
            addScanEvent());

    if (debug)
    {
        dumpBuffer(queue, mem.incrBufs[stepnum-1], mem.incrBufSizes[stepnum-1]);
    }

    ASSERT(digitnums.size() == stepnum-1);
    ASSERT(offsets.size() == stepnum-1);
    for (size_t i = stepnum-1; i > 0; i--)
    {
        offset = offsets[i-1];
        digitnum = digitnums[i-1];

        ASSERT(digitnum > scanUpWorkGroupSize);

        nativeScanDown.setArg(0, mem.incrBufs[i-1]);
        nativeScanDown.setArg(1, mem.incrBufs[i]);
        queue.enqueueNDRangeKernel(
                nativeScanDown,
                cl::NDRange(offset + scanUpWorkGroupSize),
                cl::NDRange(digitnum - scanUpWorkGroupSize),
                cl::NDRange(scanUpWorkGroupSize),
                nullptr,
                addScanEvent());

        if (debug)
        {
            dumpBuffer(queue, mem.incrBufs[i-1], mem.incrBufSizes[i-1]);
        }
    }

    offset = offsets.front();
    digitnum = digitnums.front();

    compactArr.setArg(0, digits);
    compactArr.setArg(1, digitsOut);
    compactArr.setArg(2, predicate);
    compactArr.setArg(3, mem.incrBufs.front());
    queue.enqueueNDRangeKernel(
            compactArr,
            cl::NDRange(offset),
            cl::NDRange(digitnum),
            cl::NullRange,
            nullptr,
            compactArrEv.get());

    cl_int newDigitNum;

    // Read the last element from index buffer to get the new number of digits
    queue.enqueueReadBuffer(
            mem.incrBufs.front(),
            CL_TRUE,
            (range.start + digitnum-1) * sizeof(cl_int),
            sizeof(cl_int),
            &newDigitNum,
            nullptr,
            readNewDigitNum.get());


    return newDigitNum;
}

size_t StreamCompaction::Worker::stepnum(const Fragment &range) const
{
    size_t c = 0;
    size_t itemnum = range.backlog + range.items;

    while (itemnum > 0)
    {
        itemnum /= scanUpWorkGroupSize;
        c++;
    }

    return c;
}

std::vector<std::vector<int>> StreamCompaction::Worker::getNewIdxDump() const
{
    return sumsDump;
}

Step StreamCompaction::Worker::asStep(const std::string &name) const
{
    return {name, scanEvents.front().startMs(), readNewDigitNum.endMs()};
}

void StreamCompaction::Worker::dumpBuffer(
        cl::CommandQueue queue,
        cl::Buffer buf,
        size_t size)
{
    sumsDump.emplace_back(size); 
    queue.enqueueReadBuffer(
            buf,
            CL_FALSE,
            0,
            sizeof(cl_int) * size,
            sumsDump.back().data());
}

cl::Event *StreamCompaction::Worker::addScanEvent()
{
    scanEvents.emplace_back();
    return scanEvents.back().get();
}

StreamCompaction::Worker StreamCompaction::worker()
{
    ASSERT(!mems.empty());

    DeviceMemory mem = mems.top();
    mems.pop();

    return Worker(*prg, device, mem);
}



// vim: set ts=4 sw=4 sts=4 expandtab:
