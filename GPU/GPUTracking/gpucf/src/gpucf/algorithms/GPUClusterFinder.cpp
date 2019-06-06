#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/DigitDivider.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>

#include <shared/tpc.h>

#include <cmath>
#include <functional>


using namespace gpucf;

using nonstd::optional;
using nonstd::nullopt;


const ClusterFinderConfig GPUClusterFinder::defaultConfig;


GPUClusterFinder::Worker::Worker(
        cl::Context context,
        cl::Device  device,
        cl::Program program,
        DeviceMemory mem,
        ClusterFinderConfig config,
        StreamCompaction::Worker streamCompaction,
        Worker *prev)
    : mem(mem)
    , streamCompaction(streamCompaction)
    , config(config)
{

    log::Debug() << "create clustering queue.";
    clustering = cl::CommandQueue(
            context, 
            device, 
            CL_QUEUE_PROFILING_ENABLE);

    log::Debug() << "create cleanup queue.";
    cleanup    = cl::CommandQueue(
            context, 
            device, 
            CL_QUEUE_PROFILING_ENABLE);
    log::Debug() << "finished creating cleanup queue.";

    fillChargeMap    = cl::Kernel(program, "fillChargeMap");
    findPeaks        = cl::Kernel(program, "findPeaks");
    countPeaks       = cl::Kernel(program, "countPeaks");
    computeClusters  = cl::Kernel(program, "computeClusters");
    resetMaps        = cl::Kernel(program, "resetMaps");

    if (prev != nullptr)
    {
        this->prev = Backwards();
        prev->next = Forwards(); 

        Backwards &slot = *this->prev;
        Forwards  &plug = *prev->next;

        slot.digitsToDevice = plug.digitsToDevice.get_future();
        slot.fillingChargeMap = plug.fillingChargeMap.get_future();
        slot.clusters = plug.clusters.get_future();
        plug.computeClusters = slot.computeClusters.get_future();
    }
}

template<class DigitT>
void GPUClusterFinder::Worker::runAndCatch(
        const Fragment &range,
        nonstd::span<const DigitT> digits,
        nonstd::span<Cluster> clusters)
{
    try 
    {
        run(range, digits, clusters);
    }
    catch(const cl::Error &err)
    {
        log::Error() << "Caught cl::Error: " << err.what() 
                     << "(" << log::clErrToStr(err.err()) << ")";
        throw err;
    }

}

template<class DigitT>
void GPUClusterFinder::Worker::run(
        const Fragment &range,
        nonstd::span<const DigitT> digits,
        nonstd::span<Cluster> clusters)
{
    if (prev)
    {
        std::vector<cl::Event> ev = {prev->digitsToDevice.get()};
        clustering.enqueueBarrierWithWaitList(&ev);
    }

    /*************************************************************************
     * Copy digits to device
     ************************************************************************/

    const size_t sizeofDigit = sizeof(DigitT);

    nonstd::span<const DigitT> toCopy = digits.subspan(
            range.start + range.backlog, 
            range.items + range.future);

    log::Debug() << "Sending " << toCopy.size() * sizeofDigit << " bytes to device";

    clustering.enqueueWriteBuffer(
            mem.digits,
            CL_FALSE,
            (range.start + range.backlog) * sizeofDigit,
            toCopy.size() * sizeofDigit,
            toCopy.data(),
            nullptr,
            digitsToDevice.get());

    if (next)
    {
        next->digitsToDevice.set_value(*digitsToDevice.get());
    }


    /*************************************************************************
     * Write digits to chargeMap
     ************************************************************************/

    cl::NDRange local(64);

    DBG(range.start);
    DBG(toCopy.size());

    fillChargeMap.setArg(0, mem.digits);
    fillChargeMap.setArg(1, mem.chargeMap);
    clustering.enqueueNDRangeKernel(
            fillChargeMap,
            cl::NDRange(range.start + range.backlog),
            cl::NDRange(toCopy.size()),
            local,
            nullptr,
            fillingChargeMap.get());

    if (next)
    {
        next->fillingChargeMap.set_value(*fillingChargeMap.get());
    }

    if (prev)
    {
        std::vector<cl::Event> ev = {prev->fillingChargeMap.get()};
        clustering.enqueueBarrierWithWaitList(&ev);
    }


    /*************************************************************************
     * Look for peaks
     ************************************************************************/

    DBG(range.start) 
    DBG(range.backlog + range.items);

    size_t maybePeaksNum = range.backlog + range.items;
    bool scratchpad = (config.clusterbuilder == ClusterBuilder::ScratchPad);
    size_t dummyItems = (scratchpad) ? 64 - (maybePeaksNum % 64) : 0;

    size_t peakfinderWorkitems = maybePeaksNum + dummyItems;

    findPeaks.setArg(0, mem.chargeMap);
    findPeaks.setArg(1, mem.digits);
    findPeaks.setArg(2, static_cast<cl_uint>(maybePeaksNum));
    findPeaks.setArg(3, mem.isPeak);
    findPeaks.setArg(4, mem.peakMap);
    clustering.enqueueNDRangeKernel(
            findPeaks,
            cl::NDRange(range.start),
            cl::NDRange(peakfinderWorkitems),
            local,
            nullptr,
            findingPeaks.get());


    /*************************************************************************
     * Compact peaks
     ************************************************************************/

    size_t clusternum = streamCompaction.run(
            range,
            clustering,
            mem.digits,
            mem.peaks,
            mem.isPeak);

    DBG(clusternum);
    ASSERT(clusternum <= range.backlog + range.items);
    

    /*************************************************************************
     * Count peaks around digits
     ************************************************************************/

    if (config.splitCharges)
    {
        countPeaks.setArg(0, mem.peakMap);
        countPeaks.setArg(1, mem.digits);
        countPeaks.setArg(2, static_cast<cl_uint>(maybePeaksNum));
        countPeaks.setArg(3, mem.isPeak);
        countPeaks.setArg(4, mem.peakCountMap);
        clustering.enqueueNDRangeKernel(
                countPeaks,
                cl::NDRange(range.start),
                cl::NDRange(peakfinderWorkitems),
                local,
                nullptr,
                countingPeaks.get());
    }


    /*************************************************************************
     * Compute cluster
     ************************************************************************/

    if (clusternum > 0)
    {
        size_t dummyItems = (scratchpad) ?  64 - (clusternum % 64) : 0;
                    
        cl::NDRange worksize = cl::NDRange(clusternum + dummyItems);
        cl::NDRange offset = cl::NDRange(range.start);

        computeClusters.setArg(0, mem.chargeMap);
        computeClusters.setArg(1, mem.peaks);
        computeClusters.setArg(2, mem.peakCountMap);
        computeClusters.setArg(3, mem.globalToLocalRow);
        computeClusters.setArg(4, mem.globalRowToCru);
        computeClusters.setArg(5, cl_uint(clusternum));
        computeClusters.setArg(6, mem.cluster);
        computeClusters.setArg(7, mem.peakMap);
        clustering.enqueueNDRangeKernel(
                computeClusters,
                offset,
                worksize,
                local,
                nullptr,
                computingClusters.get());
    }
    
    if (prev)
    {
        prev->computeClusters.set_value(*computingClusters.get());
    }


    /*************************************************************************
     * Reset charge map
     ************************************************************************/

    if (next)
    {
        std::vector<cl::Event> ev = {next->computeClusters.get()};
        clustering.enqueueBarrierWithWaitList(&ev);
    }

    resetMaps.setArg(0, mem.digits);
    resetMaps.setArg(1, mem.chargeMap);
    resetMaps.setArg(2, mem.peakCountMap);
    clustering.enqueueNDRangeKernel(
            resetMaps,
            cl::NDRange(range.start),
            cl::NDRange(range.backlog + range.items),
            local,
            nullptr,
            zeroChargeMap.get());


    /*************************************************************************
     * Copy cluster to host
     ************************************************************************/

    ASSERT(clusters.size() == digits.size());

    log::Info() << "Copy results back...";

    size_t clusterstart = (prev == nullopt) ? 0 : prev->clusters.get();

    DBG(clusterstart);
    DBG(clusternum);
    DBG(clusters.size());
    nonstd::span<Cluster> myCluster = clusters.subspan(clusterstart, clusternum);

    if (clusternum > 0)
    {
        clustering.enqueueReadBuffer(
                mem.cluster, 
                CL_TRUE,
                range.start * sizeof(Cluster),
                clusternum  * sizeof(Cluster), 
                myCluster.data(),
                nullptr,
                clustersToHost.get());
    }

    clusterend = clusterstart + clusternum;

    if (next)
    {
        next->clusters.set_value(clusterend);
    }

}

template<class DigitT>
void GPUClusterFinder::Worker::dispatch(
        const Fragment &fragment,
        nonstd::span<const DigitT> digits,
        nonstd::span<Cluster> clusters)
{
    myThread = std::thread(
            std::bind(
                &GPUClusterFinder::Worker::runAndCatch<DigitT>, 
                this, 
                fragment, 
                digits, 
                clusters));
}

void GPUClusterFinder::Worker::join()
{
    myThread.join();
    clustering.finish();
}


std::vector<Digit> GPUClusterFinder::getPeaks() const
{
    return peaks;
}

void GPUClusterFinder::setup(
        ClusterFinderConfig        conf, 
        ClEnv                     &env, 
        nonstd::span<const Digit>  digits)
{
    ASSERT(conf.chunks > 0);

    this->config = conf;
    this->digits = digits;

    if (config.usePackedDigits)
    {
        fillPackedDigits(); 
    }

    addDefines(env);

    streamCompaction.setup(env, config.chunks, digits.size());

    context = env.getContext(); 
    device  = env.getDevice();

    /*************************************************************************
     * Allocate clusters output
     ************************************************************************/

    clusters.resize(digits.size());


    /*************************************************************************
     * Create Buffer
     ************************************************************************/

    size_t digitsBufSize = 
        ((config.usePackedDigits) ? sizeof(PackedDigit) : sizeof(Digit)) 
        * digits.size();
    mem.digits = cl::Buffer(
            context,
            CL_MEM_READ_WRITE,
            digitsBufSize);

    mem.peaks = cl::Buffer(
            context,
            CL_MEM_READ_WRITE,
            digitsBufSize);

    std::vector<int> globalToLocalRow = RowInfo::instance().globalToLocalMap;
    const size_t numOfRows = globalToLocalRow.size();
    size_t globalToLocalRowBufSize = sizeof(cl_int) * numOfRows;
    mem.globalToLocalRow = cl::Buffer(
            context,
            CL_MEM_READ_ONLY,
            globalToLocalRowBufSize);

    std::vector<int> globalRowToCru = RowInfo::instance().globalRowToCruMap;
    size_t globalRowToCruBufSize = sizeof(cl_int) * numOfRows;
    mem.globalRowToCru = cl::Buffer(
            context,
            CL_MEM_READ_ONLY,
            globalRowToCruBufSize);

    size_t isPeakBufSize = digits.size() * sizeof(cl_uchar);
    mem.isPeak = cl::Buffer(
            context, 
            CL_MEM_READ_WRITE, 
            isPeakBufSize);

    size_t clusterBufSize = digits.size() * sizeof(Cluster);
    mem.cluster = cl::Buffer(
            context,
            CL_MEM_WRITE_ONLY,
            clusterBufSize);

    log::Info() << "Found " << numOfRows << " rows";

    size_t chargeSize = 
        (config.halfPrecisionCharges) ? sizeof(cl_half)
                                      : sizeof(cl_float);

    size_t mapEntries = numOfRows 
        * TPC_PADS_PER_ROW_PADDED * TPC_MAX_TIME_PADDED;

    size_t chargeMapSize  = mapEntries * chargeSize;
    mem.chargeMap = cl::Buffer(context, CL_MEM_READ_WRITE, chargeMapSize);

    size_t peakMapSize = mapEntries * sizeof(cl_uchar);
    mem.peakMap      = cl::Buffer(context, CL_MEM_READ_WRITE, peakMapSize);
    mem.peakCountMap = cl::Buffer(context, CL_MEM_READ_WRITE, peakMapSize);


    /************************************************************************
     * Create worker
     ***********************************************************************/

    workers.clear();

    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    for (size_t i = 0; i < config.chunks; i++)
    {
        workers.emplace_back(
                context,
                device,
                cfprg,
                mem,
                config,
                streamCompaction.worker(),
                (i == 0) ? nullptr : &workers.back());
    }

    
    /*************************************************************************
     * Init constant data
     ************************************************************************/

    log::Debug() << "creating init queue.";
    cl::CommandQueue initQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    log::Debug() << "finished creating init queue.";

    ASSERT(globalToLocalRowBufSize > 0);
    initQueue.enqueueWriteBuffer(
            mem.globalToLocalRow, 
            CL_FALSE, 
            0, 
            globalToLocalRowBufSize, 
            globalToLocalRow.data());

    ASSERT(globalRowToCruBufSize > 0);
    initQueue.enqueueWriteBuffer(
            mem.globalRowToCru, 
            CL_FALSE, 
            0, 
            globalRowToCruBufSize, 
            globalRowToCru.data());

    ASSERT(chargeMapSize > 0);
    initQueue.enqueueFillBuffer(
            mem.chargeMap, 
            0.0f, 
            0, 
            chargeMapSize);

    ASSERT(peakMapSize > 0);
    initQueue.enqueueFillBuffer(
            mem.peakMap,
            cl_uchar(0),
            0,
            peakMapSize);

    initQueue.enqueueFillBuffer(
            mem.peakCountMap,
            cl_uchar(1),
            0,
            peakMapSize);

    initQueue.finish();
}

GPUClusterFinder::Result GPUClusterFinder::run()
{
    static_assert(sizeof(cl_int) == sizeof(int));

    log::Info() << "Looking for clusters...";


    /*************************************************************************
     * Create fragments and run workers
     ************************************************************************/

    DigitDivider divider(digits, config.chunks);

    for (Worker &worker : workers)
    {
        optional<Fragment> fragment = divider.nextChunk(PADDING);

        ASSERT(fragment.has_value());

        log::Debug() << "got new fragment: {start: " << fragment->start
                     << ", backlog: " << fragment->backlog
                     << ", items: " << fragment->items
                     << ", future: " << fragment->future << "}";

        if (config.usePackedDigits)
        {
            worker.dispatch<PackedDigit>(
                    *fragment, 
                    nonstd::span<const PackedDigit>(packedDigits),
                    clusters);
        }
        else
        {
            worker.dispatch(*fragment, digits, clusters);
        }

    }

    for (Worker &worker : workers)
    {
        worker.join();
    }


    /*************************************************************************
     * Compute clusters
     ************************************************************************/

    size_t clusterNum = workers.back().clusterend;

    printClusters(clusters, 10);

    std::vector<Cluster> result(clusterNum);

    memcpy(result.data(), clusters.data(), clusterNum * sizeof(Cluster));

    log::Info() << "Found " << result.size() << " clusters.";

    std::vector<Step> steps;
    for (size_t i = 0; i < workers.size(); i++)
    {
        std::vector<Step> lane = toLane(i, workers[i]); 

        steps.insert(steps.end(), lane.begin(), lane.end());
    }

    return Result{result, steps};
}


void GPUClusterFinder::printClusters(
        const std::vector<Cluster> &clusters,
        size_t maxClusters)
{
    log::Debug() << "Printing found clusters";
    for (size_t i = 0; i < clusters.size(); i++)
    {
        log::Debug() << clusters[i];
        maxClusters--;
        if (maxClusters == 0)
        {
            break;
        }
    }
}

std::vector<Cluster> GPUClusterFinder::filterCluster(
        const std::vector<int> &isCenter,
        const std::vector<Cluster> &clusters)
{
    std::vector<Cluster> actualClusters; 

    for (size_t i = 0; i < clusters.size(); i++)
    {
        if (isCenter[i])
        {
            actualClusters.push_back(clusters[i]);
        }
    }

    return actualClusters;
}

std::vector<Digit> GPUClusterFinder::compactDigits(
        const std::vector<int> &isCenter,
        const std::vector<Digit> &digits)
{
    std::vector<Digit> peaks;    

    for (size_t i = 0; i < digits.size(); i++)
    {
        if (isCenter[i])
        {
            peaks.push_back(digits[i]);
        }
    }

    return peaks;
}

void GPUClusterFinder::fillPackedDigits()
{
    for (const Digit &digit : digits)
    {
        packedDigits.push_back(digit.toPacked());
    }
}

void GPUClusterFinder::addDefines(ClEnv &env)
{
    switch (config.layout)
    {
    case ChargemapLayout::TimeMajor: 
        break;
    case ChargemapLayout::PadMajor: 
        env.addDefine("CHARGEMAP_PAD_MAJOR_LAYOUT");
        break;
    case ChargemapLayout::Tiling4x4:
        env.addDefine("CHARGEMAP_4x4_TILING_LAYOUT");
        break;
    case ChargemapLayout::Tiling4x8:
        env.addDefine("CHARGEMAP_4x8_TILING_LAYOUT");
        break;
    case ChargemapLayout::Tiling8x4:
        env.addDefine("CHARGEMAP_8x4_TILING_LAYOUT");
        break;
    }

    switch (config.clusterbuilder)
    {
    case ClusterBuilder::Naive:
        break;
    case ClusterBuilder::ScratchPad:
        env.addDefine("BUILD_CLUSTER_SCRATCH_PAD");
        break;
    }

    if (config.usePackedDigits)
    {
        env.addDefine("USE_PACKED_DIGIT");
    }

    if (config.halfPrecisionCharges)
    {
        env.addDefine("CHARGEMAP_TYPE_HALF");
    }

    if (config.splitCharges)
    {
        env.addDefine("SPLIT_CHARGES");
    }
}


std::vector<Step> GPUClusterFinder::toLane(size_t id, const Worker &p)
{
    std::vector<Step> steps = {
        {"digitsToDevice", p.digitsToDevice},
        {"fillChargeMap", p.fillingChargeMap},
        {"findPeaks", p.findingPeaks},
        p.streamCompaction.asStep("compactPeaks"),
        {"computeCluster", p.computingClusters},
        {"resetMaps", p.zeroChargeMap},
        {"clusterToHost", p.clustersToHost},
    };

    for (Step &step : steps)
    {
        step.lane = id;
    }

    return steps;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
