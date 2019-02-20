#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/DigitDivider.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>

#include <shared/tpc.h>


using namespace gpucf;

using nonstd::optional;


const GPUClusterFinder::Config GPUClusterFinder::defaultConfig;


GPUClusterFinder::Plan::Plan(
        cl::Context context,
        cl::Device  device,
        cl::Program program,
        size_t i)
    : id(i)
{
    clustering = cl::CommandQueue(
            context, 
            device, 
            CL_QUEUE_PROFILING_ENABLE);

    cleanup    = cl::CommandQueue(
            context, 
            device, 
            CL_QUEUE_PROFILING_ENABLE);

    fillChargeMap    = cl::Kernel(program, "fillChargeMap");
    resetChargeMap   = cl::Kernel(program, "resetChargeMap");
    findPeaks        = cl::Kernel(program, "findPeaks");
    computeClusters  = cl::Kernel(program, "computeClusters");
}


/* Lane GPUClusterFinder::Worker::finish() */
/* { */
/*     clustering.finish(); */
/*     cleanup.finish(); */

/*     Lane measurements = */
/*     { */
/*         {"digitsToDevice", digitsToDevice}, */
/*         {"fillChargeMap", fillingChargeMap}, */
/*         {"findPeaks", findingPeaks}, */
/*         parent.streamCompaction.asStep("compactPeaks"), */
/*         {"computeClusters", computingClusters}, */
/*         {"resetChargeMap", zeroChargeMap}, */
/*         {"clustersToHost", clustersToHost}, */
/*     }; */

/*     return measurements; */
/* } */

std::vector<Digit> GPUClusterFinder::getPeaks() const
{
    return peaks;
}

void GPUClusterFinder::setup(Config conf, ClEnv &env, nonstd::span<const Digit> digits)
{
    ASSERT(conf.chunks > 0);

    this->config = conf;
    this->digits = digits;

    if (config.usePackedDigits)
    {
        fillPackedDigits(); 
    }

    addDefines(env);

    streamCompaction.setup(env, digits.size());

    context = env.getContext(); 
    device  = env.getDevice();

    
    /************************************************************************
     * Create execution plans
     ***********************************************************************/

    plans.clear();

    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    for (size_t i = 0; i < config.chunks; i++)
    {
        plans.emplace_back(context, device, cfprg, i);
    }

    
    /*************************************************************************
     * Allocate clusters output
     ************************************************************************/

    clusters.reserve(digits.size());


    /*************************************************************************
     * Create Buffer
     ************************************************************************/

    digitsBufSize = 
        ((config.usePackedDigits) ? sizeof(PackedDigit) : sizeof(Digit)) 
        * digits.size();
    digitsBuf = cl::Buffer(context,
            CL_MEM_READ_WRITE,
            digitsBufSize);

    peaksBuf = cl::Buffer(context,
            CL_MEM_READ_WRITE,
            digitsBufSize);

    std::vector<int> globalToLocalRow = RowInfo::instance().globalToLocalMap;
    const size_t numOfRows = globalToLocalRow.size();
    globalToLocalRowBufSize = sizeof(cl_int) * numOfRows;
    globalToLocalRowBuf = cl::Buffer(
            context,
            CL_MEM_READ_ONLY,
            globalToLocalRowBufSize);

    std::vector<int> globalRowToCru = RowInfo::instance().globalRowToCruMap;
    globalRowToCruBufSize = sizeof(cl_int) * numOfRows;
    globalRowToCruBuf = cl::Buffer(
            context,
            CL_MEM_READ_ONLY,
            globalRowToCruBufSize);

    isPeakBufSize = digits.size() * sizeof(cl_int);
    isPeakBuf = cl::Buffer(
            context, 
            CL_MEM_READ_WRITE, 
            isPeakBufSize);

    clusterBufSize = digits.size() * sizeof(Cluster);
    clusterBuf = cl::Buffer(
            context,
            CL_MEM_WRITE_ONLY,
            clusterBufSize);

    log::Info() << "Found " << numOfRows << " rows";

    chargeMapSize  = 
        numOfRows * TPC_PADS_PER_ROW_PADDED 
        * TPC_MAX_TIME_PADDED 
        * sizeof(cl_float);
    chargeMap = cl::Buffer(context, CL_MEM_READ_WRITE, chargeMapSize);

    
    /*************************************************************************
     * Init constant data
     ************************************************************************/

    cl::CommandQueue initQueue(context, device);

    ASSERT(globalToLocalRowBufSize > 0);
    initQueue.enqueueWriteBuffer(
            globalToLocalRowBuf, 
            CL_FALSE, 
            0, 
            globalToLocalRowBufSize, 
            globalToLocalRow.data());

    ASSERT(globalRowToCruBufSize > 0);
    initQueue.enqueueWriteBuffer(
            globalRowToCruBuf, 
            CL_FALSE, 
            0, 
            globalRowToCruBufSize, 
            globalRowToCru.data());

    ASSERT(chargeMapSize > 0);
    initQueue.enqueueFillBuffer(
            chargeMap, 
            0.0f, 
            0, 
            chargeMapSize);

    initQueue.finish();
}

GPUClusterFinder::Result GPUClusterFinder::run()
{
    static_assert(sizeof(cl_int) == sizeof(int));

    log::Info() << "Looking for clusters...";


    /*************************************************************************
     * Setup plan boundaries
     ************************************************************************/

    DigitDivider divider(digits, config.chunks);

    for (Plan &plan : plans)
    {
        optional<DigitDivider::Chunk> chunk = divider.nextChunk(PADDING);

        ASSERT(chunk.has_value());

        log::Debug() << "got new chunk: {start: " << chunk->start
                     << ", items: " << chunk->items
                     << ", future: " << chunk->future << "}";

        plan.start = chunk->start;
        plan.items = chunk->items;
        plan.future = chunk->future;
    }

    ASSERT(plans.front().start == 0);
    ASSERT(plans.back().start + plans.back().items == size_t(digits.size()));
    ASSERT(plans.back().future == 0);


    /*************************************************************************
     * Execute plans
     ************************************************************************/

    for (Plan &plan : plans)
    {
        if (config.usePackedDigits)
        {
            findCluster<PackedDigit>(plan, nonstd::span<const PackedDigit>(packedDigits)); 
        }
        else
        {
            findCluster(plan, digits);
        }

        plan.clustering.finish();
    }
    

    /*************************************************************************
     * Compute clusters
     ************************************************************************/

    computeAndReadClusters();



    printClusters(clusters, 10);

    log::Info() << "Found " << clusters.size() << " clusters.";


    return Result{clusters, {0,0, {{}}}};
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
    if (config.usePackedDigits)
    {
        env.addDefine("USE_PACKED_DIGIT");
    }
}

template<class DigitT>
void GPUClusterFinder::findCluster(Plan &plan, nonstd::span<const DigitT> digits)
{
    if (plan.id > 0)
    {
        std::vector<cl::Event> ev = {*plans[plan.id-1].digitsToDevice.get()};
        plan.clustering.enqueueBarrierWithWaitList(&ev);
    }

    /*************************************************************************
     * Copy digits to device
     ************************************************************************/

    const size_t sizeofDigit = sizeof(DigitT);

    nonstd::span<const DigitT> toCopy = digits.subspan(
            plan.start, 
            plan.items + plan.future);

    ASSERT(digitsBufSize > 0);
    ASSERT(toCopy.size() * sizeofDigit <= digitsBufSize);

    log::Debug() << "Sending " << toCopy.size() * sizeofDigit << " bytes to device";

    plan.clustering.enqueueWriteBuffer(
            digitsBuf,
            CL_FALSE,
            plan.start * sizeofDigit,
            toCopy.size() * sizeofDigit,
            toCopy.data(),
            nullptr,
            plan.digitsToDevice.get());


    /*************************************************************************
     * Write digits to chargeMap
     ************************************************************************/

    cl::NDRange local(64);

    DBG(plan.start);
    DBG(toCopy.size());

    plan.fillChargeMap.setArg(0, digitsBuf);
    plan.fillChargeMap.setArg(1, chargeMap);
    plan.clustering.enqueueNDRangeKernel(
            plan.fillChargeMap,
            cl::NDRange(plan.start),
            cl::NDRange(toCopy.size()),
            local,
            nullptr,
            plan.fillingChargeMap.get());

    if (plan.id > 0)
    {
        std::vector<cl::Event> ev = {*plans[plan.id-1].fillingChargeMap.get()};
        plan.clustering.enqueueBarrierWithWaitList(&ev);
    }


    /*************************************************************************
     * Look for peaks
     ************************************************************************/

    size_t backlog = (plan.id == 0) ? 0 : plans[plan.id-1].future;

    ASSERT(plan.start - backlog <= plan.start);

    DBG(plan.start - backlog);
    DBG(backlog + plan.items);

    plan.findPeaks.setArg(0, chargeMap);
    plan.findPeaks.setArg(1, digitsBuf);
    plan.findPeaks.setArg(2, isPeakBuf);
    plan.clustering.enqueueNDRangeKernel(
            plan.findPeaks,
            cl::NDRange(plan.start - backlog),
            cl::NDRange(backlog + plan.items),
            local,
            nullptr,
            plan.findingPeaks.get());
}

void GPUClusterFinder::computeAndReadClusters()
{
    Plan &plan = plans.front();


    /*************************************************************************
     * Compact peaks
     ************************************************************************/

    size_t clusterNum = streamCompaction.enqueue(
            plan.clustering,
            digitsBuf,
            peaksBuf,
            isPeakBuf);

    log::Debug() << "Found " << clusterNum << " peaks";


    /*************************************************************************
     * Compute cluster
     ************************************************************************/

    cl::NDRange local(64);

    if (clusterNum > 0)
    {
        plan.computeClusters.setArg(0, chargeMap);
        plan.computeClusters.setArg(1, peaksBuf);
        plan.computeClusters.setArg(2, globalToLocalRowBuf);
        plan.computeClusters.setArg(3, globalRowToCruBuf);
        plan.computeClusters.setArg(4, clusterBuf);
        plan.clustering.enqueueNDRangeKernel(
                plan.computeClusters,
                cl::NullRange,
                cl::NDRange(clusterNum),
                local,
                nullptr,
                plan.computingClusters.get());
    }


    /*************************************************************************
     * Reset charge map
     ************************************************************************/

    plan.resetChargeMap.setArg(0, digitsBuf);
    plan.resetChargeMap.setArg(1, chargeMap);
    plan.clustering.enqueueNDRangeKernel(
            plan.resetChargeMap,
            cl::NullRange,
            digits.size(),
            local,
            nullptr,
            plan.zeroChargeMap.get());
    

    /*************************************************************************
     * Copy cluster to host
     ************************************************************************/

    clusters.resize(clusterNum);

    log::Info() << "Copy results back...";
    ASSERT(clusters.size() * sizeof(Cluster) <= clusterBufSize);

    if (clusterNum > 0)
    {
        plan.clustering.enqueueReadBuffer(
                clusterBuf, 
                CL_TRUE, 
                0, 
                clusters.size() * sizeof(Cluster), 
                clusters.data(),
                nullptr,
                plan.clustersToHost.get());
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
