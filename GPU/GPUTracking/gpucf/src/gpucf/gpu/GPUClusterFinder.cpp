#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>

#include <shared/tpc.h>


using namespace gpucf;


const GPUClusterFinder::Config GPUClusterFinder::defaultConfig;


GPUClusterFinder::Worker::Worker(GPUClusterFinder &p)
    : parent(p)
{
    clustering = cl::CommandQueue(
            parent.context, 
            parent.device, 
            CL_QUEUE_PROFILING_ENABLE);
    cleanup    = cl::CommandQueue(
            parent.context, 
            parent.device, 
            CL_QUEUE_PROFILING_ENABLE);
}

template<class DigitT>
void GPUClusterFinder::Worker::findCluster(
        nonstd::optional<Event> prevDigitsToDevice,
        nonstd::optional<Event> prevFillingChargeMap,
        nonstd::span<const DigitT> myDigits)
{
    ASSERT(prevDigitsToDevice.has_value() == prevFillingChargeMap.has_value());

    if (prevDigitsToDevice.has_value())
    {
        std::vector<cl::Event> ev = {*prevDigitsToDevice->get()};
        clustering.enqueueBarrierWithWaitList(&ev);
    }

    ASSERT(parent.digitsBufSize > 0);
    log::Info() << "Sending " << parent.digitsBufSize << " bytes to device";
    clustering.enqueueWriteBuffer(
            parent.digitsBuf,
            CL_FALSE,
            0,
            parent.digitsBufSize,
            myDigits.data(),
            nullptr,
            digitsToDevice.get());

    ASSERT(parent.chargeMapSize > 0);
    if (parent.config.zeroChargeMap)
    {
        clustering.enqueueFillBuffer(
                parent.chargeMap,
                0.0f,
                0,
                parent.chargeMapSize,
                nullptr,
                zeroChargeMap.get());
    }

    cl::NDRange global(parent.digits.size());
    cl::NDRange local(64);

    parent.fillChargeMap.setArg(0, parent.digitsBuf);
    parent.fillChargeMap.setArg(1, parent.chargeMap);
    clustering.enqueueNDRangeKernel(
            parent.fillChargeMap,
            cl::NullRange,
            global,
            local,
            nullptr,
            fillingChargeMap.get());

    if (prevFillingChargeMap)
    {
        std::vector<cl::Event> ev = {*prevFillingChargeMap->get()};
        clustering.enqueueBarrierWithWaitList(&ev);
    }

    parent.findPeaks.setArg(0, parent.chargeMap);
    parent.findPeaks.setArg(1, parent.digitsBuf);
    parent.findPeaks.setArg(2, parent.isPeakBuf);
    clustering.enqueueNDRangeKernel(
            parent.findPeaks, 
            cl::NullRange,
            global,
            local,
            nullptr,
            findingPeaks.get());

    clusterNum = parent.streamCompaction.enqueue(
            clustering,
            parent.digitsBuf,
            parent.peaksBuf,
            parent.isPeakBuf);

    log::Debug() << "Found " << clusterNum << " peaks";

    parent.computeClusters.setArg(0, parent.chargeMap);
    parent.computeClusters.setArg(1, parent.peaksBuf);
    parent.computeClusters.setArg(2, parent.globalToLocalRowBuf);
    parent.computeClusters.setArg(3, parent.globalRowToCruBuf);
    parent.computeClusters.setArg(4, parent.clusterBuf);
    clustering.enqueueNDRangeKernel(
            parent.computeClusters,
            cl::NullRange,
            cl::NDRange(clusterNum),
            local,
            nullptr,
            computingClusters.get());
}

void GPUClusterFinder::Worker::copyCluster(
        nonstd::optional<Event> prevClustersToHost,
        nonstd::span<Cluster> clusters)
{
    ASSERT(size_t(clusters.size()) == clusterNum);

    cl::NDRange global(parent.digits.size());
    cl::NDRange local(64);
    

    if (prevClustersToHost)
    {
        /* std::vector<cl::Event> ev = {prevFillingChargeMap->get()}; */
        cl::Event tmp = *prevClustersToHost->get();
        std::vector<cl::Event> ev = {tmp};
        clustering.enqueueBarrierWithWaitList(&ev);
    }

    if (!parent.config.zeroChargeMap)
    {
        parent.resetChargeMap.setArg(0, parent.digitsBuf);
        parent.resetChargeMap.setArg(1, parent.chargeMap);
        clustering.enqueueNDRangeKernel(
                parent.resetChargeMap,
                cl::NullRange,
                global,
                local,
                nullptr,
                zeroChargeMap.get());
    }

    log::Info() << "Copy results back...";
    ASSERT(clusters.size() * sizeof(Cluster) <= parent.clusterBufSize);
    clustering.enqueueReadBuffer(
            parent.clusterBuf, 
            CL_TRUE, 
            0, 
            clusters.size() * sizeof(Cluster), 
            clusters.data(),
            nullptr,
            clustersToHost.get());
}

std::vector<Measurement> GPUClusterFinder::Worker::finish()
{
    clustering.finish();
    cleanup.finish();

    std::vector<Measurement> measurements =
    {
        {"digitsToDevice", digitsToDevice.executionTimeMs()},
        {"zeroChargeMap", 
            parent.config.zeroChargeMap ? zeroChargeMap.executionTimeMs() : 0},
        {"fillChargeMap", fillingChargeMap.executionTimeMs()},
        {"findPeaks", findingPeaks.executionTimeMs()},
        {"compactPeaks", parent.streamCompaction.executionTimeMs()},
        {"computeClusters", computingClusters.executionTimeMs()},
        {"resetChargeMap",
            parent.config.zeroChargeMap ? 0 : zeroChargeMap.executionTimeMs()},
        {"clustersToHost", clustersToHost.executionTimeMs()},
    };

    return measurements;
}

size_t GPUClusterFinder::Worker::getClusterNum() const
{
    return clusterNum;
}

std::vector<Digit> GPUClusterFinder::getPeaks() const
{
    return peaks;
}

void GPUClusterFinder::setup(Config conf, ClEnv &env, nonstd::span<const Digit> digits)
{
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

    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    fillChargeMap    = cl::Kernel(cfprg, "fillChargeMap");
    resetChargeMap   = cl::Kernel(cfprg, "resetChargeMap");
    findPeaks        = cl::Kernel(cfprg, "findPeaks");
    computeClusters  = cl::Kernel(cfprg, "computeClusters");

    // create buffers
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

    Worker worker(*this);


    if (config.usePackedDigits)
    {
        worker.findCluster(
                nonstd::nullopt, 
                nonstd::nullopt, 
                nonstd::span<const PackedDigit>(packedDigits));
    }
    else
    {
        worker.findCluster(nonstd::nullopt, nonstd::nullopt, digits);
    }

    size_t clusterNum = worker.getClusterNum();
    std::vector<Cluster> clusters(clusterNum);

    worker.copyCluster(nonstd::nullopt, clusters);

    auto measurements = worker.finish();


    printClusters(clusters, 10);

    log::Info() << "Found " << clusters.size() << " clusters.";


    return Result{clusters, measurements};
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

// vim: set ts=4 sw=4 sts=4 expandtab:
