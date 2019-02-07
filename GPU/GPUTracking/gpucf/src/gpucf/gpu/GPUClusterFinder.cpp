#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/common/Event.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>

#include <shared/tpc.h>


using namespace gpucf;


GPUClusterFinder::GPUClusterFinder()
    : GPUAlgorithm("NaiveClusterFinder")
{
}

std::vector<Digit> GPUClusterFinder::getPeaks() const
{
    return peaks;
}

void GPUClusterFinder::setupImpl(ClEnv &env, const DataSet &data)
{
    digits = data.deserialize<Digit>();

    context = env.getContext(); 
    device  = env.getDevice();

    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    fillChargeMap   = cl::Kernel(cfprg, "fillChargeMap");
    findPeaks       = cl::Kernel(cfprg, "findPeaks");
    computeClusters = cl::Kernel(cfprg, "computeClusters");

    // create buffers
    digitsBufSize = sizeof(Digit) * digits.size();
    digitsBuf = cl::Buffer(context,
                          CL_MEM_READ_WRITE,
                          digitsBufSize);

    globalToLocalRow = RowInfo::instance().globalToLocalMap;
    const size_t numOfRows = globalToLocalRow.size();
    globalToLocalRowBufSize = sizeof(cl_int) * numOfRows;
    globalToLocalRowBuf = cl::Buffer(context,
                                     CL_MEM_READ_ONLY,
                                     globalToLocalRowBufSize);

    isPeakBufSize = digits.size() * sizeof(cl_int);
    isPeakBuf = cl::Buffer(context, 
                           CL_MEM_READ_WRITE, 
                           isPeakBufSize);

    clusterBufSize = digits.size() * sizeof(Cluster);
    clusterBuf = cl::Buffer(context,
                            CL_MEM_WRITE_ONLY,
                            clusterBufSize);

    log::Info() << "Found " << numOfRows << " rows";

    chargeMapSize  = 
        numOfRows * TPC_PADS_PER_ROW_PADDED 
                  * TPC_MAX_TIME_PADDED 
                  * sizeof(cl_float);
    chargeMap = cl::Buffer(context, CL_MEM_READ_WRITE, chargeMapSize);
}

GPUAlgorithm::Result GPUClusterFinder::runImpl()
{
    static_assert(sizeof(cl_int) == sizeof(int));

    log::Info() << "Looking for clusters...";

    // Events for profiling
    Event digitsToDevice;
    Event zeroChargeMap;
    Event fillingChargeMap;
    Event findingPeaks;
    Event computingClusters;
    Event clustersToHost;

    // Setup queue
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    ASSERT(globalToLocalRowBufSize > 0);
    queue.enqueueWriteBuffer(globalToLocalRowBuf, CL_FALSE, 0, 
            globalToLocalRowBufSize, globalToLocalRow.data());

    ASSERT(digitsBufSize > 0);
    queue.enqueueWriteBuffer(digitsBuf, 
                             CL_FALSE, 
                             0, 
                             digitsBufSize, 
                             digits.data(), 
                             nullptr, 
                             digitsToDevice.get());

    ASSERT(chargeMapSize > 0);
    queue.enqueueFillBuffer(chargeMap, 
                            0.0f, 
                            0, 
                            chargeMapSize, 
                            nullptr, 
                            zeroChargeMap.get());

    cl::NDRange global(digits.size());
    cl::NDRange local(16);

    fillChargeMap.setArg(0, digitsBuf);
    fillChargeMap.setArg(1, chargeMap);
    queue.enqueueNDRangeKernel(fillChargeMap, 
                               cl::NullRange, 
                               global, 
                               local,
                               nullptr,
                               fillingChargeMap.get());

    findPeaks.setArg(0, chargeMap);
    findPeaks.setArg(1, digitsBuf);
    findPeaks.setArg(2, isPeakBuf);
    queue.enqueueNDRangeKernel(findPeaks, 
                               cl::NullRange,
                               global,
                               local,
                               nullptr,
                               findingPeaks.get());

    computeClusters.setArg(0, chargeMap);
    computeClusters.setArg(1, digitsBuf);
    computeClusters.setArg(2, globalToLocalRowBuf);
    computeClusters.setArg(3, isPeakBuf);
    computeClusters.setArg(4, clusterBuf);
    queue.enqueueNDRangeKernel(computeClusters,
                               cl::NullRange,
                               global,
                               local,
                               nullptr,
                               computingClusters.get());

    log::Info() << "Copy results back...";
    std::vector<Cluster> clusters(digits.size());
    ASSERT(clusters.size() * sizeof(Cluster) == clusterBufSize);
    queue.enqueueReadBuffer(clusterBuf, 
                            CL_TRUE, 
                            0, 
                            clusterBufSize, 
                            clusters.data(),
                            nullptr,
                            clustersToHost.get());

    std::vector<int> isClusterCenter(digits.size());
    ASSERT(isClusterCenter.size() * sizeof(int) == isPeakBufSize);
    queue.enqueueReadBuffer(isPeakBuf, CL_TRUE, 0, isPeakBufSize,
            isClusterCenter.data());

    printClusters(isClusterCenter, clusters, 10);

    clusters = filterCluster(isClusterCenter, clusters);
    peaks = compactDigits(isClusterCenter, digits);

    ASSERT(clusters.size() == peaks.size());

    log::Info() << "Found " << clusters.size() << " clusters.";

    DataSet res;
    res.serialize(clusters);

    std::vector<Measurement> measurements =
    {
        {"digitsToDevice", digitsToDevice.executionTimeMs()},   
        {"zeroChargeMap", zeroChargeMap.executionTimeMs()},   
        {"fillingChargeMap", fillingChargeMap.executionTimeMs()},   
        {"findingPeaks", findingPeaks.executionTimeMs()},   
        {"computingClusters", computingClusters.executionTimeMs()},   
        {"clustersToHost", clustersToHost.executionTimeMs()},   
    };

    return GPUAlgorithm::Result{res, measurements};
}


void GPUClusterFinder::printClusters(
        const std::vector<int> &isCenter,
        const std::vector<Cluster> &clusters,
        size_t maxClusters)
{
    log::Debug() << "Printing found clusters";
    ASSERT(isCenter.size() == clusters.size());
    for (size_t i = 0; i < isCenter.size(); i++)
    {
        ASSERT(isCenter[i] == 0 || isCenter[i] == 1);
        if (isCenter[i])
        {
            log::Debug() << clusters[i];
            maxClusters--;
            if (maxClusters == 0)
            {
                break;
            }
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

// vim: set ts=4 sw=4 sts=4 expandtab:
