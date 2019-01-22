#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/ChargeMap.h>
#include <gpucf/log.h>
#include <gpucf/RowInfo.h>


using namespace gpucf;


GPUClusterFinder::Result GPUClusterFinder::run(ClEnv &env, const std::vector<Digit> &digits)
{
    static_assert(sizeof(cl_int) == sizeof(int));

    cl::Context context = env.getContext();

    // Load kernels
    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    cl::Kernel findClusters(cfprg, "findClusters");

    log::Info() << "Looking for clusters...";

    // Create buffers
    const size_t digitBytes = sizeof(Digit) * digits.size();
    cl::Buffer digitBuf(context, CL_MEM_READ_ONLY, digitBytes);

    const std::vector<int> globalToLocalRow = 
        RowInfo::globalRowToLocalRowMap();
    const size_t numOfRows = globalToLocalRow.size();
    cl::Buffer globalToLocalRowBuf(
            context,
            CL_MEM_READ_ONLY,
            sizeof(cl_int) * numOfRows);

    const size_t isClusterCenterBytes = sizeof(cl_int) * digits.size();
    cl::Buffer isClusterCenterBuf(context, CL_MEM_WRITE_ONLY, isClusterCenterBytes);

    const size_t clusterBytes = sizeof(Cluster) * digits.size();
    cl::Buffer clusterBuf(context, CL_MEM_READ_WRITE, clusterBytes);


    ChargeMap chargeMap(context, cfprg, numOfRows);


    // Set kernel args
    findClusters.setArg(0, chargeMap.get());
    findClusters.setArg(1, digitBuf);
    findClusters.setArg(2, globalToLocalRowBuf);
    findClusters.setArg(3, isClusterCenterBuf);
    findClusters.setArg(4, clusterBuf);

    // Setup queue
    cl::Device  device  = env.getDevice();
    cl::CommandQueue queue (context, device);

    queue.enqueueWriteBuffer(globalToLocalRowBuf, CL_FALSE, 0, 
            sizeof(cl_int) * numOfRows, globalToLocalRow.data());

    queue.enqueueWriteBuffer(digitBuf, CL_FALSE, 0, digitBytes, 
            digits.data());

    cl::NDRange global(digits.size());
    cl::NDRange local(16);
    chargeMap.enqueueFill(queue, digitBuf, global, local);

    queue.enqueueNDRangeKernel(findClusters, cl::NullRange, global, local);

    log::Info() << "Copy results back...";
    std::vector<Cluster> clusters(digits.size());
    ASSERT(clusters.size() * sizeof(Cluster) == clusterBytes);
    queue.enqueueReadBuffer(clusterBuf, CL_TRUE, 0, clusterBytes, 
            clusters.data());

    std::vector<int> isClusterCenter(digits.size());
    queue.enqueueReadBuffer(isClusterCenterBuf, CL_TRUE, 0, isClusterCenterBytes,
            isClusterCenter.data());

    printClusters(isClusterCenter, clusters, 10);

    clusters = filterCluster(isClusterCenter, clusters);
    std::vector<Digit> peaks = findPeaks(isClusterCenter, digits);

    log::Info() << "Found " << clusters.size() << " clusters.";

    return Result{clusters, peaks};
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

std::vector<Digit> GPUClusterFinder::findPeaks(
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
