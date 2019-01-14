#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/log.h>

#include <shared/tpc.h>


void GPUClusterFinder::run(ClEnv &env, const std::vector<Digit> &digits)
{
    cl::Context context = env.getContext();

    // Load kernels
    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    cl::Kernel digitsToChargeMap(cfprg, "digitsToChargeMap");
    cl::Kernel find3x3Clusters(cfprg, "find3x3Clusters");

    log::Info() << "Looking for clusters...";

    // Create buffers
    const size_t digitBytes = sizeof(Digit) * digits.size();
    cl::Buffer digitBuffer(context, CL_MEM_READ_ONLY, digitBytes);

    const size_t isClusterCenterBytes = sizeof(cl_int) * digits.size();
    cl::Buffer isClusterCenterBuf(context, CL_MEM_WRITE_ONLY, isClusterCenterBytes);

    const size_t clusterBytes = sizeof(Cluster) * digits.size();
    cl::Buffer clusterBuffer(context, CL_MEM_READ_WRITE, clusterBytes);

    const size_t numOfRows = getNumOfRows(digits);
    const size_t chargeMapSize  = 
        numOfRows * TPC_PADS_PER_ROW_BUFFERED * TPC_MAX_TIME_BUFFERED;
    const size_t chargeMapBytes =  sizeof(cl_float) * chargeMapSize;
    cl::Buffer chargeMap(context, CL_MEM_READ_WRITE, chargeMapBytes);

    // Set kernel args
    digitsToChargeMap.setArg(0, digitBuffer);
    digitsToChargeMap.setArg(1, chargeMap);

    find3x3Clusters.setArg(0, chargeMap);
    find3x3Clusters.setArg(1, digitBuffer);
    find3x3Clusters.setArg(2, isClusterCenterBuf);
    find3x3Clusters.setArg(3, clusterBuffer);

    // Setup queue
    cl::Device  device  = env.getDevice();
    cl::CommandQueue queue = cl::CommandQueue(context, device);

    queue.enqueueWriteBuffer(digitBuffer, CL_FALSE, 0, digitBytes, 
            digits.data());

    const cl_float zero = 0;
    queue.enqueueFillBuffer(chargeMap, &zero, 0, chargeMapBytes);

    cl::NDRange global(digits.size());
    queue.enqueueNDRangeKernel(digitsToChargeMap, cl::NullRange, global);

    queue.enqueueNDRangeKernel(find3x3Clusters, cl::NullRange, global);

    log::Info() << "Copy results back...";
    std::vector<Cluster> clusters(digits.size());
    ASSERT(clusters.size() * sizeof(Cluster) == clusterBytes);
    queue.enqueueReadBuffer(clusterBuffer, CL_TRUE, 0, clusterBytes, 
            clusters.data());

    std::vector<int> isClusterCenter(digits.size());
    queue.enqueueReadBuffer(isClusterCenterBuf, CL_TRUE, 0, isClusterCenterBytes,
            isClusterCenter.data());

    printClusters(isClusterCenter, clusters);
}

size_t GPUClusterFinder::getNumOfRows(const std::vector<Digit> &digits)
{
    size_t numOfRows = 0;
    for (const Digit &digit : digits)
    {
        numOfRows = std::max(numOfRows, static_cast<size_t>(digit.row));
    }

    return numOfRows+1;
}

void GPUClusterFinder::printClusters(
        const std::vector<int> &isCenter,
        const std::vector<Cluster> &clusters)
{
    log::Info() << "Printing found clusters";
    ASSERT(isCenter.size() == clusters.size());
    for (size_t i = 0; i < isCenter.size(); i++)
    {
        ASSERT(isCenter[i] == 0 || isCenter[i] == 1);
        if (isCenter[i])
        {
            log::Info() << clusters[i];
        }
    }
}



// vim: set ts=4 sw=4 sts=4 expandtab:

