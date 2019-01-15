#include "GPUClusterFinder.h"

#include <gpucf/ClEnv.h>
#include <gpucf/ChargeMap.h>
#include <gpucf/log.h>


using namespace gpucf;


std::vector<Cluster> GPUClusterFinder::run(ClEnv &env, const std::vector<Digit> &digits)
{
    cl::Context context = env.getContext();

    // Load kernels
    cl::Program cfprg = env.buildFromSrc("clusterFinder.cl");
    cl::Kernel find3x3Clusters(cfprg, "findClusters");


    log::Info() << "Looking for clusters...";

    // Create buffers
    const size_t digitBytes = sizeof(Digit) * digits.size();
    cl::Buffer digitBuffer(context, CL_MEM_READ_ONLY, digitBytes);

    const size_t isClusterCenterBytes = sizeof(cl_int) * digits.size();
    cl::Buffer isClusterCenterBuf(context, CL_MEM_WRITE_ONLY, isClusterCenterBytes);

    const size_t clusterBytes = sizeof(Cluster) * digits.size();
    cl::Buffer clusterBuffer(context, CL_MEM_READ_WRITE, clusterBytes);


    ChargeMap chargeMap(context, cfprg, digits);


    // Set kernel args
    find3x3Clusters.setArg(0, chargeMap.get());
    find3x3Clusters.setArg(1, digitBuffer);
    find3x3Clusters.setArg(2, isClusterCenterBuf);
    find3x3Clusters.setArg(3, clusterBuffer);

    // Setup queue
    cl::Device  device  = env.getDevice();
    cl::CommandQueue queue (context, device);

    queue.enqueueWriteBuffer(digitBuffer, CL_FALSE, 0, digitBytes, 
            digits.data());

    cl::NDRange global(digits.size());
    cl::NDRange local(16);
    chargeMap.enqueueFill(queue, digitBuffer, global, local);

    queue.enqueueNDRangeKernel(find3x3Clusters, cl::NullRange, global, local);

    log::Info() << "Copy results back...";
    std::vector<Cluster> clusters(digits.size());
    ASSERT(clusters.size() * sizeof(Cluster) == clusterBytes);
    queue.enqueueReadBuffer(clusterBuffer, CL_TRUE, 0, clusterBytes, 
            clusters.data());

    static_assert(sizeof(cl_int) == sizeof(int));
    std::vector<int> isClusterCenter(digits.size());
    queue.enqueueReadBuffer(isClusterCenterBuf, CL_TRUE, 0, isClusterCenterBytes,
            isClusterCenter.data());

    printClusters(isClusterCenter, clusters, 10);

    return filterCluster(isClusterCenter, clusters);
}


void GPUClusterFinder::printClusters(
        const std::vector<int> &isCenter,
        const std::vector<Cluster> &clusters,
        size_t maxClusters)
{
    log::Info() << "Printing found clusters";
    ASSERT(isCenter.size() == clusters.size());
    for (size_t i = 0; i < isCenter.size(); i++)
    {
        ASSERT(isCenter[i] == 0 || isCenter[i] == 1);
        if (isCenter[i])
        {
            log::Info() << clusters[i];
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


// vim: set ts=4 sw=4 sts=4 expandtab:
