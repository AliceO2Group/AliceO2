#pragma once

#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/gpu/GPUAlgorithm.h>

#include <memory>
#include <vector>


namespace gpucf
{

class ClEnv;

class GPUClusterFinder : public GPUAlgorithm
{

public:
    GPUClusterFinder();

    std::vector<Digit> getPeaks() const;

protected:
    void setupImpl(ClEnv &, const DataSet &) override;

    GPUAlgorithm::Result runImpl() override;

private:
    static void printClusters(
            const std::vector<int> &, 
            const std::vector<Cluster> &,
            size_t);

    static std::vector<Cluster> filterCluster(
            const std::vector<int> &,
            const std::vector<Cluster> &);

    static std::vector<Digit> compactDigits(
            const std::vector<int> &,
            const std::vector<Digit> &);

    std::vector<Digit> digits;
    std::vector<Digit> peaks;

    std::vector<int> globalToLocalRow;

    cl::Context context;
    cl::Device device;

    cl::Kernel findPeaks;
    cl::Kernel fillChargeMap;
    cl::Kernel computeClusters;

    cl::Buffer chargeMap;
    size_t     chargeMapSize = 0;

    cl::Buffer digitsBuf;
    size_t     digitsBufSize = 0;

    cl::Buffer globalToLocalRowBuf;
    size_t     globalToLocalRowBufSize = 0;

    cl::Buffer isPeakBuf;
    size_t     isPeakBufSize = 0;

    cl::Buffer clusterBuf;
    size_t     clusterBufSize = 0;
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
