#pragma once

#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/common/Event.h>
#include <gpucf/common/Measurements.h>
#include <gpucf/gpu/StreamCompaction.h>

#include <nonstd/optional.hpp>
#include <nonstd/span.hpp>

#include <memory>
#include <vector>


namespace gpucf
{

class ClEnv;

class GPUClusterFinder 
{

public:
    struct Config
    {
        size_t queueNum = 1;

        bool usePackedDigits = false;
        bool zeroChargeMap   = true;
    };
    
    struct Result
    {
        std::vector<Cluster> clusters;
        Measurement profiling; 
    };

    static const Config defaultConfig;

    std::vector<Digit> getPeaks() const;

    void setup(Config, ClEnv &, nonstd::span<const Digit>);

    Result run();

private:
    class Worker
    {
    public:
        Worker(GPUClusterFinder &);

        template<class DigitT>
        void findCluster(
                nonstd::optional<Event>,
                nonstd::optional<Event>,
                nonstd::span<const DigitT>);

        void copyCluster(
                nonstd::optional<Event>,
                nonstd::span<Cluster>);

        size_t getClusterNum() const;

        Event getDigitsToDevice() const;
        Event getClustersToHost() const;
        Event getDigitsToChargeMap() const;

        Lane finish();

    private:
        GPUClusterFinder &parent;

        size_t clusterNum = 0;

        Event digitsToDevice; 
        Event zeroChargeMap;
        Event fillingChargeMap;
        Event findingPeaks;
        Event computingClusters;
        Event clustersToHost;

        cl::CommandQueue clustering;
        cl::CommandQueue cleanup;
    };

    static void printClusters(
            const std::vector<Cluster> &,
            size_t);

    static std::vector<Cluster> filterCluster(
            const std::vector<int> &,
            const std::vector<Cluster> &);

    static std::vector<Digit> compactDigits(
            const std::vector<int> &,
            const std::vector<Digit> &);

    Config config;

    nonstd::span<const Digit> digits;
    std::vector<PackedDigit>  packedDigits;
    std::vector<Digit>        peaks;

    StreamCompaction streamCompaction;

    cl::Context context;
    cl::Device device;

    cl::Kernel findPeaks;
    cl::Kernel fillChargeMap;
    cl::Kernel computeClusters;
    cl::Kernel resetChargeMap;

    cl::Buffer chargeMap;
    size_t     chargeMapSize = 0;

    cl::Buffer digitsBuf;
    cl::Buffer peaksBuf;
    size_t     digitsBufSize = 0;

    cl::Buffer globalToLocalRowBuf;
    size_t     globalToLocalRowBufSize = 0;

    cl::Buffer globalRowToCruBuf;
    size_t     globalRowToCruBufSize = 0;

    cl::Buffer isPeakBuf;
    size_t     isPeakBufSize = 0;

    cl::Buffer clusterBuf;
    size_t     clusterBufSize = 0;


    void fillPackedDigits();

    void addDefines(ClEnv &);
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
