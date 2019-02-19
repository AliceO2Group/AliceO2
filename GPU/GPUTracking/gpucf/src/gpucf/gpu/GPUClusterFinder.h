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
        size_t chunks = 1;

        bool usePackedDigits = false;
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
    struct Plan
    {
        /**
         * Index of the first digit that is processed with this plan.
         */
        size_t start;

        /**
         * Number of digits that have to be transferred to the device,
         * written to the chargeMap and have to run the clusterFinder on.
         */
        size_t items;

        /**
         * Number of digits that have to be transferred to the device and
         * written to the chargeMap but are not looked at for peaks yet.
         * These are necessary as the cluster finder has to look up to two
         * timesteps into the future to compute cluster. So if items and
         * backlog contain digits up to timestep t then the future consists of
         * digits from timesteps t+1 and t+2.
         *
         * Future digits are further processed by the next plan.
         * The future of the last plan is always zero.
         */
        size_t future;

        /**
         * Number of peaks that have been found in plans[id-1].future+items.
         */
        size_t peaks;

        /**
         * my id
         */
        size_t id;

        Event digitsToDevice; 
        Event zeroChargeMap;
        Event fillingChargeMap;
        Event findingPeaks;
        Event computingClusters;
        Event clustersToHost;

        cl::Kernel findPeaks;
        cl::Kernel fillChargeMap;
        cl::Kernel computeClusters;
        cl::Kernel resetChargeMap;

        cl::CommandQueue clustering;
        cl::CommandQueue cleanup;

        Plan(cl::Context, cl::Device, cl::Program, size_t);
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

    std::vector<Cluster> clusters;

    std::vector<Plan> plans;

    StreamCompaction streamCompaction;

    cl::Context context;
    cl::Device device;


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


    template<class DigitT>
    void findCluster(Plan &, nonstd::span<const DigitT>);

    void computeAndReadClusters();
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
