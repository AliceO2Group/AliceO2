#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/algorithms/StreamCompaction.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/common/Event.h>
#include <gpucf/common/Fragment.h>
#include <gpucf/common/Measurements.h>

#include <nonstd/optional.hpp>
#include <nonstd/span.hpp>

#include <future>
#include <memory>
#include <thread>
#include <vector>


namespace gpucf
{

class ClEnv;

class GPUClusterFinder 
{

public:

    struct Result
    {
        std::vector<Cluster> clusters;
        std::vector<Step> profiling; 
    };

    static const ClusterFinderConfig defaultConfig;

    std::vector<Digit> getPeaks() const;

    void setup(ClusterFinderConfig, ClEnv &, nonstd::span<const Digit>);

    Result run();

private:
    struct DeviceMemory
    {
        cl::Buffer digits;
        cl::Buffer isPeak;
        cl::Buffer peaks;
        cl::Buffer chargeMap;
        cl::Buffer peakMap;
        cl::Buffer cluster;

        cl::Buffer globalToLocalRow;
        cl::Buffer globalRowToCru;
    };


    struct Backwards
    {
        std::future<cl::Event> digitsToDevice;
        std::future<cl::Event> fillingChargeMap;

        std::future<size_t> clusters;

        std::promise<cl::Event> computeClusters;
    };

    struct Forwards
    {
        std::promise<cl::Event> digitsToDevice;
        std::promise<cl::Event> fillingChargeMap;

        std::promise<size_t> clusters;

        std::future<cl::Event> computeClusters;   
    };


    struct Worker
    {
        /**
         * Number of peaks that have been found by this instance.
         */
        size_t peaks;

        nonstd::optional<Backwards> prev;
        nonstd::optional<Forwards> next;

        Event digitsToDevice; 
        Event zeroChargeMap;
        Event fillingChargeMap;
        Event findingPeaks;
        Event countingPeaks;
        Event computingClusters;
        Event clustersToHost;

        cl::Kernel findPeaks;
        cl::Kernel fillChargeMap;
        cl::Kernel countPeaks;
        cl::Kernel computeClusters;
        cl::Kernel resetMaps;

        DeviceMemory mem;
        StreamCompaction::Worker streamCompaction;

        cl::CommandQueue clustering;
        cl::CommandQueue cleanup;

        std::thread myThread;

        size_t clusterend;

        ClusterFinderConfig config;


        Worker(
                cl::Context, 
                cl::Device, 
                cl::Program, 
                DeviceMemory, 
                ClusterFinderConfig,
                StreamCompaction::Worker,
                Worker *);

        template<class DigitT>
        void run(
                const Fragment &, 
                nonstd::span<const DigitT>, 
                nonstd::span<Cluster>);

        template<class DigitT>
        void runAndCatch(
                const Fragment &, 
                nonstd::span<const DigitT>, 
                nonstd::span<Cluster>);

        template<class DigitT>
        void dispatch(
                const Fragment &,
                nonstd::span<const DigitT>,
                nonstd::span<Cluster>);

        void join();
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

    ClusterFinderConfig config;

    nonstd::span<const Digit> digits;
    std::vector<PackedDigit>  packedDigits;
    std::vector<Digit>        peaks;

    std::vector<Cluster> clusters;

    std::vector<Worker> workers;

    StreamCompaction streamCompaction;

    cl::Context context;
    cl::Device device;

    DeviceMemory mem;


    void fillPackedDigits();

    void addDefines(ClEnv &);

    std::vector<Step> toLane(size_t, const Worker &);

    size_t computeAndReadClusters();
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
