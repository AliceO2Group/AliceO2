#pragma once

#include <gpucf/common/Cluster.h>
#include <gpucf/common/ClusterMap.h>
#include <gpucf/common/float.h>

#include <nonstd/optional.hpp>
#include <nonstd/span.hpp>

#include <unordered_map>
#include <vector>


namespace gpucf
{

class ClusterChecker
{

public:
    ClusterChecker(nonstd::span<const Cluster>);

    bool verify(nonstd::span<const Cluster>, bool showExamples=true);

private:
    using ClusterPair = std::pair<Cluster, Cluster>;



    std::vector<Cluster> findWrongClusters(nonstd::span<const Cluster>);

    std::vector<ClusterPair> findTruth(nonstd::span<const Cluster>);

    void findAndLogTruth(
            nonstd::span<const Cluster>,
            const std::string &testPrefix,
            bool showExample,
            float,
            float,
            Cluster::FieldMask);

    void printDuplicates(
            nonstd::span<const Cluster>,
            Cluster::FieldMask);

    ClusterMap truth;
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
