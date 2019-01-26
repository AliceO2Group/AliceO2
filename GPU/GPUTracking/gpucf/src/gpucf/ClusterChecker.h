#pragma once

#include <gpucf/common/Cluster.h>
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

    class ClusterMap
    {

    public:
        void add(const Cluster &);
        void addAll(nonstd::span<const Cluster>);

        bool contains(const Cluster &) const;
        nonstd::optional<Cluster> tryLookup(const Cluster &) const;

        void setClusterEqParams(float, Cluster::FieldMask);

        size_t size() const;
        
    private:
        std::unordered_map<int, std::vector<Cluster>> clusters;

        float epsilon = FEQ_EPSILON;
        Cluster::FieldMask mask = Cluster::Field_all;
    };


    std::vector<Cluster> findWrongClusters(nonstd::span<const Cluster>);

    std::vector<ClusterPair> findTruth(nonstd::span<const Cluster>);

    void findAndLogTruth(
            nonstd::span<const Cluster>,
            const std::string &testPrefix,
            bool showExample,
            float,
            Cluster::FieldMask);

    ClusterMap truth;
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
