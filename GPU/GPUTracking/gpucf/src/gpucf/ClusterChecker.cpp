#include "ClusterChecker.h"

#include <gpucf/common/float.h>
#include <gpucf/common/log.h>

#include <algorithm>


using namespace gpucf;


void ClusterChecker::ClusterMap::add(const Cluster &c)
{
    clusters[c.globalRow()].push_back(c);
}

void ClusterChecker::ClusterMap::addAll(nonstd::span<const Cluster> cs)
{
    for (const Cluster &c : cs)
    {
        add(c);
    }
}

size_t ClusterChecker::ClusterMap::size() const
{
    size_t totalSize = 0;

    for (const auto &p : clusters)
    {
        totalSize += p.second.size();
    }

    return totalSize;
}

bool ClusterChecker::ClusterMap::contains(
        const Cluster &c) const
{
    auto tgtEntry = clusters.find(c.globalRow());

    if (tgtEntry == clusters.end())
    {
        return false;
    }

    const std::vector<Cluster> &tgtRow = tgtEntry->second;

    auto lookup = std::find_if(tgtRow.begin(), tgtRow.end(), 
            [&](const Cluster &c2) 
            { 
                return c.eq(c2, epsilonSmall, epsilonBig, mask); 
            });

    return lookup != tgtRow.end();
}

nonstd::optional<Cluster> ClusterChecker::ClusterMap::tryLookup(
        const Cluster &c) const
{
    auto tgtEntry = clusters.find(c.globalRow());

    if (tgtEntry == clusters.end())
    {
        return nonstd::nullopt;
    }

    const std::vector<Cluster> &tgtRow = tgtEntry->second;

    auto lookup = std::find_if(tgtRow.begin(), tgtRow.end(), 
            [&](const Cluster &c2) 
            { 
                return c.eq(c2, epsilonSmall, epsilonBig, mask); 
            });

    if (lookup == tgtRow.end())
    {
        return nonstd::nullopt;
    }

    return *lookup;
}

void ClusterChecker::ClusterMap::setClusterEqParams(
        float epsilonSmall, 
        float epsilonBig,
        Cluster::FieldMask mask)
{
    this->epsilonSmall = epsilonSmall;
    this->epsilonBig   = epsilonBig;
    this->mask         = mask; 
}


ClusterChecker::ClusterChecker(nonstd::span<const Cluster> t)
{
    truth.addAll(t); 
}


bool ClusterChecker::verify(
        nonstd::span<const Cluster> clusters, 
        bool showExamples)
{
    gpucf::log::Info() << "Reviewing found clusters...";

    gpucf::log::Info() << "Found " << clusters.size() << " clusters.";
    gpucf::log::Info() << "Groundtruth has " << truth.size() << " clusters.";

    truth.setClusterEqParams(
            FEQ_EPSILON_SMALL, 
            FEQ_EPSILON_BIG, 
            Cluster::Field_all);
    std::vector<Cluster> wrongClusters = findWrongClusters(clusters);
    gpucf::log::Info() << clusters.size() - wrongClusters.size() 
                       << " correct clusters.";

    gpucf::log::Info() << "Testing remaining " << wrongClusters.size() 
                       << " clusters with relaxed equality tests.";

    findAndLogTruth(
        wrongClusters,
        "Eq with bigger epsilon",
        showExamples,
        FEQ_EPSILON_SMALL * 2,
        FEQ_EPSILON_BIG * 2,
        Cluster::Field_all);

    findAndLogTruth(
        wrongClusters,
        "Eq without field Q",
        showExamples,
        FEQ_EPSILON_SMALL,
        FEQ_EPSILON_BIG,
        Cluster::Field_all ^ Cluster::Field_Q);

    findAndLogTruth(
        wrongClusters,
        "Eq without field QMax",
        showExamples,
        FEQ_EPSILON_SMALL,
        FEQ_EPSILON_BIG,
        Cluster::Field_all ^ Cluster::Field_QMax);

    findAndLogTruth(
        wrongClusters,
        "Eq without field padSigma",
        showExamples,
        FEQ_EPSILON_SMALL,
        FEQ_EPSILON_BIG,
        Cluster::Field_all ^ Cluster::Field_padSigma);

    findAndLogTruth(
        wrongClusters,
        "Eq without field timeSigma",
        showExamples,
        FEQ_EPSILON_SMALL,
        FEQ_EPSILON_BIG,
        Cluster::Field_all ^ Cluster::Field_timeSigma);

    findAndLogTruth(
        wrongClusters,
        "Eq only with timeMean and padMean",
        showExamples,
        FEQ_EPSILON_SMALL,
        FEQ_EPSILON_BIG,
        Cluster::Field_timeMean | Cluster::Field_padMean);

    size_t brokenClusters = 0;
    size_t nansFound = 0;
    for (const Cluster &cluster : clusters)
    {
        brokenClusters += cluster.hasNegativeEntries();
        nansFound      += cluster.hasNaN();
    }

    gpucf::log::Debug() << "Found " << nansFound << " NaNs.";
    gpucf::log::Debug() << "Found " << brokenClusters << " other weird entries.";

    bool ok = (brokenClusters == 0) && (nansFound == 0);

    return ok;
}

std::vector<Cluster> ClusterChecker::findWrongClusters(
        nonstd::span<const Cluster> clusters)
{
    std::vector<Cluster> wrongCluster;
    for (const Cluster &cluster : clusters)
    {
        if (!truth.contains(cluster))
        {
            wrongCluster.push_back(cluster);
        }
    }

    return wrongCluster;
}

std::vector<ClusterChecker::ClusterPair> ClusterChecker::findTruth(
        nonstd::span<const Cluster> clusters)
{
    std::vector<ClusterPair> withTruth;
    for (const Cluster &cluster : clusters)
    {
        auto optTruth = truth.tryLookup(cluster); 
        if (optTruth != nonstd::nullopt)
        {
            withTruth.push_back({*optTruth, cluster});
        }
    }
    return withTruth;
}

void ClusterChecker::findAndLogTruth(
        nonstd::span<const Cluster> clusters,
        const std::string &testPrefix,
        bool showExample,
        float epsilonSmall,
        float epsilonBig,
        Cluster::FieldMask mask)
{
    truth.setClusterEqParams(epsilonSmall, epsilonBig, mask);    

    std::vector<ClusterPair> withTruth = findTruth(clusters);

    log::Info() << testPrefix << ": " << withTruth.size();


    if (showExample && !withTruth.empty())
    {
        constexpr size_t maxExamples = 5;
        for (size_t i = 0; i < std::min(withTruth.size(), maxExamples); i++)
        {
            log::Debug() << "Example " << i << ": ";
            log::Debug() << "Truth: " << withTruth[i].first;
            log::Debug() << "GPU:   " << withTruth[i].second;
        }
    }
}


// vim: set ts=4 sw=4 sts=4 expandtab:
