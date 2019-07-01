#include "ClusterChecker.h"

#include <gpucf/common/float.h>
#include <gpucf/common/log.h>

#include <algorithm>


using namespace gpucf;


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

    if (wrongClusters.size() == 0)
    {
        gpucf::log::Success() << "All clusters look correct.";
        return true;
    }

    gpucf::log::Error() << "Found " 
                        << clusters.size() - wrongClusters.size() 
                        << " correct cluster.";


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

    /* printDuplicates(clusters, Cluster::Field_all ^ Cluster::Field_QMax); */

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

void ClusterChecker::printDuplicates(
        nonstd::span<const Cluster> clusters,
        Cluster::FieldMask mask)
{
    ClusterMap map;     

    map.setClusterEqParams(0, 0, mask);
    map.addAll(clusters);

    std::vector<Cluster> duplicates = map.findDuplicates();

    log::Info() << "Found duplicates: " << duplicates.size();

    for (const Cluster &d : duplicates)
    {
        log::Debug() << d;
    }
}


// vim: set ts=4 sw=4 sts=4 expandtab:
