#include "ClusterChecker.h"

#include <gpucf/log.h>

#include <algorithm>


using gpucf::ClusterChecker;


bool ClusterChecker::verify(
        const std::vector<Cluster> &clusters,
        const std::vector<Cluster> &truth)
{
    gpucf::log::Info() << "Reviewing found clusters...";

    size_t correctClusters = countCorrectClusters(clusters, truth);
    gpucf::log::Info() << "Found " << clusters.size() << " clusters.";
    gpucf::log::Info() << "Groundtruth has " << truth.size() << " clusters.";
    gpucf::log::Info() << correctClusters << " correct clusters.";

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

size_t ClusterChecker::countCorrectClusters(
        const std::vector<Cluster> &clusters, 
        const std::vector<Cluster> &truth)
{
    size_t correctClusters = 0;
    for (const Cluster &cluster : clusters)
    {
        if (std::find(truth.begin(), truth.end(), cluster) != truth.end())
        {
            correctClusters++; 
        }
    }

    return correctClusters;
}


// vim: set ts=4 sw=4 sts=4 expandtab:
