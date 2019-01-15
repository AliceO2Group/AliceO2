#include "ClusterChecker.h"

#include <gpucf/log.h>

#include <cmath>


using gpucf::ClusterChecker;


bool ClusterChecker::verify(const std::vector<Cluster> &clusters)
{
    gpucf::log::Info() << "Reviewing found clusters...";

    size_t brokenClusters = 0;
    size_t nansFound = 0;
    for (const Cluster &cluster : clusters)
    {
        brokenClusters += hasWeirdEntries(cluster);
        nansFound      += hasNaN(cluster);
    }

    gpucf::log::Info() << "Found " << clusters.size() << " clusters.";    
    gpucf::log::Info() << "Found " << nansFound << " NaNs.";
    gpucf::log::Info() << "Found " << brokenClusters << " other weird entries.";

    bool ok = (brokenClusters == 0) && (nansFound == 0);

    return ok;
}

bool ClusterChecker::hasNaN(const Cluster &c)
{
    return std::isnan(c.cru) 
             || std::isnan(c.row)
             || std::isnan(c.Q)
             || std::isnan(c.QMax)
             || std::isnan(c.padMean)
             || std::isnan(c.timeMean)
             || std::isnan(c.padSigma)
             || std::isnan(c.timeSigma);
}

bool ClusterChecker::hasWeirdEntries(const Cluster &c)
{
    return c.cru < 0
             || c.row < 0
             || c.Q < 0
             || c.QMax < 0
             || c.padMean < 0
             || c.timeMean < 0
             || c.padSigma < 0
             || c.timeSigma < 0; 
}

// vim: set ts=4 sw=4 sts=4 expandtab:
