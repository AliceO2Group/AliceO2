#include "ClusterParser.h"

#include "parse.h"

#include <gpucf/log.h>


using namespace gpucf;


std::regex ClusterParser::prefix = std::regex("Cluster:\\s*(.*)");


bool ClusterParser::operator()(const std::string &line, 
        std::vector<Cluster> *cluster) 
{
    std::smatch sm; 
    bool isCluster = std::regex_match(line, sm, prefix);

    if (!isCluster) 
    {
        return true;
    }
    ASSERT(sm.size() == 2);

    const std::string &clusterMembers = sm[1];    

    MATCH_INT(clusterMembers, cru);
    MATCH_INT(clusterMembers, row);
    MATCH_INT(clusterMembers, Q);
    MATCH_INT(clusterMembers, Qmax);
    MATCH_FLOAT(clusterMembers, padMean);
    MATCH_FLOAT(clusterMembers, timeMean);
    MATCH_FLOAT(clusterMembers, padSigma);
    MATCH_FLOAT(clusterMembers, timeSigma);

    cluster->emplace_back(cru, row, Q, Qmax, padMean, timeMean, 
            padSigma, timeSigma);

    return true;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
