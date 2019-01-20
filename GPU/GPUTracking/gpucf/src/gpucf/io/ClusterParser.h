#pragma once

#include <gpucf/common/Cluster.h>

#include <regex>
#include <string>
#include <vector>


namespace gpucf
{

class ClusterParser 
{
    
public:
    
    bool operator()(const std::string &line, std::vector<Cluster> *);

private:
    static std::regex prefix;

};

} // namespace gpucf


// vim: set ts=4 sw=4 sts=4 expandtab:
