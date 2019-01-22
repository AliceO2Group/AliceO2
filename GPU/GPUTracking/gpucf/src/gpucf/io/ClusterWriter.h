#pragma once

#include <gpucf/common/Cluster.h>

#include <args/args.hxx>


namespace gpucf
{

class ClusterWriter
{

public:
    ClusterWriter(const std::string &out)
        : fName(out)
    {
    }

    void write(const std::vector<Cluster> &);
        
private:
    std::string fName;
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
