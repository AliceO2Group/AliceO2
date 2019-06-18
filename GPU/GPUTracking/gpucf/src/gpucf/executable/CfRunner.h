#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/executable/Executable.h>


namespace gpucf
{

class CfRunner : public Executable
{
    
public:
    CfRunner();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    OptStringFlag digitFile;
    OptStringFlag clusterResultFile;
    OptStringFlag peakFile;

    std::unique_ptr<args::Group> cfconfig;
    OptFlag cpu;

#define CLUSTER_FINDER_FLAG(name, val, def, desc) OptFlag name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define MEMORY_LAYOUT(name, def, desc) OptFlag layout##name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define CLUSTER_BUILDER(name, def, desc) OptFlag builder##name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

