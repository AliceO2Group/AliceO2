#pragma once

#include <gpucf/executable/Executable.h>
#include <gpucf/ClEnv.h>

#include <nonstd/optional.hpp>


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

};

}

// vim: set ts=4 sw=4 sts=4 expandtab:

