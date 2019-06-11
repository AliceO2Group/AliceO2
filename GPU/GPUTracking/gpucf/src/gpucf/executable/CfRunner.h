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
    OptFlag tiling4x4;
    OptFlag tiling4x8;
    OptFlag tiling8x4;
    OptFlag scratchpad;
    OptFlag padMajor;
    OptFlag halfs;
    OptFlag splitCharges;
    OptFlag cpu;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

