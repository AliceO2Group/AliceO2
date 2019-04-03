#pragma once

#include <gpucf/executable/Executable.h>
#include <gpucf/ClEnv.h>


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
    OptFlag chargemapIdxMacro;
    OptFlag tiling4x4;
    OptFlag tiling4x8;
    OptFlag tiling8x4;
    OptFlag padMajor;
    OptFlag halfs;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

