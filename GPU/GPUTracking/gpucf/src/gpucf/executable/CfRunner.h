#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/executable/CfCLIFlags.h>
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
    std::unique_ptr<CfCLIFlags> cfflags;
    OptStringFlag digitFile;
    OptStringFlag clusterResultFile;
    OptStringFlag peakFile;

    OptFlag cpu;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

