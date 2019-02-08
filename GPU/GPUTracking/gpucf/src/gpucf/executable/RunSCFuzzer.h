#pragma once

#include <gpucf/ClEnv.h>
#include <gpucf/executable/Executable.h>


namespace gpucf
{
    
class RunSCFuzzer : public Executable
{
public:
    RunSCFuzzer();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;

    OptIntFlag numRuns;

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
