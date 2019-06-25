#pragma once

#include <gpucf/common/ClEnv.h>
#include <gpucf/common/Digit.h>
#include <gpucf/executable/Executable.h>
#include <gpucf/experiments/Experiment.h>


namespace gpucf
{
    
class Benchmark : public Executable
{
public:
    Benchmark();

protected:
    void setupFlags(args::Group &, args::Group &) override;
    int mainImpl() override;

private:
    std::unique_ptr<ClEnv::Flags> envFlags;
    OptStringFlag digitFile;
    OptStringFlag benchmarkDir;
    OptIntFlag iterations;

    filesystem::path baseDir;

    std::vector<std::shared_ptr<Experiment>> experiments;

    std::vector<Digit> digits;

    void registerExperiments();

    void runExperiments();
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
