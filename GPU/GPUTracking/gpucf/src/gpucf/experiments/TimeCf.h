#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/experiments/Experiment.h>
#include <gpucf/algorithms/GPUClusterFinder.h>

#include <nonstd/span.hpp>


namespace gpucf
{

class TimeCf : public Experiment
{
public:
    TimeCf( const std::string &,
            filesystem::path,
            GPUClusterFinder::Config,
            nonstd::span<const Digit>, 
            size_t, 
            filesystem::path);

    void run(ClEnv &) override;

private:
    std::string name;
    filesystem::path tgtFile;

    GPUClusterFinder::Config config;
    size_t repeats;
    nonstd::span<const Digit> digits;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
