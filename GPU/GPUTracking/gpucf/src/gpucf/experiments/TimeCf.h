#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/Digit.h>
#include <gpucf/experiments/Experiment.h>

#include <nonstd/span.hpp>


namespace gpucf
{

class TimeCf : public Experiment
{
public:
    TimeCf( const std::string &,
            filesystem::path,
            ClusterFinderConfig,
            nonstd::span<const Digit>, 
            size_t);

    void run(ClEnv &) override;

private:
    std::string name;
    filesystem::path tgtFile;

    size_t repeats;
    nonstd::span<const Digit> digits;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
