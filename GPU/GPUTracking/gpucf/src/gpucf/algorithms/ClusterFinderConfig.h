#pragma once

#include <gpucf/algorithms/ChargemapLayout.h>
#include <gpucf/algorithms/ClusterBuilder.h>

#include <cstddef>


namespace gpucf
{

struct ClusterFinderConfig
{
    size_t chunks = 1;

    bool usePackedDigits = true;

    bool halfPrecisionCharges = false;

    bool splitCharges = false;

    ChargemapLayout layout = ChargemapLayout::TimeMajor;

    ClusterBuilder clusterbuilder = ClusterBuilder::Naive;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
