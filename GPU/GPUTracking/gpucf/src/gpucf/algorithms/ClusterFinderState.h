// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/ClEnv.h>

#include <CL/cl2.hpp>


namespace gpucf
{
    
struct ClusterFinderState
{
    ClusterFinderConfig cfg;
    
    size_t digitnum = 0;
    cl::Buffer digits;

    cl::Buffer isPeak;

    size_t peaknum = 0;
    cl::Buffer peaks;

    size_t filteredPeakNum = 0;
    cl::Buffer filteredPeaks;

    cl::Buffer chargeMap;
    cl::Buffer peakMap;
    
    size_t maxClusterPerRow = 0;
    cl::Buffer clusterInRow;
    cl::Buffer clusterByRow;

    ClusterFinderState(ClusterFinderConfig, size_t, cl::Context, cl::Device);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
