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

#include <gpucf/algorithms/ClusterFinderState.h>
#include <gpucf/algorithms/StreamCompaction.h>

#include <nonstd/optional.hpp>


namespace gpucf
{

class CompactClusters
{

public:
    CompactClusters(ClEnv env, size_t digitnum)
    {
        sc.setup(env, StreamCompaction::CompType::Cluster, 1, digitnum);
        worker = sc.worker();
    }

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        state.cutoffClusternum = worker->run(
                state.peaknum, 
                queue,
                state.clusterNative, 
                state.clusterNativeCutoff, 
                state.aboveQTotCutoff);
    }

    Step step()
    {
        return worker->asStep("compactCluster");
    }
    

private:
    StreamCompaction sc;    
    nonstd::optional<StreamCompaction::Worker> worker;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
