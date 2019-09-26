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

#include <gpucf/common/Kernel1D.h>

namespace gpucf
{

class ComputeCluster : public Kernel1D
{

 public:
  DECL_KERNEL(ComputeCluster, "computeClusters");

  void call(ClusterFinderState& state, cl::CommandQueue queue)
  {
    bool scratchpad = (state.cfg.clusterbuilder == ClusterBuilder::ScratchPad);
    size_t dummyItems = (scratchpad)
                          ? state.cfg.wgSize - (state.filteredPeakNum % state.cfg.wgSize)
                          : 0;
    size_t workitems = state.filteredPeakNum + dummyItems;

    kernel.setArg(0, state.chargeMap);
    kernel.setArg(1, state.filteredPeaks);
    kernel.setArg(2, cl_uint(state.filteredPeakNum));
    kernel.setArg(3, cl_uint(state.maxClusterPerRow));
    kernel.setArg(4, state.clusterInRow);
    kernel.setArg(5, state.clusterByRow);

    Kernel1D::call(0, workitems, state.cfg.wgSize, queue);
  }
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
