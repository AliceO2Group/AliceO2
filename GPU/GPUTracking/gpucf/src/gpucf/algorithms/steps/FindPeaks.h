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

class FindPeaks : public Kernel1D
{

public:
    DECL_KERNEL(FindPeaks, "findPeaks");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        bool scratchpad = (state.cfg.clusterbuilder == ClusterBuilder::ScratchPad);
        size_t dummyItems = (scratchpad) 
            ? 64 - (state.digitnum % 64) : 0;

        size_t workitems = state.digitnum + dummyItems;

        kernel.setArg(0, state.chargeMap);
        kernel.setArg(1, state.digits);
        kernel.setArg(2, static_cast<cl_uint>(state.digitnum));
        kernel.setArg(3, state.isPeak);
        kernel.setArg(4, state.peakMap);

        Kernel1D::call(0, workitems, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
