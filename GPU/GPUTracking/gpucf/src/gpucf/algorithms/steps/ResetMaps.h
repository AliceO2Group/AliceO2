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

class ResetMaps : public Kernel1D
{

public:
    DECL_KERNEL(ResetMaps, "resetMaps");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        kernel.setArg(0, state.digits);
        kernel.setArg(1, state.chargeMap);
        kernel.setArg(2, state.peakMap);
        Kernel1D::call(0, state.digitnum, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
