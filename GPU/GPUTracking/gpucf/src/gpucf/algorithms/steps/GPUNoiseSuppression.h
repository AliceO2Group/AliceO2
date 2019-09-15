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

class GPUNoiseSuppression
{

public:

    GPUNoiseSuppression(cl::Program prg)
        : noiseSuppression("noiseSuppression", prg)
        , updatePeaks("updatePeaks", prg)
    {
    }

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        bool scratchpad = (state.cfg.clusterbuilder == ClusterBuilder::ScratchPad);
        size_t dummyItems = (scratchpad) ? 64 - (state.peaknum % 64) : 0;

        size_t workitems = state.peaknum + dummyItems;

        noiseSuppression.setArg(0, state.chargeMap);
        noiseSuppression.setArg(1, state.peakMap);
        noiseSuppression.setArg(2, state.peaks);
        noiseSuppression.setArg(3, static_cast<cl_uint>(state.peaknum));
        noiseSuppression.setArg(4, state.isPeak);

        noiseSuppression.call(0, workitems, 64, queue);


        updatePeaks.setArg(0, state.peaks);
        updatePeaks.setArg(1, state.isPeak);
        updatePeaks.setArg(2, state.peakMap);

        updatePeaks.call(0, state.peaknum, 64, queue);
    }

    Step asStep(const std::string &name) const
    {
        Timestamp start = noiseSuppression.getEvent().start();
        Timestamp end   = updatePeaks.getEvent().end();

        return {name, start, start, start, end};
    }

private:

    Kernel1D noiseSuppression;
    Kernel1D updatePeaks;
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
