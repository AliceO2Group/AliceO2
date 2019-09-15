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

#include <gpucf/noisesuppression/Delta.h>
#include <gpucf/noisesuppression/NoiseSuppression.h>


namespace gpucf
{

class NoiseSuppressionOverArea : public NoiseSuppression
{

public:

    NoiseSuppressionOverArea(int, int, int, int);

protected:

    std::vector<Digit> runImpl(
            View<Digit>, 
            const Map<bool> &, 
            const Map<float> &) override;

private:

    std::unordered_map<Delta, std::vector<Delta>> outerToInner;

    std::vector<std::pair<Delta, std::vector<Delta>>> neighbors;

    int radPad;
    int radTime;
    int cutoff;
    int epsilon;

};

} // namespace gpucf


// vim: set ts=4 sw=4 sts=4 expandtab:
