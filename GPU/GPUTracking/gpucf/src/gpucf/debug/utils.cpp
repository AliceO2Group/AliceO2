// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "utils.h"


using namespace gpucf;


std::vector<Digit> gpucf::digitize(const Array2D<float> &charges)
{
    std::vector<Digit> digits;

    for (size_t pad = 0; pad < charges.size(); pad++)
    {
        for (size_t time = 0; time < charges[pad].size(); time++)
        {
            float q = charges[pad][time];
            if (q > 0.f)
            {
                digits.emplace_back(q, 0, pad, time);
            }
        }
    }

    return digits;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

