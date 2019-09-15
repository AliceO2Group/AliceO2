// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "NoiseSuppressionOverArea.h"


using namespace gpucf;


NoiseSuppressionOverArea::NoiseSuppressionOverArea(
        int radPad, 
        int radTime, 
        int cutoff, 
        int epsilon)
    : NoiseSuppression("Bereich " + std::to_string(radPad*2+1) 
        + "x" + std::to_string(radTime*2+1) )
    , radPad(radPad)
    , radTime(radTime)
    , cutoff(cutoff)
    , epsilon(epsilon)
{
    outerToInner = {
        { {-2, -2}, {{-1, -1}} },
        { {-2,  2}, {{-1,  1}} },
        { { 2, -2}, {{ 1, -1}} },
        { { 2,  2}, {{ 1,  1}} },

        { {-2,  0}, {{-1,  0}} },
        { { 0, -2}, {{ 0, -1}} },
        { { 0,  2}, {{ 0,  1}} },
        { { 2,  0}, {{ 1,  0}} },

        { {-2,  1}, {{-1,  1}} },
        { {-2, -1}, {{-1, -1}} },
        { {-1, -2}, {{-1, -1}} },
        { {-1,  2}, {{-1,  1}} },
        { { 1, -2}, {{ 1, -1}} },
        { { 1,  2}, {{ 1,  1}} },
        { { 2, -1}, {{ 1, -1}} },
        { { 2,  1}, {{ 1,  1}} },
        /* { {-2,  1}, {{-1,  1}, {-1,  0} }}, */
        /* { {-2, -1}, {{-1, -1}, {-1,  0} }}, */
        /* { {-1, -2}, {{-1, -1}, { 0, -1} }}, */
        /* { {-1,  2}, {{-1,  1}, { 0,  1} }}, */
        /* { { 1, -2}, {{ 1, -1}, { 0, -1} }}, */
        /* { { 1,  2}, {{ 1,  1}, { 0,  1} }}, */
        /* { { 2, -1}, {{ 1, -1}, { 1,  0} }}, */
        /* { { 2,  1}, {{ 1,  1}, { 1,  0} }}, */

        { {-3, -3}, {{-2, -2}, {-1, -1}} },
        { {-3,  3}, {{-2,  2}, {-1,  1}} },
        { { 3, -3}, {{ 2, -2}, { 1, -1}} },
        { { 3,  3}, {{ 2,  2}, { 1,  1}} },

        { {-3,  0}, {{-2,  0}, {-1,  0}} },
        { { 0, -3}, {{ 0, -2}, { 0, -1}} },
        { { 0,  3}, {{ 0,  2}, { 0,  1}} },
        { { 3,  0}, {{ 2,  0}, { 1,  0}} },

        { {-3, -2}, {{-2, -1}, {-1, -1}} },
        { {-2, -3}, {{-1, -2}, {-1, -1}} },
        { {-3,  2}, {{-2,  1}, {-1,  1}} },
        { {-2,  3}, {{-1,  2}, {-1,  1}} },
        { { 3, -2}, {{ 2, -1}, { 1, -1}} },
        { { 2, -3}, {{ 1, -2}, { 1, -1}} },
        { { 3,  2}, {{ 2,  1}, { 1,  1}} },
        { { 2,  3}, {{ 1,  2}, { 1,  1}} },

        { { 3,  1}, {{ 2,  1}, { 1,  1}} },
        { { 1,  3}, {{ 1,  2}, { 1,  1}} },
        { {-3,  1}, {{-2,  1}, {-1,  1}} },
        { {-1,  3}, {{-1,  2}, {-1,  1}} },
        { { 3, -1}, {{ 2, -1}, { 1, -1}} },
        { { 1, -3}, {{ 1, -2}, { 1, -1}} },
        { {-3, -1}, {{-2, -1}, {-1, -1}} },
        { {-1, -3}, {{-1, -2}, {-1, -1}} },

        { { 4,  3}, {{ 3,  2}, { 2,  2}, { 1,  1}} },
        { { 4,  2}, {{ 3,  2}, { 2,  1}, { 1,  1}} },
        { { 4,  1}, {{ 3,  1}, { 2,  1}, { 1,  1}} },
        { {-4,  3}, {{-3,  2}, {-2,  2}, {-1,  1}} },
        { {-4,  2}, {{-3,  2}, {-2,  1}, {-1,  1}} },
        { {-4,  1}, {{-3,  1}, {-2,  1}, {-1,  1}} },
        { { 4, -3}, {{ 3, -2}, { 2, -2}, { 1, -1}} },
        { { 4, -2}, {{ 3, -2}, { 2, -1}, { 1, -1}} },
        { { 4, -1}, {{ 3, -1}, { 2, -1}, { 1, -1}} },
        { {-4, -3}, {{-3, -2}, {-2, -2}, {-1, -1}} },
        { {-4, -2}, {{-3, -2}, {-2, -1}, {-1, -1}} },
        { {-4, -1}, {{-3, -1}, {-2, -1}, {-1, -1}} },

        { { 4,  0}, {{ 3,  0}, { 2,  0}, { 1,  0}} },
        { {-4,  0}, {{-3,  0}, {-2,  0}, {-1,  0}} },
    };

    for (auto &p : outerToInner)
    {
        if (std::abs(p.first.pad) <= radPad 
                && std::abs(p.first.time) <= radTime)
        {
            neighbors.push_back(p);    
        }
    }
}


std::vector<Digit> NoiseSuppressionOverArea::runImpl(
        View<Digit> peaks,
        const Map<bool> &peakMap,
        const Map<float> &chargeMap)
{
    std::vector<Digit> filtered;

    for (const Digit &p : peaks)
    {
        if (p.charge <= cutoff)
        {
            continue;
        }

        bool keepMe = true;
        for (const auto &n : neighbors)
        {
            int dp = n.first.pad;
            int dt = n.first.time;

            Position other(p, dp, dt);

            bool otherIsPeak = peakMap[other];
            if (!otherIsPeak)
            {
                continue;
            }

            float q = p.charge;
            float oq = chargeMap[other];
            if (oq <= q)
            {
                continue;
            }

            bool hasMinima = false;
            for (const Delta &b : n.second)
            {
                Position between(p, b.pad, b.time);

                float bq = chargeMap[between];

                hasMinima |= (q - bq > epsilon);
            }

            keepMe &= hasMinima;

            if (!keepMe)
            {
                break;
            }
        }

        if (keepMe)
        {
            filtered.push_back(p);
        }

    }

    return filtered;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
