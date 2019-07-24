#include "cpu.h"


using namespace gpucf;


std::vector<Digit> gpucf::findPeaks(
        View<Digit> digits,
        const Map<float> &chargeMap)
{
    std::vector<Digit> peaks;

    for (const Digit &d : digits)
    {
        if (isPeak(d, chargeMap))
        {
            peaks.push_back(d);
        }
    }

    return peaks;
}


RowMap<std::vector<Digit>> gpucf::findPeaksByRow(
        View<Digit> digits, 
        const Map<float> &chargeMap)
{
    RowMap<std::vector<Digit>> peaks;

    for (const Digit &d : digits)
    {
        if (isPeak(d, chargeMap))
        {
            peaks[d.row].push_back(d);
        }
    }

    return peaks;
}

bool gpucf::isPeak(const Digit &d, const Map<float> &chargeMap, float cutoff)
{
    const float q = d.charge;

    if (q <= cutoff)
    {
        return false;
    }

    bool peak = true;
    
    peak &= chargeMap[{d, -1, -1}] <= q;
    peak &= chargeMap[{d, -1,  0}] <= q;
    peak &= chargeMap[{d, -1,  1}] <= q;
    peak &= chargeMap[{d,  0, -1}] <= q;
    peak &= chargeMap[{d,  0,  1}] <  q;
    peak &= chargeMap[{d,  1, -1}] <  q;
    peak &= chargeMap[{d,  1,  0}] <  q;
    peak &= chargeMap[{d,  1,  1}] <  q;

    return peak;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
