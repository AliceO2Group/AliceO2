#include "NoiseSuppression.h"


using namespace gpucf;


RowMap<std::vector<Digit>> NoiseSuppression::run(
        const RowMap<std::vector<Digit>> &digits,
        const RowMap<Map<bool>> &isPeak,
        const Map<float> &chargeMap)
{
    RowMap<std::vector<Digit>> filteredPeaks;

    for (size_t row = 0; row < TPC_NUM_OF_ROWS; row++)
    {
        filteredPeaks[row] = runImpl(digits[row], isPeak[row], chargeMap);
    }

    return filteredPeaks;
}


// vim: set ts=4 sw=4 sts=4 expandtab:
