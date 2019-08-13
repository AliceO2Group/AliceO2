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

