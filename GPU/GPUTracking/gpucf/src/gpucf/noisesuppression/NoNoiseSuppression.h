#pragma once

#include <gpucf/noisesuppression/NoiseSuppression.h>


namespace gpucf
{

class NoNoiseSuppression : public NoiseSuppression
{

public:

    NoNoiseSuppression() : NoiseSuppression("unfiltered")
    {
    }

protected:

    std::vector<Digit> runImpl(
            View<Digit> digits, 
            const Map<bool> &, 
            const Map<float> &)
    {
        return std::vector<Digit>(digits.begin(), digits.end());
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

