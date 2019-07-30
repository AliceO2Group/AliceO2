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
