#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>

#include <nonstd/span.hpp>

#include <sstream>
#include <vector>


namespace gpucf
{

class DigitDrawer
{

public:

    DigitDrawer(
            nonstd::span<const Digit>, 
            nonstd::span<unsigned char>, 
            nonstd::span<unsigned char>);

    DigitDrawer(
            nonstd::span<const Digit>, 
            nonstd::span<const Digit>,
            nonstd::span<const Digit>);

    std::string drawArea(const Digit &, int r);

private:

    Map<float> chargeMap;
    Map<unsigned char>  peakGTMap;
    Map<unsigned char>  peakMap;

    std::string toFixed(float);

    void printAt(std::stringstream &, const Position &);

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
