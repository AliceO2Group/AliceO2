#include "DigitDrawer.h"

#include <gpucf/common/log.h>

#include <iomanip>
#include <sstream>


using namespace gpucf;


DigitDrawer::DigitDrawer(
        nonstd::span<const Digit> digits,
        nonstd::span<unsigned char> isPeakGT,
        nonstd::span<unsigned char> isPeak)
    : chargeMap(digits, [](const Digit &d) { return d.charge; }, 0.f)
    , peakGTMap(digits, isPeakGT, 0)
    , peakMap(digits, isPeak, 0)
{
}

DigitDrawer::DigitDrawer(
        nonstd::span<const Digit> digits,
        nonstd::span<const Digit> peaksGT,
        nonstd::span<const Digit> peaks)
    : chargeMap(digits, [](const Digit &d) { return d.charge; }, 0.f)
    , peakGTMap(peaksGT, 1, 0)
    , peakMap(peaks, 1, 0)
{
}

std::string DigitDrawer::drawArea(const Digit &center, int radius)
{
    std::stringstream ss;
    for (int dp = -radius; dp <= radius; dp++)
    {
        for (int dt = -radius; dt <= radius; dt++)
        {
            printAt(ss, {center, dp, dt}); 
            ss << " ";
        }
        ss << "\n";
    }

    return ss.str();
}

std::string DigitDrawer::toFixed(float q)
{
    std::stringstream ss;
    ss << std::setw(3) << std::setfill('0') << static_cast<int>(q); 

    return ss.str();
}

void DigitDrawer::printAt(std::stringstream &ss, const Position &p)
{
    bool isPeakGT = peakGTMap[p];
    bool isPeak   = peakMap[p];

    if (isPeakGT && isPeak)
    {
        ss << log::Formatter<log::Format::Blue>();
    }
    else if (isPeakGT)
    {
        ss << log::Formatter<log::Format::Green>();
    }
    else if (isPeak)
    {
        ss << log::Formatter<log::Format::Red>();
    }

    ss << toFixed(chargeMap[p]);

    ss << log::Formatter<log::Format::DefaultColor>();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
