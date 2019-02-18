#include "DigitDivider.h"


using namespace gpucf;

using nonstd::optional;
using nonstd::nullopt;
using nonstd::span;


DigitDivider::DigitDivider(
        nonstd::span<const Digit> ds,
        size_t                    cs)
    : digits(ds)
    , chunksRequested(cs)
{
    int start = digits.front().time;
    int end   = digits.end().time;
    stepsPerChunk = std::ceil((end - start) / float(chunksRequested));
}

optional<Chunk> DigitDivider::nextChunk(int padding)
{
    if (start >= digits.size())
    {
        return nullopt;
    }

    size_t end = timeSliceEnd(currTime() + stepsPerChunk);

    Chunk res;
    res.digits = digits.subspan(start, end);

    size_t paddedEnd = timeSliceEnd(currTime() + stepsPerChunk + padding);
    res.paddedDigits = digits.subspan(start, paddedEnd);

    start = end;

    return res;
}

size_t DigitDivider::timeSliceEnd(int time)
{
    size_t left = start;
    size_t right = digit.size() - 1;

    while (left < right)
    {
        size_t pos = (left + right) / 2; 

        int lt = digits[pos].time;
        int rt = digits[pos+1].time;

        if (lt <= time)
        {
            if (rt > time)
            {
                return pos;
            }
            else
            {
                left = pos + 1;
            }
        }
        else
        {
            right = pos - 1;
        }
    }

    ASSERT(false); // Should never be reached

    return 0;
}

size_t DigitDivider::currTime() const
{
    return digits.at(start).time;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
