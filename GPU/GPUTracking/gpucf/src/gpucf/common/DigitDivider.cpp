#include "DigitDivider.h"

#include <gpucf/common/log.h>

#include <cmath>


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
    int end   = digits.back().time;

    DBG(start);
    DBG(end);
    DBG(chunksRequested);

    stepsPerChunk = std::ceil((end - start) / float(chunksRequested));
}

optional<DigitDivider::Chunk> DigitDivider::nextChunk(size_t padding)
{
    if (start >= size_t(digits.size()))
    {
        return nullopt;
    }

    Chunk res;
    res.start = start;

    DBG(currTime());
    DBG(stepsPerChunk);

    size_t end = timeSliceEnd(currTime() + stepsPerChunk);
    res.items = end - start;


    size_t paddedEnd = timeSliceEnd(currTime() + stepsPerChunk + padding);
    res.future = paddedEnd - end;

    start = paddedEnd;

    return res;
}

size_t DigitDivider::timeSliceEnd(int time)
{
    size_t left = start;
    size_t right = digits.size() - 1;

    DBG(time);

    if (time >= digits.back().time)
    {
        return digits.size();
    }

    while (left <= right)
    {
        size_t pos = (left + right) / 2; 

        DBG(left);
        DBG(right);
        DBG(pos);

        int lt = digits[pos].time;
        int rt = digits[pos+1].time;

        DBG(lt);
        DBG(rt);

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

int DigitDivider::currTime() const
{
    return digits.at(start).time;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
