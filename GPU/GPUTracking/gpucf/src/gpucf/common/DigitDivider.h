#pragma once

#include <gpucf/common/Digit.h>

#include <nonstd/optional.hpp>
#include <nonstd/span.hpp>


namespace gpucf
{

class DigitDivider
{

public: 
    DigitDivider(nonstd::span<const Digit>, size_t);

    struct Chunk
    {
        size_t start;
        size_t items;
        size_t future;
    };
    nonstd::optional<Chunk> nextChunk(size_t);

private:
    nonstd::span<const Digit> digits;

    size_t chunksRequested = 0;

    size_t stepsPerChunk = 0;

    size_t start = 0;

    size_t timeSliceEnd(int);

    int currTime() const;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
