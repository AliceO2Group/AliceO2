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
        nonstd::span<const Digit> digits;    
        nonstd::span<const Digit> digitsPadded;
    };
    nonstd::optional<Chunk> nextChunk(size_t);

private:
    nonstd::span<const Digit> digits;

    size_t chunksRequested = 0;

    size_t stepsPerChunk = 0;

    size_t start = 0;

    size_t timeSliceEnd(size_t);

    size_t currTime();

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
