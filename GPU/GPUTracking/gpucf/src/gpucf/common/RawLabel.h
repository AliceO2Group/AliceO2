#pragma once

#include <cstdint>


namespace gpucf
{

struct RawLabel
{
    int32_t id;
    int32_t event;
    int32_t track;
    int16_t isNoise;
    int16_t isSet;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
