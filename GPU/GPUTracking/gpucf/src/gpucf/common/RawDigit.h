#pragma once

#include <cstdint>


namespace gpucf
{

struct RawDigit
{
    int32_t row;
    int32_t pad;
    int32_t time;
    float   charge;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
