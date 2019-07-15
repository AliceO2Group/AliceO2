#pragma once

#include <cstdint>


namespace gpucf
{

struct RawLabel
{
    int32_t id;
    int32_t source;
    int32_t event;
    int32_t track;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
