#pragma once

namespace gpucf
{

    enum class ChargemapLayout
    {
        TimeMajor,
        PadMajor,
        Tiling4x4,
        Tiling8x4,
        Tiling4x8,
    };

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
