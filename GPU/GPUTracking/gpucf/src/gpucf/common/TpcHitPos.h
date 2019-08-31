#pragma once

#include <gpucf/common/MCLabel.h>


namespace gpucf
{

struct TpcHitPos
{
    short sector;
    short row;
    MCLabel label;

    bool operator==(const TpcHitPos &other) const
    {
        return sector == other.sector 
            && row == other.row 
            && label == other.label;
    }
};

} // namespace gpucf


namespace std
{

    template<>
    struct hash<gpucf::TpcHitPos>
    {
        size_t operator()(const gpucf::TpcHitPos &p) const
        {
            static_assert(sizeof(p.label.event) == sizeof(short), "");
            static_assert(sizeof(p.label.track) == sizeof(short), "");

            size_t h = (size_t(p.sector)      << (8*3*sizeof(short)))
                     | (size_t(p.row)         << (8*2*sizeof(short)))
                     | (size_t(p.label.event) << (8*1*sizeof(short)))
                     | size_t(p.label.track);
            return std::hash<size_t>()(h);
        } 
    };

} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:
