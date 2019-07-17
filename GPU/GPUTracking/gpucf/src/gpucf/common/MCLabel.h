#pragma once

#include <gpucf/common/RawLabel.h>

#include <functional>
#include <iosfwd>


namespace gpucf
{
    
struct MCLabel
{
    int event; 
    int track;

    MCLabel(const RawLabel &l)
        : event(l.event)
        , track(l.track)
    {
    }

    bool operator==(const MCLabel &other) const
    {
        return other.event == event && other.track == track;
    }

    bool operator<(const MCLabel &other) const
    {
        return event < other.event || (event == other.event && track < other.track);
    }

};

std::ostream &operator<<(std::ostream &o, const MCLabel &l);

} // namespace gpucf


namespace std
{
    template <>
    struct hash<gpucf::MCLabel>
    {
        size_t operator()(const gpucf::MCLabel &l) const
        {
            return std::hash<int>()(l.event) ^ std::hash<int>()(l.track);
        }
    };
    
} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:
