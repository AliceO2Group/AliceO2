#pragma once

#include <functional>


namespace gpucf
{

struct Delta
{

    int time;
    int pad;

    bool operator==(const Delta &other) const
    {
        return other.time == time && other.pad == pad;
    }

};

} // namespace gpucf


namespace std
{

    template<>
    struct hash<gpucf::Delta>
    {
        size_t operator()(const gpucf::Delta &d) const
        {
            static_assert(sizeof(size_t) >= 2*sizeof(int));
            size_t h = size_t(d.time) | (size_t(d.pad) << sizeof(int));
            return std::hash<size_t>()(h);
        }
    };

} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:

