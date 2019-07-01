#pragma once

#include <gpucf/common/Digit.h>

#include <shared/tpc.h>


namespace gpucf
{

class Position
{

public:
    row_t     row;
    pad_t     pad;
    timestamp time;

    Position(const Digit &);
    Position(const Digit &, int, int);
    Position(row_t, pad_t, timestamp);
    
    bool operator==(const Position &) const;

    size_t idx() const;
};

} // namespace gpucf


namespace std 
{

    template<> 
    struct hash<gpucf::Position>
    {

        size_t operator()(const gpucf::Position &p) const
        {
            return std::hash<size_t>()(p.idx());
        }

    };

} // namespace std

// vim: set ts=4 sw=4 sts=4 expandtab:
