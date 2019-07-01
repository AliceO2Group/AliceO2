#pragma once

#include <gpucf/common/Position.h>

#include <nonstd/span.hpp>

#include <functional>
#include <unordered_map>


namespace gpucf
{

template<typename T>
class Map
{

public:
    using Predicate = std::function<T(const Digit &)>;

    Map(nonstd::span<const Digit> keys, Predicate pred, T fallback)
        : fallback(fallback)
    {
        for (const Digit &d : keys)
        {
            data[d] = pred(d);
        }
    }

    Map(nonstd::span<const Digit> keys, T pred, T fallback)
        : Map(keys, [pred](const Digit &) { return pred; }, fallback)
    {
    }

    const T &operator[](const Position &p) const
    {
        auto lookup = data.find(p);
        return (lookup == data.end()) ? fallback : lookup->second;
    }

private:
    std::unordered_map<Position, T> data;

    T fallback;

}; 

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
