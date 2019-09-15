// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/log.h>
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

    Map() = default;

    Map(T fallback) : fallback(fallback) {}

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

    Map(nonstd::span<const Digit> keys, nonstd::span<const T> pred, T fallback)
        : fallback(fallback)
    {
        ASSERT(keys.size() == pred.size());
        for (size_t i = 0; i < size_t(keys.size()); i++)
        {
            data[keys[i]] = pred[i];
        }
    }

    const T &operator[](const Position &p) const
    {
        auto lookup = data.find(p);
        return (lookup == data.end()) ? fallback : lookup->second;
    }

    void insert(const Position &p, const T &val)
    {
        data[p] = val;
    }

private:

    std::unordered_map<Position, T> data;

    T fallback;

}; 

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
