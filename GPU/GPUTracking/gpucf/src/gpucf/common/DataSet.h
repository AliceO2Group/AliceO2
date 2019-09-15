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

#include <gpucf/common/Object.h>

#include <filesystem/path.h>

#include <string>
#include <vector>


namespace gpucf
{

class DataSet
{

public:
    void read(const filesystem::path &);
    void write(const filesystem::path &) const;

    std::vector<Object> get() const;

    template<class T>
    void serialize(const std::vector<T> &in)
    {
        objs.clear();
        objs.reserve(in.size());

        for (const T &o : in)
        {
            objs.push_back(o.serialize());
        }
    }

    template<class T>
    std::vector<T> deserialize() const
    {
        std::vector<T> out;
        out.reserve(objs.size());

        for (const Object &o : objs)
        {
            out.emplace_back();
            out.back().deserialize(o);
        }

        return out;
    }

private:
    std::vector<Object> objs;
    
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
