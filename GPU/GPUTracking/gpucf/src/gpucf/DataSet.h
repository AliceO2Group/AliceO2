#pragma once

#include <gpucf/Object.h>

#include <filesystem/fwd.h>

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
