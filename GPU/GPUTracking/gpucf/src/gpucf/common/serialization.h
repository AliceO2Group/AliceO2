#pragma once

#include <gpucf/common/SectorMap.h>

#include <filesystem/path.h>

#include <nonstd/span.hpp>

#include <cstdint>
#include <fstream>
#include <vector>


namespace gpucf
{

template<typename T>
struct SectorData
{
    SectorMap<uint64_t> elemsBySector;
    std::vector<T> data;

    size_t sizeByHeader() const 
    {
        size_t n = 0;
        for (auto s : elemsBySector)
        {
            n += s;    
        }
        return n;
    }
};

template<typename T>
SectorData<T> read(filesystem::path f)
{
    std::ifstream in(f.str(), std::ios::binary);
    
    SectorData<T> raw;
    in.read(reinterpret_cast<char *>(raw.elemsBySector.data()), 
            sizeof(uint64_t) * raw.elemsBySector.size());

    raw.data.resize(raw.sizeByHeader());

    in.read(reinterpret_cast<char *>(raw.data.data()), raw.data.size() * sizeof(T));

    return raw;
}

template<typename R>
void write(filesystem::path f, nonstd::span<const R> data)
{
    std::ofstream out(f.str(), std::ios::binary);

    uint64_t size = data.size();
    out.write(reinterpret_cast<const char *>(&size), sizeof(uint64_t));

    out.write(reinterpret_cast<const char *>(data.data()), size * sizeof(R));
}

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
