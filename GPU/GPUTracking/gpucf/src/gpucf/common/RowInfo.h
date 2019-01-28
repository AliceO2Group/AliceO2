#pragma once

#include <array>
#include <vector>


class RowInfo
{

public:
    static const RowInfo &instance();

    static constexpr int regionNum = 10;
    static const std::array<int, regionNum> rowsPerRegion;

    std::vector<int>              globalToLocalMap;
    std::vector<std::vector<int>> localToGlobalMap;

    RowInfo();

    int globalToLocal(int) const;

    int localToGlobal(int, int) const;

    int numOfRows() const;
};

// vim: set ts=4 sw=4 sts=4 expandtab:
