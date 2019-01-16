#pragma once

#include <array>
#include <vector>


class RowInfo
{

public:
    static constexpr int regionNum = 10;
    static const std::array<int, regionNum> rowsPerRegion;

    static std::vector<int> globalRowToLocalRowMap();
    static int numOfRows();

};

// vim: set ts=4 sw=4 sts=4 expandtab:
