#include "RowInfo.h"

#include <numeric>


const std::array<int, RowInfo::regionNum> RowInfo::rowsPerRegion =
    { 17, 15, 16, 15, 18, 16, 16, 14, 13, 12 };


std::vector<int> RowInfo::globalRowToLocalRowMap()
{
    std::vector<int> globalToLocalRow(numOfRows());
    int globalRow = 0;
    for (int rows : rowsPerRegion)
    {
        for (int localRow  = 0; localRow < rows; localRow++)
        {
            globalToLocalRow[globalRow] = localRow;
            globalRow++;
        }
    }

    return globalToLocalRow;
}

int RowInfo::numOfRows()
{
    return std::accumulate(rowsPerRegion.begin(), rowsPerRegion.end(), 0); 
}

// vim: set ts=4 sw=4 sts=4 expandtab:
