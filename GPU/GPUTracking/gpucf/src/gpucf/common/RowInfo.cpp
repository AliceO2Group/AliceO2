#include "RowInfo.h"

#include <numeric>


const std::array<int, RowInfo::regionNum> RowInfo::rowsPerRegion =
    { 17, 15, 16, 15, 18, 16, 16, 14, 13, 12 };


const RowInfo &RowInfo::instance()
{
    static RowInfo theInstance;

    return theInstance;
}


RowInfo::RowInfo()
    : globalToLocalMap(numOfRows())
    , localToGlobalMap(regionNum)
{
    int globalRow = 0;
    for (int region = 0; region < regionNum; region++)
    {
        int rows = rowsPerRegion[region];
        for (int localRow  = 0; localRow < rows; localRow++)
        {
            globalToLocalMap[globalRow] = localRow;
            localToGlobalMap[region].push_back(globalRow);
            globalRowToCruMap.push_back(region);
            globalRow++;
        }
    }
}

int RowInfo::globalRowToCru(int row) const
{
    return globalRowToCruMap.at(row);
}

int RowInfo::globalToLocal(int row) const
{
    return globalToLocalMap.at(row);
}

int RowInfo::localToGlobal(int cru, int row) const
{
    return localToGlobalMap.at(cru).at(row);
}

int RowInfo::numOfRows() const
{
    return std::accumulate(rowsPerRegion.begin(), rowsPerRegion.end(), 0); 
}

// vim: set ts=4 sw=4 sts=4 expandtab:
