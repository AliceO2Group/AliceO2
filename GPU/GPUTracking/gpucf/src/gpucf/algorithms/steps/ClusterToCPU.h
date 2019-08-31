#pragma once

#include <gpucf/common/buffer.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>


namespace gpucf
{
    
class ClusterToCPU
{

public:

    std::vector<Cluster> call(
            ClusterFinderState &state, 
            cl::CommandQueue queue)
    {
        static_assert(sizeof(row_t) == sizeof(unsigned char), "");
        static_assert(sizeof(cl_uint) == sizeof(unsigned int), "");

        std::vector<unsigned int> clusterInRow(RowInfo::instance().numOfRows());

        gpucpy<cl_uint>(
                state.clusterInRow, 
                clusterInRow, 
                clusterInRow.size(), 
                queue,
                true);

        std::vector<std::vector<ClusterNative>> clusterByRow;

        for (size_t i = 0; i < clusterInRow.size(); i++)
        {
            size_t clusternum = clusterInRow[i];
            clusterByRow.emplace_back(clusternum);

            bool blocking = (i == clusterInRow.size()-1);
            gpucpy<ClusterNative>(
                    state.clusterByRow,
                    clusterByRow.back(),
                    clusternum,
                    queue,
                    blocking,
                    state.maxClusterPerRow * i);
        }

        std::vector<Cluster> cluster;
        for (size_t row = 0; row < clusterByRow.size(); row++)
        {
            for (const ClusterNative &cn : clusterByRow[row])
            {
                cluster.emplace_back(row, cn);
            }
        }

        return cluster;
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
