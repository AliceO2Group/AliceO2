#pragma once

#include <gpucf/common/buffer.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/log.h>


namespace gpucf
{
    
class ClusterToCPU
{

public:

    std::vector<Cluster> call(
            ClusterFinderState &state, 
            bool compacted,
            cl::CommandQueue queue)
    {
        static_assert(sizeof(row_t) == sizeof(unsigned char));

        std::vector<unsigned char> rows(state.peaknum);

        size_t clusternum = 
            (compacted) ? state.cutoffClusternum : state.peaknum;

        std::vector<ClusterNative> cn(clusternum);

        cl::Buffer clustersrc = 
            (compacted) ? state.clusterNativeCutoff : state.clusterNative;

        gpucpy<row_t>(state.rows, rows, rows.size(), queue, true);
        gpucpy<ClusterNative>(clustersrc, cn, cn.size(), queue, true);

        std::vector<Cluster> cluster;
        if (compacted)
        {
            std::vector<unsigned char> abovecutoff(state.peaknum);
            
            gpucpy<cl_uchar>(
                    state.aboveQTotCutoff,
                    abovecutoff,
                    abovecutoff.size(),
                    queue,
                    true);

            size_t cutpos = 0;
            for (size_t i = 0; i < abovecutoff.size(); i++)
            {
                if (abovecutoff[i])
                {
                    cluster.emplace_back(rows[i], cn[cutpos]);
                    cutpos++;
                }
            }
        }
        else
        {
            ASSERT(cn.size() == rows.size());
            for (size_t i = 0; i < cn.size(); i++)
            {
                cluster.emplace_back(rows[i], cn[i]);
            }
        }

        return cluster;
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
