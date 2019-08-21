#include "ClusterFinderState.h"

#include <gpucf/common/buffer.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/common/RowInfo.h>

#include <shared/ClusterNative.h>
#include <shared/Digit.h>


using namespace gpucf;


ClusterFinderState::ClusterFinderState(
        ClusterFinderConfig cfg, 
        size_t digitnum, 
        cl::Context context,
        cl::Device  device)
    : cfg(cfg)
    , digitnum(digitnum)
{
    digits = makeBuffer<Digit>(digitnum, Memory::ReadWrite, context);
    peaks  = makeBuffer<Digit>(digitnum, Memory::ReadWrite, context);

    isPeak = makeBuffer<cl_uchar>(digitnum, Memory::ReadWrite, context);

    std::vector<int> globalToLocalRowMap = RowInfo::instance().globalToLocalMap;
    std::vector<int> globalRowToCruMap = RowInfo::instance().globalRowToCruMap;
    const size_t numOfRows = globalToLocalRowMap.size();

    const size_t mapEntries = TPC_NUM_OF_PADS * TPC_MAX_TIME_PADDED; 

    chargeMap = makeBuffer<cl_ushort>(mapEntries, Memory::ReadWrite, context);

    peakMap = makeBuffer<cl_uchar>(mapEntries, Memory::ReadWrite, context);


    maxClusterPerRow = 0.01f * digitnum;

    clusterInRow = makeBuffer<cl_uint>(numOfRows, Memory::ReadWrite, context);

    clusterByRow = makeBuffer<ClusterNative>(
            maxClusterPerRow * numOfRows, 
            Memory::ReadWrite, 
            context);


    cl::CommandQueue init(context, device);

    fill<cl_ushort>(chargeMap, 0, mapEntries, init);
    fill<cl_uchar>(peakMap, 0, mapEntries, init);
    fill<cl_uint>(clusterInRow, 0, numOfRows, init);

    init.finish();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
