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

    globalToLocalRow = makeBuffer<cl_int>(numOfRows, Memory::ReadOnly, context);
    globalRowToCru = makeBuffer<cl_int>(numOfRows, Memory::ReadOnly, context);


    const size_t mapEntries = numOfRows 
        * TPC_PADS_PER_ROW_PADDED * TPC_MAX_TIME_PADDED; 

    if (cfg.halfs)
    {
        chargeMap = makeBuffer<cl_half>(mapEntries, Memory::ReadWrite, context);
    } 
    else 
    {
        chargeMap = makeBuffer<cl_float>(mapEntries, Memory::ReadWrite, context);
    }

    peakMap = makeBuffer<cl_uchar>(mapEntries, Memory::ReadWrite, context);

    aboveQTotCutoff = 
        makeBuffer<cl_uchar>(digitnum, Memory::ReadWrite, context);

    clusterNative = 
        makeBuffer<ClusterNative>(digitnum, Memory::ReadWrite, context);

    clusterNativeCutoff =
        makeBuffer<ClusterNative>(digitnum, Memory::ReadWrite, context);

    cluster = makeBuffer<Cluster>(digitnum, Memory::ReadWrite, context);


    cl::CommandQueue init(context, device);

    gpucpy<int>(globalToLocalRowMap, globalToLocalRow, numOfRows, init);
    gpucpy<int>(globalRowToCruMap, globalToLocalRow, numOfRows, init);

    if (cfg.halfs)
    {
        fill<cl_half>(chargeMap, 0.f, mapEntries, init);
    } else {
        fill<cl_float>(chargeMap, 0.f, mapEntries, init);
    }
    fill<cl_uchar>(peakMap, 0, mapEntries, init);

    init.finish();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
