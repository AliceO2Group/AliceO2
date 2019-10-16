// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

namespace gpucf
{
#include "clusterFinderDefs.h"
}

#if !defined(GPUCA_ALIGPUCODE)
namespace CAMath
{
template <class T> T Min(const T a, const T b) { return ::min(a, b); }
template <class T, class S> T AtomicAdd(__generic T* a, const S b) { return ::atomic_add(a, b); }
}
namespace GPUTPCClusterFinderKernels
{
struct GPUTPCSharedMemory
{
    union {
        gpucf::search_t search;
        gpucf::noise_t noise;
        gpucf::count_t count;
        gpucf::build_t build;
    };
};
}
#endif

namespace gpucf
{

GPUd() packed_charge_t packCharge(charge_t q, bool peak3x3, bool wasSplit)
{
    packed_charge_t p = q * 16.f;
    p = CAMath::Min((packed_charge_t)0x3FFF, p); // ensure only lower 14 bits are set
    p |= (wasSplit << 14);
    p |= (peak3x3 << 15);
    return p;
}

GPUd() charge_t unpackCharge(packed_charge_t p)
{
    return (p & 0x3FFF) / 16.f;
}

GPUd() bool has3x3Peak(packed_charge_t p)
{
    return p & (1 << 15);
}

GPUd() bool wasSplit(packed_charge_t p)
{
    return p & (1 << 14);
}

GPUconstexpr() delta2_t INNER_NEIGHBORS[8] =
{
    {-1, -1},

    {-1, 0},

    {-1, 1},

    {0, -1},

    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1}
};

GPUconstexpr() bool INNER_TEST_EQ[8] =
{
    true,  true,  true,  true,
    false, false, false, false
};

GPUconstexpr() delta2_t OUTER_NEIGHBORS[16] =
{
    {-2, -1},
    {-2, -2},
    {-1, -2},

    {-2,  0},

    {-2,  1},
    {-2,  2},
    {-1,  2},

    { 0, -2},

    { 0,  2},

    { 2, -1},
    { 2, -2},
    { 1, -2},

    { 2,  0},

    { 2,  1},
    { 2,  2},
    { 1,  2}
};

GPUconstexpr() uchar OUTER_TO_INNER[16] =
{
    0, 0, 0,

    1,

    2, 2, 2,

    3,

    4,

    5, 5, 5,

    6,

    7, 7, 7
};

// outer to inner mapping change for the peak counting step,
// as the other position is the position of the peak
GPUconstexpr() uchar OUTER_TO_INNER_INV[16] =
{
    1,
    0,
    3,
    1,
    1,
    2,
    4,
    3,
    4,
    6,
    5,
    3,
    6,
    6,
    7,
    4
};


#define NOISE_SUPPRESSION_NEIGHBOR_NUM 34
GPUconstexpr() delta2_t NOISE_SUPPRESSION_NEIGHBORS[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
{
    {-2, -3},
    {-2, -2},
    {-2, -1},
    {-2, 0},
    {-2, 1},
    {-2, 2},
    {-2, 3},

    {-1, -3},
    {-1, -2},
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {-1, 2},
    {-1, 3},

    {0, -3},
    {0, -2},
    {0, -1},

    {0, 1},
    {0, 2},
    {0, 3},

    {1, -3},
    {1, -2},
    {1, -1},
    {1, 0},
    {1, 1},
    {1, 2},
    {1, 3},
             
    {2, -3},
    {2, -2},
    {2, -1},
    {2, 0},
    {2, 1},
    {2, 2},
    {2, 3},
};

GPUconstexpr() uint NOISE_SUPPRESSION_MINIMA[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
{
    (1 << 8) | (1 << 9),
    (1 << 9),
    (1 << 9),
    (1 << 10),
    (1 << 11),
    (1 << 11),
    (1 << 11) | (1 << 12),
    (1 << 8) | (1 << 9),
    (1 << 9),
    0,
    0,
    0,
    (1 << 11),
    (1 << 11) | (1 << 12),
    (1 << 15) | (1 << 16),
    (1 << 16),
    0,
    0,
    (1 << 17),
    (1 << 18) | (1 << 19),
    (1 << 21) | (1 << 22),
    (1 << 22),
    0,
    0,
    0,
    (1 << 24),
    (1 << 24) | (1 << 25),
    (1 << 21) | (1 << 22),
    (1 << 22),
    (1 << 22),
    (1 << 23),
    (1 << 24),
    (1 << 24),
    (1 << 24) | (1 << 25)
};


GPUd() bool isAtEdge(const Digit *d)
{
    return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW-2);
}

GPUd() bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
{
    return aboveThreshold & (1 << OUTER_TO_INNER[outerIdx]);
}


GPUd() bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
{
    return aboveThreshold & (1 << OUTER_TO_INNER_INV[outerIdx]);
}


GPUd() void toNative(const ClusterAccumulator *cluster, const Digit *d, ClusterNative *cn)
{
    uchar isEdgeCluster = isAtEdge(d);
    uchar splitInTime   = cluster->splitInTime >= MIN_SPLIT_NUM;
    uchar splitInPad    = cluster->splitInPad  >= MIN_SPLIT_NUM;
    uchar flags =
          (isEdgeCluster << CN_FLAG_POS_IS_EDGE_CLUSTER)
        | (splitInTime   << CN_FLAG_POS_SPLIT_IN_TIME)
        | (splitInPad    << CN_FLAG_POS_SPLIT_IN_PAD);

    cn->qmax = d->charge;
    cn->qtot = cluster->Q;
    cnSetTimeFlags(cn, cluster->timeMean, flags);
    cnSetPad(cn, cluster->padMean);
    cnSetSigmaTime(cn, cluster->timeSigma);
    cnSetSigmaPad(cn, cluster->padSigma);
}

GPUd() void collectCharge(
        ClusterAccumulator *cluster,
        charge_t            splitCharge,
        delta_t             dp,
        delta_t             dt)
{
    cluster->Q         += splitCharge;
    cluster->padMean   += splitCharge*dp;
    cluster->timeMean  += splitCharge*dt;
    cluster->padSigma  += splitCharge*dp*dp;
    cluster->timeSigma += splitCharge*dt*dt;
}


GPUd() charge_t updateClusterInner(
        ClusterAccumulator *cluster,
        packed_charge_t     charge,
        delta_t             dp,
        delta_t             dt)
{
    charge_t q = unpackCharge(charge);

    collectCharge(cluster, q, dp, dt);

    bool split = wasSplit(charge);
    cluster->splitInTime += (dt != 0 && split);
    cluster->splitInPad  += (dp != 0 && split);

    return q;
}

GPUd() void updateClusterOuter(
        ClusterAccumulator *cluster,
        packed_charge_t     charge,
        delta_t             dp,
        delta_t             dt)
{
    charge_t q = unpackCharge(charge);

    bool split  = wasSplit(charge);
    bool has3x3 = has3x3Peak(charge);

    collectCharge(cluster, (has3x3) ? 0.f : q, dp, dt);

    cluster->splitInTime += (dt != 0 && split && !has3x3);
    cluster->splitInPad  += (dp != 0 && split && !has3x3);
}

GPUd() void mergeCluster(
                    ushort              ll,
                    ushort              otherll,
                    ClusterAccumulator *myCluster,
              const ClusterAccumulator *otherCluster,
        GPUsharedref()       charge_t           *clusterBcast)
{
    clusterBcast[otherll] = otherCluster->Q;
    GPUbarrier();
    myCluster->Q           += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->padMean;
    GPUbarrier();
    myCluster->padMean     += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->timeMean;
    GPUbarrier();
    myCluster->timeMean    += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->padSigma;
    GPUbarrier();
    myCluster->padSigma    += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->timeSigma;
    GPUbarrier();
    myCluster->timeSigma   += clusterBcast[ll];

    GPUsharedref() int *splitBcast = (GPUsharedref() int *)clusterBcast;

    splitBcast[otherll] = otherCluster->splitInTime;
    GPUbarrier();
    myCluster->splitInTime += splitBcast[ll];

    splitBcast[otherll] = otherCluster->splitInPad;
    GPUbarrier();
    myCluster->splitInPad  += splitBcast[ll];
}


GPUd() void addOuterCharge(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ClusterAccumulator *cluster,
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    packed_charge_t p = CHARGE(chargeMap, gpad+dp, time+dt);
    updateClusterOuter(cluster, p, dp, dt);
}

GPUd() charge_t addInnerCharge(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ClusterAccumulator *cluster,
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    packed_charge_t p = CHARGE(chargeMap, gpad+dp, time+dt);
    return updateClusterInner(cluster, p, dp, dt);
}

GPUd() void addCorner(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ClusterAccumulator *myCluster,
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    charge_t q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

    if (q > CHARGE_THRESHOLD)
    {
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp,   dt);
        addOuterCharge(chargeMap, myCluster, gpad, time,   dp, 2*dt);
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp, 2*dt);
    }
}

GPUd() void addLine(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ClusterAccumulator *myCluster,
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    charge_t q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

    if (q > CHARGE_THRESHOLD)
    {
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp, 2*dt);
    }
}

GPUd() void reset(ClusterAccumulator *clus)
{
    clus->Q           = 0.f;
    clus->padMean     = 0.f;
    clus->timeMean    = 0.f;
    clus->padSigma    = 0.f;
    clus->timeSigma   = 0.f;
    clus->splitInTime = 0;
    clus->splitInPad  = 0;
}

GPUd() ushort partition(GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem, ushort ll, bool pred, ushort partSize, ushort *newPartSize)
{
    bool participates = ll < partSize;

    IF_DBG_INST DBGPR_1("ll = %d", ll);
    IF_DBG_INST DBGPR_1("partSize = %d", partSize);
    IF_DBG_INST DBGPR_1("pred = %d", pred);

    ushort lpos = work_group_scan_inclusive_add((int) (!pred && participates));
    IF_DBG_INST DBGPR_1("lpos = %d", lpos);

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE-1);
    IF_DBG_INST DBGPR_1("part = %d", part);

    lpos -= 1;
    ushort pos = (participates && !pred) ? lpos : part;
    IF_DBG_INST DBGPR_1("pos = %d", pos);

    *newPartSize = part;
    return pos;
}


#define DECL_FILL_SCRATCH_PAD(type, accessFunc) \
    GPUd() void fillScratchPad_##type( \
            GPUglobalref()   const type      *chargeMap, \
                           uint       wgSize, \
                           uint       elems, \
                           ushort     ll, \
                           uint       offset, \
                           uint       N, \
            GPUconstexprref()      const delta2_t  *neighbors, \
            GPUsharedref()    const ChargePos *posBcast, \
            GPUsharedref()          type      *buf) \
{ \
    ushort x = ll % N; \
    ushort y = ll / N; \
    delta2_t d = neighbors[x + offset]; \
    delta_t dp = d.x; \
    delta_t dt = d.y; \
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N)) \
    { \
        ChargePos readFrom = posBcast[i]; \
        uint writeTo = N * i + x; \
        buf[writeTo] = accessFunc(chargeMap, readFrom.gpad+dp, readFrom.time+dt); \
    } \
    GPUbarrier(); \
} \
void anonymousFunction()

#define DECL_FILL_SCRATCH_PAD_COND(type, accessFunc, expandFunc, nameAppendix, null) \
    GPUd() void fillScratchPad##nameAppendix##_##type( \
            GPUglobalref()   const type      *chargeMap, \
                           uint       wgSize, \
                           uint       elems, \
                           ushort     ll, \
                           uint       offset, \
                           uint       N, \
            GPUconstexprref()      const delta2_t  *neighbors, \
            GPUsharedref()    const ChargePos *posBcast, \
            GPUsharedref()    const uchar     *aboveThreshold, \
            GPUsharedref()          type      *buf) \
{ \
    ushort y = ll / N; \
    ushort x = ll % N; \
    delta2_t d = neighbors[x + offset]; \
    delta_t dp = d.x; \
    delta_t dt = d.y; \
    LOOP_UNROLL_ATTR for (unsigned int i = y; i < wgSize; i += (elems / N)) \
    { \
        ChargePos readFrom = posBcast[i]; \
        uchar above = aboveThreshold[i]; \
        uint writeTo = N * i + x; \
        type v = null; \
        if (expandFunc(above, x + offset)) \
        { \
            v = accessFunc(chargeMap, readFrom.gpad+dp, readFrom.time+dt); \
        } \
        buf[writeTo] = v; \
    } \
    GPUbarrier(); \
} \
void anonymousFunction()

DECL_FILL_SCRATCH_PAD(packed_charge_t, CHARGE);
DECL_FILL_SCRATCH_PAD(uchar, IS_PEAK);
DECL_FILL_SCRATCH_PAD_COND(packed_charge_t, CHARGE, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(packed_charge_t, CHARGE, innerAboveThresholdInv, CondInv, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThresholdInv, CondInv, 0);

GPUd() void fillScratchPadNaive(
        GPUglobalref()   const uchar     *chargeMap,
                       uint       wgSize,
                       ushort     ll,
                       uint       offset,
                       uint       N,
        GPUconstexprref()      const delta2_t  *neighbors,
        GPUsharedref()    const ChargePos *posBcast,
        GPUsharedref()          uchar     *buf)
{
    if (ll >= wgSize)
    {
        return;
    }

    ChargePos readFrom = posBcast[ll];

    for (unsigned int i = 0; i < N; i++)
    {
        delta2_t d = neighbors[i + offset];
        delta_t dp = d.x;
        delta_t dt = d.y;

        uint writeTo = N * ll + i;
        buf[writeTo] = IS_PEAK(chargeMap, readFrom.gpad+dp, readFrom.time+dt);
    }

    GPUbarrier();
}


GPUd() void updateClusterScratchpadInner(
                    ushort              lid,
                    ushort              N,
        GPUsharedref() const packed_charge_t    *buf,
                    ClusterAccumulator *cluster,
        GPUsharedref()       uchar              *innerAboveThreshold)
{
    uchar aboveThreshold = 0;

    LOOP_UNROLL_ATTR for (ushort i = 0; i < N; i++)
    {
        delta2_t d = INNER_NEIGHBORS[i];

        delta_t dp = d.x;
        delta_t dt = d.y;

        packed_charge_t p = buf[N * lid + i];

        charge_t q = updateClusterInner(cluster, p, dp, dt);

        aboveThreshold |= ((q > CHARGE_THRESHOLD) << i);
    }

    IF_DBG_INST DBGPR_1("bitset = 0x%02x", aboveThreshold);

    innerAboveThreshold[lid] = aboveThreshold;

    GPUbarrier();
}



GPUd() void updateClusterScratchpadOuter(
                    ushort              lid,
                    ushort              N,
                    ushort              M,
                    ushort              offset,
        GPUsharedref() const packed_charge_t    *buf,
                    ClusterAccumulator *cluster)
{
    IF_DBG_INST DBGPR_1("bitset = 0x%02x", aboveThreshold);

    LOOP_UNROLL_ATTR for (ushort i = offset; i < M+offset; i++)
    {
        packed_charge_t p = buf[N * lid + i];

        delta2_t d = OUTER_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;

        updateClusterOuter(cluster, p, dp, dt);
    }
}


GPUd() void buildClusterScratchPad(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ChargePos           pos,
        GPUsharedref()        ChargePos          *posBcast,
        GPUsharedref()        packed_charge_t    *buf,
        GPUsharedref()        uchar              *innerAboveThreshold,
                     ClusterAccumulator *myCluster)
{
    reset(myCluster);

    ushort ll = get_local_id(0);

    posBcast[ll] = pos;
    GPUbarrier();

    /* IF_DBG_INST DBGPR_2("lid = (%d, %d)", lid.x, lid.y); */
    fillScratchPad_packed_charge_t(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            8,
            INNER_NEIGHBORS,
            posBcast,
            buf);
    updateClusterScratchpadInner(
            ll,
            8,
            buf,
            myCluster,
            innerAboveThreshold);

    ushort wgSizeHalf = SCRATCH_PAD_WORK_GROUP_SIZE / 2;

    bool inGroup1 = ll < wgSizeHalf;

    ushort llhalf = (inGroup1) ? ll : (ll-wgSizeHalf);

    /* ClusterAccumulator otherCluster; */
    /* reset(&otherCluster); */

    fillScratchPadCond_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            OUTER_NEIGHBORS,
            posBcast,
            innerAboveThreshold,
            buf);
    if (inGroup1)
    {
        updateClusterScratchpadOuter(
                llhalf,
                16,
                16,
                0,
                buf,
                myCluster);
    }

    fillScratchPadCond_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            OUTER_NEIGHBORS,
            posBcast+wgSizeHalf,
            innerAboveThreshold+wgSizeHalf,
            buf);
    if (!inGroup1)
    {
        updateClusterScratchpadOuter(
                llhalf,
                16,
                16,
                0,
                buf,
                myCluster);
    }

    /* mergeCluster( */
    /*         ll, */
    /*         (inGroup1) ? ll+wgSizeHalf : llhalf, */
    /*         myCluster, */
    /*         &otherCluster, */
    /*         (GPUsharedref() void *)(posBcast)); */
}


GPUd() void buildClusterNaive(
        GPUglobalref() const packed_charge_t    *chargeMap,
                     ClusterAccumulator *myCluster,
                     global_pad_t        gpad,
                     timestamp           time)
{
    reset(myCluster);


    // Add charges in top left corner:
    // O O o o o
    // O I i i o
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, gpad, time, -1, -1);

    // Add upper charges
    // o o O o o
    // o i I i o
    // o i c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, gpad, time,  0, -1);

    // Add charges in top right corner:
    // o o o O O
    // o i i I O
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, gpad, time, 1, -1);


    // Add left charges
    // o o o o o
    // o i i i o
    // O I c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, gpad, time, -1,  0);

    // Add right charges
    // o o o o o
    // o i i i o
    // o i c I O
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, gpad, time,  1,  0);


    // Add charges in bottom left corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // O I i i o
    // O O o o o
    addCorner(chargeMap, myCluster, gpad, time, -1, 1);

    // Add bottom charges
    // o o o o o
    // o i i i o
    // o i c i o
    // o i I i o
    // o o O o o
    addLine(chargeMap, myCluster, gpad, time,  0,  1);

    // Add charges in bottom right corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // o i i I O
    // o o o O O
    addCorner(chargeMap, myCluster, gpad, time, 1, 1);
}

GPUd() bool isPeakScratchPad(
               GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
               const Digit           *digit,
                     ushort           N,
        GPUglobalref() const packed_charge_t *chargeMap,
        GPUsharedref()        ChargePos       *posBcast,
        GPUsharedref()        packed_charge_t *buf)
{
    ushort ll = get_local_id(0);

    const timestamp time = digit->time;
    const row_t row = digit->row;
    const pad_t pad = digit->pad;

    const global_pad_t gpad = tpcGlobalPadIdx(row, pad);
    ChargePos pos = {gpad, time};

    bool belowThreshold = (digit->charge <= QMAX_CUTOFF);

    ushort lookForPeaks;
    ushort partId = partition(
            smem,
            ll,
            belowThreshold,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            &lookForPeaks);

    if (partId < lookForPeaks)
    {
        posBcast[partId] = pos;
    }
    GPUbarrier();

    fillScratchPad_packed_charge_t(
            chargeMap,
            lookForPeaks,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            N,
            INNER_NEIGHBORS,
            posBcast,
            buf);

    if (belowThreshold)
    {
        return false;
    }

    bool peak = true;
    for (ushort i = 0; i < N; i++)
    {
        charge_t q = unpackCharge(buf[N * partId + i]);
        peak &= (digit->charge > q)
            || (INNER_TEST_EQ[i] && digit->charge == q);
    }

    return peak;
}

GPUd() bool isPeak(
               const Digit           *digit,
        GPUglobalref() const packed_charge_t *chargeMap)
{
    if (digit->charge <= QMAX_CUTOFF)
    {
        return false;
    }

    const charge_t myCharge = digit->charge;
    const timestamp time = digit->time;
    const row_t row = digit->row;
    const pad_t pad = digit->pad;

    const global_pad_t gpad = tpcGlobalPadIdx(row, pad);

    bool peak = true;

#define CMP_NEIGHBOR(dp, dt, cmpOp) \
    do \
    { \
        const packed_charge_t p = CHARGE(chargeMap, gpad+dp, time+dt); \
        const charge_t otherCharge = unpackCharge(p); \
        peak &= (otherCharge cmpOp myCharge); \
    } \
    while (false)

#define CMP_LT CMP_NEIGHBOR(-1, -1, <=)
#define CMP_T  CMP_NEIGHBOR(-1, 0, <=)
#define CMP_RT CMP_NEIGHBOR(-1, 1, <=)

#define CMP_L  CMP_NEIGHBOR(0, -1, <=)
#define CMP_R  CMP_NEIGHBOR(0, 1, < )

#define CMP_LB CMP_NEIGHBOR(1, -1, < )
#define CMP_B  CMP_NEIGHBOR(1, 0, < )
#define CMP_RB CMP_NEIGHBOR(1, 1, < )

#if defined(CHARGEMAP_TILING_LAYOUT)
    CMP_LT;
    CMP_T;
    CMP_RT;
    CMP_R;
    CMP_RB;
    CMP_B;
    CMP_LB;
    CMP_L;
#else
    CMP_LT;
    CMP_T;
    CMP_RT;
    CMP_L;
    CMP_R;
    CMP_LB;
    CMP_B;
    CMP_RB;
#endif

#undef CMP_LT
#undef CMP_T
#undef CMP_RT
#undef CMP_L
#undef CMP_R
#undef CMP_LB
#undef CMP_B
#undef CMP_RB
#undef CMP_NEIGHBOR

    return peak;
}


GPUd() void finalize(
              ClusterAccumulator *pc,
        const Digit              *myDigit)
{
    pc->Q += myDigit->charge;
    if (pc->Q == 0) {
      return; // TODO: Why does this happen?
    }

    pc->padMean   /= pc->Q;
    pc->timeMean  /= pc->Q;
    pc->padSigma  /= pc->Q;
    pc->timeSigma /= pc->Q;

    pc->padSigma  = sqrt(pc->padSigma  - pc->padMean*pc->padMean);
    pc->timeSigma = sqrt(pc->timeSigma - pc->timeMean*pc->timeMean);

    pc->padMean  += myDigit->pad;
    pc->timeMean += myDigit->time;

#if defined(CORRECT_EDGE_CLUSTERS)
    if (isAtEdge(myDigit))
    {
        float s = (myDigit->pad < 2) ? 1.f : -1.f;
        bool c  = s*(pc->padMean - myDigit->pad) > 0.f;
        pc->padMean = (c) ? myDigit->pad : pc->padMean;
    }
#endif
}

GPUd() char countPeaksAroundDigit(
               const global_pad_t  gpad,
               const timestamp     time,
        GPUglobalref() const uchar        *peakMap)
{
    char peakCount = 0;

    uchar aboveThreshold = 0;
    for (uchar i = 0; i < 8; i++)
    {
        delta2_t d = INNER_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;

        uchar p = IS_PEAK(peakMap, gpad+dp, time+dt);
        peakCount += GET_IS_PEAK(p);
        aboveThreshold |= GET_IS_ABOVE_THRESHOLD(p) << i;
    }

    if (peakCount > 0)
    {
        return peakCount;
    }

    for (uchar i = 0; i < 16; i++)
    {
        delta2_t d = OUTER_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;

        if (innerAboveThresholdInv(aboveThreshold, i))
        {
            peakCount -= GET_IS_PEAK(IS_PEAK(peakMap, gpad+dp, time+dt));
        }
    }

    return peakCount;
}

GPUd() char countPeaksScratchpadInner(
                    ushort  ll,
        GPUsharedref() const uchar  *isPeak,
                    uchar  *aboveThreshold)
{
    char peaks = 0;
    for (uchar i = 0; i < 8; i++)
    {
        uchar p = isPeak[ll * 8 + i];
        peaks += GET_IS_PEAK(p);
        *aboveThreshold |= GET_IS_ABOVE_THRESHOLD(p) << i;
    }

    return peaks;
}

GPUd() char countPeaksScratchpadOuter(
                    ushort  ll,
                    ushort  offset,
                    uchar   aboveThreshold,
        GPUsharedref() const uchar  *isPeak)
{
    char peaks = 0;
    for (uchar i = 0; i < 16; i++)
    {
        uchar p = isPeak[ll * 16 + i];
        /* bool extend = innerAboveThresholdInv(aboveThreshold, i + offset); */
        /* p = (extend) ? p : 0; */
        peaks += GET_IS_PEAK(p);
    }

    return peaks;
}



GPUd() void sortIntoBuckets(
               const ClusterNative *cluster,
               const uint           bucket,
               const uint           maxElemsPerBucket,
        GPUglobalref()       uint          *elemsInBucket,
        GPUglobalref()       ClusterNative *buckets)
{
    uint posInBucket = CAMath::AtomicAdd(&elemsInBucket[bucket], 1);

    buckets[maxElemsPerBucket * bucket + posInBucket] = *cluster; // TODO: Must check for overflow over maxElemsPerBucket!
}


GPUd() void checkForMinima(
        float            q,
        float            epsilon,
        packed_charge_t  other,
        int              pos,
        ulong           *minimas,
        ulong           *bigger)
{
    float r = unpackCharge(other);

    bool isMinima = (q - r > epsilon);
    *minimas |= (isMinima << pos);

    bool lq = (r > q);
    *bigger |= (lq << pos);
}


GPUd() void noiseSuppressionFindMinimaScratchPad(
        GPUsharedref() const packed_charge_t *buf,
              const ushort           ll,
              const int              N,
                    int              pos,
              const float            q,
              const float            epsilon,
                    ulong           *minimas,
                    ulong           *bigger)
{
    for (int i = 0; i < N; i++, pos++)
    {
        packed_charge_t other = buf[N * ll + i];
        
        checkForMinima(q, epsilon, other, pos, minimas, bigger);
    }
}

GPUd() void noiseSuppressionFindPeaksScratchPad(
        GPUsharedref() const uchar  *buf,
              const ushort  ll,
              const int     N,
                    int     pos,
                    ulong  *peaks)
{
    for (int i = 0; i < N; i++, pos++)
    {
        uchar p = GET_IS_PEAK(buf[N * ll + i]);

        *peaks |= (p << pos);
    }
}


GPUd() void noiseSuppressionFindMinima(
        GPUglobalref() const packed_charge_t *chargeMap,
               const global_pad_t     gpad,
               const timestamp        time,
               const float            q,
               const float            epsilon,
                     ulong           *minimas,
                     ulong           *bigger)
{
    *minimas = 0;
    *bigger  = 0;

    for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++)
    {
        delta2_t d = NOISE_SUPPRESSION_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;

        packed_charge_t other = CHARGE(chargeMap, gpad+dp, time+dt);

        checkForMinima(q, epsilon, other, i, minimas, bigger);
    }
}

GPUd() ulong noiseSuppressionFindPeaks(
        GPUglobalref() const uchar        *peakMap,
               const global_pad_t  gpad,
               const timestamp     time)
{
    ulong peaks = 0;
    for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++)
    {
        delta2_t d = NOISE_SUPPRESSION_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;

        uchar p = IS_PEAK(peakMap, gpad+dp, time+dt);

        peaks |= (GET_IS_PEAK(p) << i);
    }

    return peaks;
}

GPUd() bool noiseSuppressionKeepPeak(
        ulong minima,
        ulong peaks)
{
    bool keepMe = true;

    for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++)
    {
        bool otherPeak = (peaks & (1 << i));
        bool minimaBetween = (minima & NOISE_SUPPRESSION_MINIMA[i]);

        keepMe &= (!otherPeak || minimaBetween);
    }

    return keepMe;
}

GPUd() void noiseSuppressionFindMinmasAndPeaksScratchpad(
        GPUglobalref() const packed_charge_t *chargeMap,
        GPUglobalref() const uchar           *peakMap,
                     float            q,
                     global_pad_t     gpad,
                     timestamp        time,
        GPUsharedref()        ChargePos       *posBcast,
        GPUsharedref()        packed_charge_t *buf,
                     ulong           *minimas,
                     ulong           *bigger,
                     ulong           *peaks)
{
    ushort ll = get_local_id(0);

    posBcast[ll] = (ChargePos){gpad, time};
    GPUbarrier();

    ushort wgSizeHalf = SCRATCH_PAD_WORK_GROUP_SIZE / 2;
    bool inGroup1 = ll < wgSizeHalf;
    ushort llhalf = (inGroup1) ? ll : (ll-wgSizeHalf);

    *minimas = 0;
    *bigger  = 0;
    *peaks   = 0;


    /**************************************
     * Look for minima
     **************************************/

    fillScratchPad_packed_charge_t(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            2,
            NOISE_SUPPRESSION_NEIGHBORS+16,
            posBcast,
            buf);

    noiseSuppressionFindMinimaScratchPad(
            buf,
            ll,
            2,
            16,
            q,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            minimas,
            bigger);


    fillScratchPad_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast,
            buf);

    if (inGroup1)
    {
        noiseSuppressionFindMinimaScratchPad(
            buf,
            llhalf,
            16,
            0,
            q,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            minimas,
            bigger);
    }

    fillScratchPad_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            18,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast,
            buf);

    if (inGroup1)
    {
        noiseSuppressionFindMinimaScratchPad(
            buf,
            llhalf,
            16,
            18,
            q,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            minimas,
            bigger);
    }

    fillScratchPad_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast+wgSizeHalf,
            buf);

    if (!inGroup1)
    {
        noiseSuppressionFindMinimaScratchPad(
            buf,
            llhalf,
            16,
            0,
            q,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            minimas,
            bigger);
    }

    fillScratchPad_packed_charge_t(
            chargeMap,
            wgSizeHalf,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            18,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast+wgSizeHalf,
            buf);

    if (!inGroup1)
    {
        noiseSuppressionFindMinimaScratchPad(
            buf,
            llhalf,
            16,
            18,
            q,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            minimas,
            bigger);
    }


    GPUsharedref() uchar *bufp = (GPUsharedref() uchar *) buf;

    /**************************************
     * Look for peaks
     **************************************/

    fillScratchPad_uchar(
            peakMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast,
            bufp);

    noiseSuppressionFindPeaksScratchPad(
            bufp,
            ll,
            16,
            0,
            peaks);

    fillScratchPad_uchar(
            peakMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            18,
            16,
            NOISE_SUPPRESSION_NEIGHBORS,
            posBcast,
            bufp);

    noiseSuppressionFindPeaksScratchPad(
            bufp,
            ll,
            16,
            18,
            peaks);
}



GPUd()
void fillChargeMap(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit           *digits,
        GPUglobalref()       packed_charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = packCharge(myDigit.charge, false, false);
}


GPUd()
void resetMaps(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit           *digits,
        GPUglobalref()       packed_charge_t *chargeMap,
        GPUglobalref()       uchar           *isPeakMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = 0;
    IS_PEAK(isPeakMap, gpad, myDigit.time) = 0;
}


GPUd()
void findPeaks(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const packed_charge_t *chargeMap,
        GPUglobalref() const Digit           *digits,
                     uint             digitnum,
        GPUglobalref()       uchar           *isPeakPredicate,
        GPUglobalref()       uchar           *peakMap)
{
    size_t idx = get_global_id(0);

    // For certain configurations dummy work items are added, so the total
    // number of work items is dividable by 64.
    // These dummy items also compute the last digit but discard the result.
    Digit myDigit = digits[CAMath::Min(idx, (size_t)(digitnum-1) )];

    uchar peak;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    IF_DBG_INST printf("Looking for peaks (using LDS)\n");
    peak = isPeakScratchPad(smem, &myDigit, SCRATCH_PAD_SEARCH_N, chargeMap, smem.search.posBcast, smem.search.buf);
#else
    peak = isPeak(&myDigit, chargeMap);
#endif

    // Exit early if dummy. See comment above.
    bool iamDummy = (idx >= digitnum);
    if (iamDummy)
    {
        return;
    }

    isPeakPredicate[idx] = peak;

    const global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    IS_PEAK(peakMap, gpad, myDigit.time) =
        ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}

GPUd()
void noiseSuppression(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const packed_charge_t *chargeMap,
        GPUglobalref() const uchar           *peakMap,
        GPUglobalref() const Digit           *peaks,
               const uint             peaknum,
        GPUglobalref()       uchar           *isPeakPredicate)
{
    size_t idx = get_global_id(0);

    Digit myDigit = peaks[CAMath::Min(idx, (size_t)(peaknum-1) )];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    ulong minimas, bigger, peaksAround;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)

    noiseSuppressionFindMinmasAndPeaksScratchpad(
            chargeMap,
            peakMap,
            myDigit.charge,
            gpad,
            myDigit.time,
            smem.noise.posBcast,
            smem.noise.buf,
            &minimas,
            &bigger,
            &peaksAround);
#else
    noiseSuppressionFindMinima(
            chargeMap,
            gpad,
            myDigit.time,
            myDigit.charge,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            &minimas,
            &bigger);

    peaksAround = noiseSuppressionFindPeaks(peakMap, gpad, myDigit.time);
#endif

    peaksAround &= bigger;

    bool keepMe = noiseSuppressionKeepPeak(minimas, peaksAround);

    bool iamDummy = (idx >= peaknum);
    if (iamDummy)
    {
        return;
    }

    isPeakPredicate[idx] = keepMe;
}

GPUd()
void updatePeaks(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit *peaks,
        GPUglobalref() const uchar *isPeak,
        GPUglobalref()       uchar *peakMap)
{
    size_t idx = get_global_id(0);

    Digit myDigit = peaks[idx];
    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    uchar peak = isPeak[idx];

    IS_PEAK(peakMap, gpad, myDigit.time) =
        ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}


GPUd()
void countPeaks(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const uchar           *peakMap,
        GPUglobalref()       packed_charge_t *chargeMap,
        GPUglobalref() const Digit           *digits,
               const uint             digitnum)
{
    size_t idx = get_global_id(0);

    bool iamDummy = (idx >= digitnum);
    /* idx = select(idx, (size_t)(digitnum-1), (size_t)iamDummy); */
    idx = iamDummy ? digitnum-1 : idx;

    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    bool iamPeak  = GET_IS_PEAK(IS_PEAK(peakMap, gpad, myDigit.time));

    char peakCount = (iamPeak) ? 1 : 0;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    ushort ll = get_local_id(0);
    ushort partId = ll;

    ushort in3x3 = 0;
    partId = partition(smem, ll, iamPeak, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

    IF_DBG_INST DBGPR_2("partId = %d, in3x3 = %d", partId, in3x3);

    if (partId < in3x3)
    {
        smem.count.posBcast1[partId] = (ChargePos){gpad, myDigit.time};
    }
    GPUbarrier();

    fillScratchPad_uchar(
            peakMap,
            in3x3,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            8,
            INNER_NEIGHBORS,
            smem.count.posBcast1,
            smem.count.buf);

    uchar aboveThreshold = 0;
    if (partId < in3x3)
    {
        peakCount = countPeaksScratchpadInner(partId, smem.count.buf, &aboveThreshold);
    }


    ushort in5x5 = 0;
    partId = partition(smem, partId, peakCount > 0, in3x3, &in5x5);

    IF_DBG_INST DBGPR_2("partId = %d, in5x5 = %d", partId, in5x5);

    if (partId < in5x5)
    {
        smem.count.posBcast1[partId] = (ChargePos){gpad, myDigit.time};
        smem.count.aboveThresholdBcast[partId] = aboveThreshold;
    }
    GPUbarrier();

    fillScratchPadCondInv_uchar(
            peakMap,
            in5x5,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll,
            0,
            16,
            OUTER_NEIGHBORS,
            smem.count.posBcast1,
            smem.count.aboveThresholdBcast,
            smem.count.buf);

    if (partId < in5x5)
    {
        peakCount = countPeaksScratchpadOuter(partId, 0, aboveThreshold, smem.count.buf);
        peakCount *= -1;
    }

    /* fillScratchPadCondInv_uchar( */
    /*         peakMap, */
    /*         in5x5, */
    /*         SCRATCH_PAD_WORK_GROUP_SIZE, */
    /*         ll, */
    /*         8, */
    /*         N, */
    /*         OUTER_NEIGHBORS, */
    /*         smem.count.posBcast1, */
    /*         smem.count.aboveThresholdBcast, */
    /*         smem.count.buf); */
    /* if (partId < in5x5) */
    /* { */
    /*     peakCount += countPeaksScratchpadOuter(partId, 8, aboveThreshold, smem.count.buf); */
    /*     peakCount *= -1; */
    /* } */

#else
    peakCount = countPeaksAroundDigit(gpad, myDigit.time, peakMap);
    peakCount = iamPeak ? 1 : peakCount;
#endif

    if (iamDummy)
    {
        return;
    }

    bool has3x3 = (peakCount > 0);
    peakCount = abs(peakCount);
    bool split  = (peakCount > 1);

    peakCount = (peakCount == 0) ? 1 : peakCount;

    packed_charge_t p = packCharge(myDigit.charge / peakCount, has3x3, split);

    CHARGE(chargeMap, gpad, myDigit.time) = p;
}


GPUd()
void computeClusters(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const packed_charge_t *chargeMap,
        GPUglobalref() const Digit           *digits,
                     uint             clusternum,
                     uint             maxClusterPerRow,
        GPUglobalref()       uint            *clusterInRow,
        GPUglobalref()       ClusterNative   *clusterByRow)
{
    uint idx = get_global_id(0);

    // For certain configurations dummy work items are added, so the total
    // number of work items is dividable by 64.
    // These dummy items also compute the last cluster but discard the result.
    Digit myDigit = digits[CAMath::Min(idx, clusternum-1)];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    ClusterAccumulator pc;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    buildClusterScratchPad(
            chargeMap,
            (ChargePos){gpad, myDigit.time},
            smem.build.posBcast,
            smem.build.buf,
            smem.build.innerAboveThreshold,
            &pc);
#else
    buildClusterNaive(chargeMap, &pc, gpad, myDigit.time);
#endif

    if (idx >= clusternum) {
      return;
    }
    finalize(&pc, &myDigit);

    ClusterNative myCluster;
    toNative(&pc, &myDigit, &myCluster);

#if defined(CUT_QTOT)
    bool aboveQTotCutoff = (pc.Q > QTOT_CUTOFF);
#else
    bool aboveQTotCutoff = true;
#endif

    if (aboveQTotCutoff)
    {
        sortIntoBuckets(
                &myCluster,
                myDigit.row,
                maxClusterPerRow,
                clusterInRow,
                clusterByRow);
    }
}

} // namespace gpucf

#if !defined(GPUCA_ALIGPUCODE)

GPUg()
void fillChargeMap_kernel(
        GPUglobal() const gpucf::Digit           *digits,
        GPUglobal()       gpucf::packed_charge_t *chargeMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::fillChargeMap(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, digits, chargeMap);
}

GPUg()
void resetMaps_kernel(
        GPUglobal() const gpucf::Digit           *digits,
        GPUglobal()       gpucf::packed_charge_t *chargeMap,
        GPUglobal()       gpucf::uchar           *isPeakMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::resetMaps(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, digits, chargeMap, isPeakMap);
}

GPUg()
void findPeaks_kernel(
        GPUglobal() const gpucf::packed_charge_t *chargeMap,
        GPUglobal() const gpucf::Digit           *digits,
                     uint             digitnum,
        GPUglobal()       gpucf::uchar           *isPeakPredicate,
        GPUglobal()       gpucf::uchar           *peakMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::findPeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, digits, digitnum, isPeakPredicate, peakMap);
}

GPUg()
void noiseSuppression_kernel(
        GPUglobal() const gpucf::packed_charge_t *chargeMap,
        GPUglobal() const gpucf::uchar           *peakMap,
        GPUglobal() const gpucf::Digit           *peaks,
               const uint             peaknum,
        GPUglobal()       gpucf::uchar           *isPeakPredicate)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::noiseSuppression(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, peakMap, peaks, peaknum, isPeakPredicate);
}

GPUg()
void updatePeaks_kernel(
        GPUglobal() const gpucf::Digit *peaks,
        GPUglobal() const gpucf::uchar *isPeak,
        GPUglobal()       gpucf::uchar *peakMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::updatePeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, peaks, isPeak, peakMap);
}

GPUg()
void countPeaks_kernel(
        GPUglobal() const gpucf::uchar           *peakMap,
        GPUglobal()       gpucf::packed_charge_t *chargeMap,
        GPUglobal() const gpucf::Digit           *digits,
               const uint             digitnum)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::countPeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, peakMap, chargeMap, digits, digitnum);
}

GPUg()
void computeClusters_kernel(
        GPUglobal() const gpucf::packed_charge_t *chargeMap,
        GPUglobal() const gpucf::Digit           *digits,
                     uint             clusternum,
                     uint             maxClusterPerRow,
        GPUglobal()       uint            *clusterInRow,
        GPUglobal()       gpucf::ClusterNative   *clusterByRow)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::computeClusters(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, chargeMap, digits, clusternum, maxClusterPerRow, clusterInRow, clusterByRow);
}
#endif
