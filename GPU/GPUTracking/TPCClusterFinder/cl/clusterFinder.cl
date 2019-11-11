// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "clusterFinderDefs.h"


#if !defined(GPUCA_ALIGPUCODE)
namespace CAMath
{
template <class T> T Min(const T a, const T b) { return ::min(a, b); }
template <class T, class S> T AtomicAdd(__generic T* a, const S b) { return ::atomic_add(a, b); }
}
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

using namespace deprecated;

GPUconstexpr() Delta2 INNER_NEIGHBORS[8] =
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

GPUconstexpr() Delta2 OUTER_NEIGHBORS[16] =
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
GPUconstexpr() Delta2 NOISE_SUPPRESSION_NEIGHBORS[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
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
        Charge            splitCharge,
        Delta             dp,
        Delta             dt)
{
    cluster->Q         += splitCharge;
    cluster->padMean   += splitCharge*dp;
    cluster->timeMean  += splitCharge*dt;
    cluster->padSigma  += splitCharge*dp*dp;
    cluster->timeSigma += splitCharge*dt*dt;
}


GPUd() Charge updateClusterInner(
        ClusterAccumulator *cluster,
        PackedCharge     charge,
        Delta             dp,
        Delta             dt)
{
    Charge q = unpackCharge(charge);

    collectCharge(cluster, q, dp, dt);

    bool split = wasSplit(charge);
    cluster->splitInTime += (dt != 0 && split);
    cluster->splitInPad  += (dp != 0 && split);

    return q;
}

GPUd() void updateClusterOuter(
        ClusterAccumulator *cluster,
        PackedCharge     charge,
        Delta             dp,
        Delta             dt)
{
    Charge q = unpackCharge(charge);

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
        GPUsharedref()       Charge           *clusterBcast)
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
        GPUglobalref() const PackedCharge    *chargeMap,
                     ClusterAccumulator *cluster,
                     GlobalPad        gpad,
                     Timestamp           time,
                     Delta             dp,
                     Delta             dt)
{
    PackedCharge p = CHARGE(chargeMap, gpad+dp, time+dt);
    updateClusterOuter(cluster, p, dp, dt);
}

GPUd() Charge addInnerCharge(
        GPUglobalref() const PackedCharge    *chargeMap,
                     ClusterAccumulator *cluster,
                     GlobalPad        gpad,
                     Timestamp           time,
                     Delta             dp,
                     Delta             dt)
{
    PackedCharge p = CHARGE(chargeMap, gpad+dp, time+dt);
    return updateClusterInner(cluster, p, dp, dt);
}

GPUd() void addCorner(
        GPUglobalref() const PackedCharge    *chargeMap,
                     ClusterAccumulator *myCluster,
                     GlobalPad        gpad,
                     Timestamp           time,
                     Delta             dp,
                     Delta             dt)
{
    Charge q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

    if (q > CHARGE_THRESHOLD)
    {
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp,   dt);
        addOuterCharge(chargeMap, myCluster, gpad, time,   dp, 2*dt);
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp, 2*dt);
    }
}

GPUd() void addLine(
        GPUglobalref() const PackedCharge    *chargeMap,
                     ClusterAccumulator *myCluster,
                     GlobalPad        gpad,
                     Timestamp           time,
                     Delta             dp,
                     Delta             dt)
{
    Charge q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

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

    ushort lpos = work_group_scan_inclusive_add((int) (!pred && participates));

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE-1);

    lpos -= 1;
    ushort pos = (participates && !pred) ? lpos : part;

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
            GPUconstexprref()      const Delta2  *neighbors, \
            GPUsharedref()    const ChargePos *posBcast, \
            GPUsharedref()          type      *buf) \
{ \
    ushort x = ll % N; \
    ushort y = ll / N; \
    Delta2 d = neighbors[x + offset]; \
    Delta dp = d.x; \
    Delta dt = d.y; \
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
            GPUconstexprref()      const Delta2  *neighbors, \
            GPUsharedref()    const ChargePos *posBcast, \
            GPUsharedref()    const uchar     *aboveThreshold, \
            GPUsharedref()          type      *buf) \
{ \
    ushort y = ll / N; \
    ushort x = ll % N; \
    Delta2 d = neighbors[x + offset]; \
    Delta dp = d.x; \
    Delta dt = d.y; \
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

DECL_FILL_SCRATCH_PAD(PackedCharge, CHARGE);
DECL_FILL_SCRATCH_PAD(uchar, IS_PEAK);
DECL_FILL_SCRATCH_PAD_COND(PackedCharge, CHARGE, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(PackedCharge, CHARGE, innerAboveThresholdInv, CondInv, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThresholdInv, CondInv, 0);

GPUd() void fillScratchPadNaive(
        GPUglobalref()   const uchar     *chargeMap,
                       uint       wgSize,
                       ushort     ll,
                       uint       offset,
                       uint       N,
        GPUconstexprref()      const Delta2  *neighbors,
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
        Delta2 d = neighbors[i + offset];
        Delta dp = d.x;
        Delta dt = d.y;

        uint writeTo = N * ll + i;
        buf[writeTo] = IS_PEAK(chargeMap, readFrom.gpad+dp, readFrom.time+dt);
    }

    GPUbarrier();
}


GPUd() void updateClusterScratchpadInner(
                    ushort              lid,
                    ushort              N,
        GPUsharedref() const PackedCharge    *buf,
                    ClusterAccumulator *cluster,
        GPUsharedref()       uchar              *innerAboveThreshold)
{
    uchar aboveThreshold = 0;

    LOOP_UNROLL_ATTR for (ushort i = 0; i < N; i++)
    {
        Delta2 d = INNER_NEIGHBORS[i];

        Delta dp = d.x;
        Delta dt = d.y;

        PackedCharge p = buf[N * lid + i];

        Charge q = updateClusterInner(cluster, p, dp, dt);

        aboveThreshold |= ((q > CHARGE_THRESHOLD) << i);
    }

    innerAboveThreshold[lid] = aboveThreshold;

    GPUbarrier();
}



GPUd() void updateClusterScratchpadOuter(
                    ushort              lid,
                    ushort              N,
                    ushort              M,
                    ushort              offset,
        GPUsharedref() const PackedCharge    *buf,
                    ClusterAccumulator *cluster)
{
    LOOP_UNROLL_ATTR for (ushort i = offset; i < M+offset; i++)
    {
        PackedCharge p = buf[N * lid + i];

        Delta2 d = OUTER_NEIGHBORS[i];
        Delta dp = d.x;
        Delta dt = d.y;

        updateClusterOuter(cluster, p, dp, dt);
    }
}


GPUd() void buildClusterScratchPad(
        GPUglobalref() const PackedCharge    *chargeMap,
                     ChargePos           pos,
        GPUsharedref()        ChargePos          *posBcast,
        GPUsharedref()        PackedCharge    *buf,
        GPUsharedref()        uchar              *innerAboveThreshold,
                     ClusterAccumulator *myCluster)
{
    reset(myCluster);

    ushort ll = get_local_id(0);

    posBcast[ll] = pos;
    GPUbarrier();

    fillScratchPad_PackedCharge(
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

    fillScratchPadCond_PackedCharge(
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

    fillScratchPadCond_PackedCharge(
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
        GPUglobalref() const PackedCharge    *chargeMap,
                     ClusterAccumulator *myCluster,
                     GlobalPad        gpad,
                     Timestamp           time)
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
        GPUglobalref() const PackedCharge *chargeMap,
        GPUsharedref()        ChargePos       *posBcast,
        GPUsharedref()        PackedCharge *buf)
{
    ushort ll = get_local_id(0);

    const Timestamp time = digit->time;
    const Row row = digit->row;
    const Pad pad = digit->pad;

    const GlobalPad gpad = tpcGlobalPadIdx(row, pad);
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

    fillScratchPad_PackedCharge(
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
        Charge q = unpackCharge(buf[N * partId + i]);
        peak &= (digit->charge > q)
            || (INNER_TEST_EQ[i] && digit->charge == q);
    }

    return peak;
}

GPUd() bool isPeak(
               const Digit           *digit,
        GPUglobalref() const PackedCharge *chargeMap)
{
    if (digit->charge <= QMAX_CUTOFF)
    {
        return false;
    }

    const Charge myCharge = digit->charge;
    const Timestamp time = digit->time;
    const Row row = digit->row;
    const Pad pad = digit->pad;

    const GlobalPad gpad = tpcGlobalPadIdx(row, pad);

    bool peak = true;

#define CMP_NEIGHBOR(dp, dt, cmpOp) \
    do \
    { \
        const PackedCharge p = CHARGE(chargeMap, gpad+dp, time+dt); \
        const Charge otherCharge = unpackCharge(p); \
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
               const GlobalPad  gpad,
               const Timestamp     time,
        GPUglobalref() const uchar        *peakMap)
{
    char peakCount = 0;

    uchar aboveThreshold = 0;
    for (uchar i = 0; i < 8; i++)
    {
        Delta2 d = INNER_NEIGHBORS[i];
        Delta dp = d.x;
        Delta dt = d.y;

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
        Delta2 d = OUTER_NEIGHBORS[i];
        Delta dp = d.x;
        Delta dt = d.y;

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
        PackedCharge  other,
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
        GPUsharedref() const PackedCharge *buf,
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
        PackedCharge other = buf[N * ll + i];
        
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
        GPUglobalref() const PackedCharge *chargeMap,
               const GlobalPad     gpad,
               const Timestamp        time,
               const float            q,
               const float            epsilon,
                     ulong           *minimas,
                     ulong           *bigger)
{
    *minimas = 0;
    *bigger  = 0;

    for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++)
    {
        Delta2 d = NOISE_SUPPRESSION_NEIGHBORS[i];
        Delta dp = d.x;
        Delta dt = d.y;

        PackedCharge other = CHARGE(chargeMap, gpad+dp, time+dt);

        checkForMinima(q, epsilon, other, i, minimas, bigger);
    }
}

GPUd() ulong noiseSuppressionFindPeaks(
        GPUglobalref() const uchar        *peakMap,
               const GlobalPad  gpad,
               const Timestamp     time)
{
    ulong peaks = 0;
    for (int i = 0; i < NOISE_SUPPRESSION_NEIGHBOR_NUM; i++)
    {
        Delta2 d = NOISE_SUPPRESSION_NEIGHBORS[i];
        Delta dp = d.x;
        Delta dt = d.y;

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
        GPUglobalref() const PackedCharge *chargeMap,
        GPUglobalref() const uchar           *peakMap,
                     float            q,
                     GlobalPad     gpad,
                     Timestamp        time,
        GPUsharedref()        ChargePos       *posBcast,
        GPUsharedref()        PackedCharge *buf,
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

    fillScratchPad_PackedCharge(
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


    fillScratchPad_PackedCharge(
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

    fillScratchPad_PackedCharge(
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

    fillScratchPad_PackedCharge(
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

    fillScratchPad_PackedCharge(
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
void fillChargeMapImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit           *digits,
        GPUglobalref()       PackedCharge *chargeMap,
        size_t maxDigit)
{
    size_t idx = get_global_id(0);
    if (idx >= maxDigit) {
      return;
    }
    Digit myDigit = digits[idx];
    
    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    
    CHARGE(chargeMap, gpad, myDigit.time) = packCharge(myDigit.charge, false, false);
}


GPUd()
void resetMapsImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit           *digits,
        GPUglobalref()       PackedCharge *chargeMap,
        GPUglobalref()       uchar           *isPeakMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = 0;
    IS_PEAK(isPeakMap, gpad, myDigit.time) = 0;
}


GPUd()
void findPeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const PackedCharge *chargeMap,
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

    const GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    IS_PEAK(peakMap, gpad, myDigit.time) =
        ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}

GPUd()
void noiseSuppressionImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const PackedCharge *chargeMap,
        GPUglobalref() const uchar           *peakMap,
        GPUglobalref() const Digit           *peaks,
               const uint             peaknum,
        GPUglobalref()       uchar           *isPeakPredicate)
{
    size_t idx = get_global_id(0);

    Digit myDigit = peaks[CAMath::Min(idx, (size_t)(peaknum-1) )];

    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

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
void updatePeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const Digit *peaks,
        GPUglobalref() const uchar *isPeak,
        GPUglobalref()       uchar *peakMap)
{
    size_t idx = get_global_id(0);

    Digit myDigit = peaks[idx];
    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    uchar peak = isPeak[idx];

    IS_PEAK(peakMap, gpad, myDigit.time) =
        ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}


GPUd()
void countPeaksImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const uchar           *peakMap,
        GPUglobalref()       PackedCharge *chargeMap,
        GPUglobalref() const Digit           *digits,
               const uint             digitnum)
{
    size_t idx = get_global_id(0);

    bool iamDummy = (idx >= digitnum);
    /* idx = select(idx, (size_t)(digitnum-1), (size_t)iamDummy); */
    idx = iamDummy ? digitnum-1 : idx;

    Digit myDigit = digits[idx];

    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    bool iamPeak  = GET_IS_PEAK(IS_PEAK(peakMap, gpad, myDigit.time));

    char peakCount = (iamPeak) ? 1 : 0;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    ushort ll = get_local_id(0);
    ushort partId = ll;

    ushort in3x3 = 0;
    partId = partition(smem, ll, iamPeak, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

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

    PackedCharge p = packCharge(myDigit.charge / peakCount, has3x3, split);

    CHARGE(chargeMap, gpad, myDigit.time) = p;
}


GPUd()
void computeClustersImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
        GPUglobalref() const PackedCharge *chargeMap,
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

    GlobalPad gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

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

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#if !defined(GPUCA_ALIGPUCODE)

GPUg()
void fillChargeMap_kernel(
        GPUglobal() const gpucf::Digit           *digits,
        GPUglobal()       gpucf::PackedCharge *chargeMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::fillChargeMap(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, digits, chargeMap, get_global_size(0));
}

GPUg()
void resetMaps_kernel(
        GPUglobal() const gpucf::Digit           *digits,
        GPUglobal()       gpucf::PackedCharge *chargeMap,
        GPUglobal()       gpucf::uchar           *isPeakMap)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::resetMaps(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, digits, chargeMap, isPeakMap);
}

GPUg()
void findPeaks_kernel(
        GPUglobal() const gpucf::PackedCharge *chargeMap,
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
        GPUglobal() const gpucf::PackedCharge *chargeMap,
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
        GPUglobal()       gpucf::PackedCharge *chargeMap,
        GPUglobal() const gpucf::Digit           *digits,
               const uint             digitnum)
{
    GPUshared() GPUTPCClusterFinderKernels::GPUTPCSharedMemory smem;
    gpucf::countPeaks(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), smem, peakMap, chargeMap, digits, digitnum);
}

GPUg()
void computeClusters_kernel(
        GPUglobal() const gpucf::PackedCharge *chargeMap,
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
