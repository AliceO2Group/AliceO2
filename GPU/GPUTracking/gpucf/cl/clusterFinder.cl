#include "config.h"
#include "debug.h"

#include "shared/ClusterNative.h"
#include "shared/constants.h"
#include "shared/tpc.h"


#if defined(DEBUG_ON)
# define IF_DBG_INST if (get_global_linear_id() == 8)
# define IF_DBG_GROUP if (get_group_id(0) == 0)
#else
# define IF_DBG_INST if (false)
# define IF_DBG_GROUP if (false)
#endif

#define SCRATCH_PAD_WORK_GROUP_SIZE 64


#define GET_IS_PEAK(val) (val & 0x01)
#define GET_IS_ABOVE_THRESHOLD(val) (val >> 1)


typedef ushort packed_charge_t;

packed_charge_t packCharge(charge_t q, bool peak3x3, bool wasSplit)
{
    packed_charge_t p = q * 16.f;
    p = min((packed_charge_t)0x3FFF, p); // ensure only lower 14 bits are set
    p |= (wasSplit << 14);
    p |= (peak3x3 << 15);
    return p;
}

charge_t unpackCharge(packed_charge_t p)
{
    return (p & 0x3FFF) / 16.f;
}

bool has3x3Peak(packed_charge_t p)
{
    return p & (1 << 15);
}

bool wasSplit(packed_charge_t p)
{
    return p & (1 << 14);
}


typedef struct ClusterAccumulator_s
{
    charge_t Q;
    charge_t padMean;
    charge_t padSigma;
    charge_t timeMean;
    charge_t timeSigma;
    uchar    splitInTime;
    uchar    splitInPad;
} ClusterAccumulator;

typedef short delta_t;
typedef short2 delta2_t;

typedef struct ChargePos_s
{
    global_pad_t gpad;
    timestamp    time;
} ChargePos;

typedef short2 local_id;

constant delta2_t INNER_NEIGHBORS[8] =
{
    (delta2_t)(-1, -1), 

    (delta2_t)(-1, 0), 

    (delta2_t)(-1, 1),

    (delta2_t)(0, -1),

    (delta2_t)(0, 1),
    (delta2_t)(1, -1),
    (delta2_t)(1, 0), 
    (delta2_t)(1, 1),
};

constant bool INNER_TEST_EQ[8] =
{
    true,  true,  true,  true,
    false, false, false, false
}; 

constant delta2_t OUTER_NEIGHBORS[16] = 
{
    (delta2_t)(-2, -1),
    (delta2_t)(-2, -2),
    (delta2_t)(-1, -2),

    (delta2_t)(-2,  0),

    (delta2_t)(-2,  1),
    (delta2_t)(-2,  2),
    (delta2_t)(-1,  2),

    (delta2_t)( 0, -2),

    (delta2_t)( 0,  2),

    (delta2_t)( 2, -1),
    (delta2_t)( 2, -2),
    (delta2_t)( 1, -2),

    (delta2_t)( 2,  0),

    (delta2_t)( 2,  1),
    (delta2_t)( 2,  2),
    (delta2_t)( 1,  2)
};

constant uchar OUTER_TO_INNER[16] = 
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
constant uchar OUTER_TO_INNER_INV[16] =
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
constant delta2_t NOISE_SUPPRESSION_NEIGHBORS[NOISE_SUPPRESSION_NEIGHBOR_NUM] =
{
    (delta2_t)(-2, -3), 
    (delta2_t)(-2, -2), 
    (delta2_t)(-2, -1), 
    (delta2_t)(-2, 0), 
    (delta2_t)(-2, 1), 
    (delta2_t)(-2, 2), 

    (delta2_t)(-1, -3), 
    (delta2_t)(-1, -2), 
    (delta2_t)(-1, -1), 
    (delta2_t)(-1, 0), 
    (delta2_t)(-1, 1), 
    (delta2_t)(-1, 2), 
    (delta2_t)(-1, 3),

    (delta2_t)(0, -3), 
    (delta2_t)(0, -2), 
    (delta2_t)(0, -1),              
    (delta2_t)(0, 1), 
    (delta2_t)(0, 2), 
    (delta2_t)(0, 3),

    (delta2_t)(1, -3), 
    (delta2_t)(1, -2), 
    (delta2_t)(1, -1), 
    (delta2_t)(1, 0), 
    (delta2_t)(1, 1), 
    (delta2_t)(1, 2), 
    (delta2_t)(1, 3),
             
    (delta2_t)(2, -2), 
    (delta2_t)(2, -1), 
    (delta2_t)(2, 0), 
    (delta2_t)(2, 1), 
    (delta2_t)(2, 2), 
    (delta2_t)(2, 3),
    (delta2_t)(2, -3), 
    (delta2_t)(-2, 3)
};

constant uint NOISE_SUPPRESSION_MINIMA[NOISE_SUPPRESSION_NEIGHBOR_NUM] = 
{
    (1 << 7) | (1 << 8),
    (1 << 8),
    (1 << 8),
    (1 << 9),
    (1 << 10),
    (1 << 10),
    (1 << 7) | (1 << 8),
    (1 << 8),
    0,
    0,
    0,
    (1 << 10),
    (1 << 10) | (1 << 11),
    (1 << 14) | (1 << 15),
    (1 << 15),
    0,
    0,
    (1 << 16),
    (1 << 17) | (1 << 18),
    (1 << 20) | (1 << 21),
    (1 << 21),
    0,
    0,
    0,
    (1 << 23),
    (1 << 23) | (1 << 24),
    (1 << 21),
    (1 << 21),
    (1 << 22),
    (1 << 23),
    (1 << 23),
    (1 << 23) | (1 << 24),
    (1 << 10) | (1 << 11),
    (1 << 20) | (1 << 21)
};


bool isAtEdge(const Digit *d)
{
    return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW-2);
}

bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
{
    return aboveThreshold & (1 << OUTER_TO_INNER[outerIdx]);
}


bool innerAboveThresholdInv(uchar aboveThreshold, ushort outerIdx)
{
    return aboveThreshold & (1 << OUTER_TO_INNER_INV[outerIdx]);
}


void toNative(const ClusterAccumulator *cluster, const Digit *d, ClusterNative *cn)
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

void collectCharge(
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


charge_t updateClusterInner(
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

void updateClusterOuter(
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

void mergeCluster(
                    ushort              ll,
                    ushort              otherll,
                    ClusterAccumulator *myCluster,
              const ClusterAccumulator *otherCluster,
        local       charge_t           *clusterBcast)
{
    clusterBcast[otherll] = otherCluster->Q;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->Q           += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->padMean;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->padMean     += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->timeMean;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->timeMean    += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->padSigma;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->padSigma    += clusterBcast[ll];

    clusterBcast[otherll] = otherCluster->timeSigma;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->timeSigma   += clusterBcast[ll];

    local int *splitBcast = (local void *)clusterBcast;

    splitBcast[otherll] = otherCluster->splitInTime;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->splitInTime += splitBcast[ll];

    splitBcast[otherll] = otherCluster->splitInPad;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    myCluster->splitInPad  += splitBcast[ll];
}


void addOuterCharge(
        global const packed_charge_t    *chargeMap,
                     ClusterAccumulator *cluster, 
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    packed_charge_t p = CHARGE(chargeMap, gpad+dp, time+dt);
    updateClusterOuter(cluster, p, dp, dt);
}

charge_t addInnerCharge(
        global const packed_charge_t    *chargeMap,
                     ClusterAccumulator *cluster,
                     global_pad_t        gpad,
                     timestamp           time,
                     delta_t             dp,
                     delta_t             dt)
{
    packed_charge_t p = CHARGE(chargeMap, gpad+dp, time+dt);
    return updateClusterInner(cluster, p, dp, dt);
}

void addCorner(
        global const packed_charge_t    *chargeMap,
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

void addLine(
        global const packed_charge_t    *chargeMap,
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

void reset(ClusterAccumulator *clus)
{
    clus->Q           = 0.f;
    clus->padMean     = 0.f;
    clus->timeMean    = 0.f;
    clus->padSigma    = 0.f;
    clus->timeSigma   = 0.f;
    clus->splitInTime = 0;
    clus->splitInPad  = 0;
}

ushort partition(ushort ll, bool pred, ushort partSize, ushort *newPartSize)
{
    bool participates = ll < partSize;

    IF_DBG_INST DBGPR_1("ll = %d", ll);
    IF_DBG_INST DBGPR_1("partSize = %d", partSize);
    IF_DBG_INST DBGPR_1("pred = %d", pred);

    ushort lpos = work_group_scan_inclusive_add(!pred && participates);
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
    void fillScratchPad_##type( \
            global   const type      *chargeMap, \
                           uint       wgSize, \
                           uint       elems, \
                           ushort     ll, \
                           uint       offset, \
                           uint       N, \
            constant       delta2_t  *neighbors, \
            local    const ChargePos *posBcast, \
            local          type      *buf) \
{ \
    ushort x = ll % N; \
    ushort y = ll / N; \
    delta2_t d = neighbors[x + offset]; \
    delta_t dp = d.x; \
    delta_t dt = d.y; \
    LOOP_UNROLL_ATTR for (int i = y; i < wgSize; i += (elems / N)) \
    { \
        ChargePos readFrom = posBcast[i]; \
        uint writeTo = N * i + x; \
        buf[writeTo] = accessFunc(chargeMap, readFrom.gpad+dp, readFrom.time+dt); \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
} \
void anonymousFunction()

#define DECL_FILL_SCRATCH_PAD_COND(type, accessFunc, expandFunc, nameAppendix, null) \
    void fillScratchPad##nameAppendix##_##type( \
            global   const type      *chargeMap, \
                           uint       wgSize, \
                           uint       elems, \
                           ushort     ll, \
                           uint       offset, \
                           uint       N, \
            constant       delta2_t  *neighbors, \
            local    const ChargePos *posBcast, \
            local    const uchar     *aboveThreshold, \
            local          type      *buf) \
{ \
    ushort y = ll / N; \
    ushort x = ll % N; \
    delta2_t d = neighbors[x + offset]; \
    delta_t dp = d.x; \
    delta_t dt = d.y; \
    LOOP_UNROLL_ATTR for (int i = y; i < wgSize; i += (elems / N)) \
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
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
} \
void anonymousFunction()

DECL_FILL_SCRATCH_PAD(packed_charge_t, CHARGE);
DECL_FILL_SCRATCH_PAD(uchar, IS_PEAK);
DECL_FILL_SCRATCH_PAD_COND(packed_charge_t, CHARGE, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThreshold, Cond, 0);
DECL_FILL_SCRATCH_PAD_COND(packed_charge_t, CHARGE, innerAboveThresholdInv, CondInv, 0);
DECL_FILL_SCRATCH_PAD_COND(uchar, IS_PEAK, innerAboveThresholdInv, CondInv, 0);

void fillScratchPadNaive(
        global   const uchar     *chargeMap,
                       uint       wgSize,
                       ushort     ll,
                       uint       offset,
                       uint       N,
        constant       delta2_t  *neighbors,
        local    const ChargePos *posBcast,
        local          uchar     *buf)
{
    if (ll >= wgSize)
    {
        return;
    }

    ChargePos readFrom = posBcast[ll];

    for (int i = 0; i < N; i++)
    {
        delta2_t d = neighbors[i + offset];
        delta_t dp = d.x;
        delta_t dt = d.y;

        uint writeTo = N * ll + i;
        buf[writeTo] = IS_PEAK(chargeMap, readFrom.gpad+dp, readFrom.time+dt);
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
}


void updateClusterScratchpadInner(
                    ushort              lid,
                    ushort              N,
        local const packed_charge_t    *buf,
                    ClusterAccumulator *cluster,
        local       uchar              *innerAboveThreshold)
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

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
}



void updateClusterScratchpadOuter(
                    ushort              lid,
                    ushort              N,
                    ushort              M,
                    ushort              offset,
        local const packed_charge_t    *buf,
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


void buildClusterScratchPad(
        global const packed_charge_t    *chargeMap,
                     ChargePos           pos,
        local        ChargePos          *posBcast,
        local        packed_charge_t    *buf,
        local        uchar              *innerAboveThreshold,
                     ClusterAccumulator *myCluster)
{
    reset(myCluster);

    ushort ll = get_local_linear_id();

    posBcast[ll] = pos;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

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
    /*         (local void *)(posBcast)); */
}


void buildClusterNaive(
        global const packed_charge_t    *chargeMap,
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

bool isPeakScratchPad(
               const Digit           *digit,
                     ushort           N,
        global const packed_charge_t *chargeMap,
        local        ChargePos       *posBcast,
        local        packed_charge_t *buf)
{
    ushort ll = get_local_linear_id();

    const timestamp time = digit->time;
    const row_t row = digit->row;
    const pad_t pad = digit->pad;

    const global_pad_t gpad = tpcGlobalPadIdx(row, pad);
    ChargePos pos = {gpad, time};

    bool belowThreshold = (digit->charge <= QMAX_CUTOFF);

    ushort lookForPeaks;
    ushort partId = partition(
            ll, 
            belowThreshold, 
            SCRATCH_PAD_WORK_GROUP_SIZE, 
            &lookForPeaks);

    if (partId < lookForPeaks)
    {
        posBcast[partId] = pos;
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

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

bool isPeak(
               const Digit           *digit,
        global const packed_charge_t *chargeMap)
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


void finalize(
              ClusterAccumulator *pc,
        const Digit              *myDigit)
{
    pc->Q += myDigit->charge;

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

char countPeaksAroundDigit(
               const global_pad_t  gpad,
               const timestamp     time,
        global const uchar        *peakMap)
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

char countPeaksScratchpadInner(
                    ushort  ll,
        local const uchar  *isPeak,
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

char countPeaksScratchpadOuter(
                    ushort  ll,
                    ushort  offset,
                    uchar   aboveThreshold,
        local const uchar  *isPeak)
{
    char peaks = 0;
    for (uchar i = 0; i < 8; i++)
    {
        uchar p = isPeak[ll * 8 + i];
        /* bool extend = innerAboveThresholdInv(aboveThreshold, i + offset); */
        /* p = (extend) ? p : 0; */
        peaks += GET_IS_PEAK(p);
    }

    return peaks;
}



void sortIntoBuckets(
               const ClusterNative *cluster,
               const uint           bucket,
               const uint           maxElemsPerBucket,
        global       uint          *elemsInBucket,
        global       ClusterNative *buckets)
{
    uint posInBucket = atomic_add(&elemsInBucket[bucket], 1);

    buckets[maxElemsPerBucket * bucket + posInBucket] = *cluster;
}


void noiseSuppressionFindMinima(
        global const packed_charge_t *chargeMap,
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

        float r = unpackCharge(other);

        bool isMinima = (q - r > epsilon);
        *minimas |= (isMinima << i);

        bool lq = (r > q);
        *bigger |= (lq << i);
    }
}

ulong noiseSuppressionFindPeaks(
        global const uchar        *peakMap,
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

bool noiseSuppressionKeepPeak(
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


kernel
void fillChargeMap(
        global const Digit           *digits,
        global       packed_charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = packCharge(myDigit.charge, false, false);
}


kernel
void resetMaps(
        global const Digit           *digits,
        global       packed_charge_t *chargeMap,
        global       uchar           *isPeakMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = 0;
    IS_PEAK(isPeakMap, gpad, myDigit.time) = 0;
}


kernel
void findPeaks(
        global const packed_charge_t *chargeMap,
        global const Digit           *digits,
                     uint             digitnum,
        global       uchar           *isPeakPredicate,
        global       uchar           *peakMap)
{
    size_t idx = get_global_linear_id();

    // For certain configurations dummy work items are added, so the total 
    // number of work items is dividable by 64.
    // These dummy items also compute the last digit but discard the result.
    Digit myDigit = digits[min(idx, (size_t)(digitnum-1) )];

    uchar peak;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    IF_DBG_INST printf("Looking for peaks (using LDS)\n");
    const ushort N = 8;
    local ChargePos        posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    local packed_charge_t  buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];
    peak = isPeakScratchPad(&myDigit, N, chargeMap, posBcast, buf);
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

kernel
void noiseSuppression(
        global const packed_charge_t *chargeMap,
        global const uchar           *peakMap,
        global const Digit           *peaks,
               const uint             peaknum,
        global       uchar           *isPeakPredicate)
{
    size_t idx = get_global_linear_id();

    Digit myDigit = peaks[min(idx, (size_t)(peaknum-1) )];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    ulong minimas, bigger;
    noiseSuppressionFindMinima(
            chargeMap,
            gpad,
            myDigit.time,
            myDigit.charge,
            NOISE_SUPPRESSION_MINIMA_EPSILON,
            &minimas,
            &bigger);

    ulong peaksAround = noiseSuppressionFindPeaks(peakMap, gpad, myDigit.time);

    peaksAround &= bigger;

    bool keepMe = noiseSuppressionKeepPeak(minimas, peaksAround);

    bool iamDummy = (idx >= peaknum);
    if (iamDummy)
    {
        return;
    }

    isPeakPredicate[idx] = keepMe;
}

kernel
void updatePeaks(
        global const Digit *peaks,
        global const uchar *isPeak,
        global       uchar *peakMap)
{
    size_t idx = get_global_linear_id();

    Digit myDigit = peaks[idx];
    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    uchar peak = isPeak[idx];

    IS_PEAK(peakMap, gpad, myDigit.time) = 
        ((myDigit.charge > CHARGE_THRESHOLD) << 1) | peak;
}


kernel
void countPeaks(
        global const uchar           *peakMap,
        global       packed_charge_t *chargeMap,
        global const Digit           *digits,
               const uint             digitnum)
{
    size_t idx = get_global_linear_id();

    bool iamDummy = (idx >= digitnum);
    /* idx = select(idx, (size_t)(digitnum-1), (size_t)iamDummy); */
    idx = iamDummy ? digitnum-1 : idx;

    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    bool iamPeak  = GET_IS_PEAK(IS_PEAK(peakMap, gpad, myDigit.time));

    char peakCount = (iamPeak) ? 1 : 0;

#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    ushort ll = get_local_linear_id();
    ushort partId = ll;

    const ushort N = 8;

    local ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    local uchar     aboveThresholdBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    local uchar     buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];

    ushort in3x3 = 0;
    partId = partition(ll, iamPeak, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

    IF_DBG_INST DBGPR_2("partId = %d, in3x3 = %d", partId, in3x3);

    if (partId < in3x3)
    {
        posBcast1[partId] = (ChargePos){gpad, myDigit.time};
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    fillScratchPad_uchar(
            peakMap, 
            in3x3, 
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll, 
            0, 
            N, 
            INNER_NEIGHBORS, 
            posBcast1, 
            buf);

    uchar aboveThreshold = 0;
    if (partId < in3x3)
    {
        peakCount = countPeaksScratchpadInner(partId, buf, &aboveThreshold);
    }


    ushort in5x5 = 0;
    partId = partition(partId, peakCount > 0, in3x3, &in5x5);

    IF_DBG_INST DBGPR_2("partId = %d, in5x5 = %d", partId, in5x5);

    if (partId < in5x5)
    {
        posBcast1[partId] = (ChargePos){gpad, myDigit.time};
        aboveThresholdBcast[partId] = aboveThreshold;
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    fillScratchPadCondInv_uchar(
            peakMap, 
            in5x5, 
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll, 
            0, 
            N, 
            OUTER_NEIGHBORS, 
            posBcast1, 
            aboveThresholdBcast, 
            buf);

    if (partId < in5x5)
    {
        peakCount = countPeaksScratchpadOuter(partId, 0, aboveThreshold, buf);
    }

    fillScratchPadCondInv_uchar(
            peakMap, 
            in5x5, 
            SCRATCH_PAD_WORK_GROUP_SIZE,
            ll, 
            8, 
            N, 
            OUTER_NEIGHBORS, 
            posBcast1, 
            aboveThresholdBcast, 
            buf);
    if (partId < in5x5)
    {
        peakCount += countPeaksScratchpadOuter(partId, 8, aboveThreshold, buf);
        peakCount *= -1;
    }

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


kernel
void computeClusters(
        global const packed_charge_t *chargeMap,
        global const Digit           *digits,
                     uint             clusternum,
                     uint             maxClusterPerRow,
        global       uint            *clusterInRow,
        global       ClusterNative   *clusterByRow)
{
    uint idx = get_global_linear_id();

    // For certain configurations dummy work items are added, so the total 
    // number of work items is dividable by 64.
    // These dummy items also compute the last cluster but discard the result.
    Digit myDigit = digits[min(idx, clusternum-1)];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    ClusterAccumulator pc;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    const ushort N = 8;
    local ChargePos          posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    local packed_charge_t    buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];
    local uchar              innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];

    buildClusterScratchPad(
            chargeMap,
            (ChargePos){gpad, myDigit.time},
            posBcast,
            buf,
            innerAboveThreshold,
            &pc);
#else
    buildClusterNaive(chargeMap, &pc, gpad, myDigit.time);
#endif

    finalize(&pc, &myDigit);

    ClusterNative myCluster;
    toNative(&pc, &myDigit, &myCluster);

    bool iamDummy = (idx >= clusternum);
    if (iamDummy)
    {
        return;
    }

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
