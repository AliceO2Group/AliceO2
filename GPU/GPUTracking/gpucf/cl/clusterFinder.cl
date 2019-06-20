#include "config.h"
#include "debug.h"

#include "shared/ClusterNative.h"
#include "shared/tpc.h"


#if 0
# define IF_DBG_INST if (get_global_linear_id() == 0)
# define IF_DBG_GROUP if (get_group_id(0) == 0)
#else
# define IF_DBG_INST if (false)
# define IF_DBG_GROUP if (false)
#endif

#define SCRATCH_PAD_WORK_GROUP_SIZE 64


typedef struct PartialCluster_s
{
    charge_t Q;
    charge_t padMean;
    charge_t padSigma;
    charge_t timeMean;
    charge_t timeSigma;
} PartialCluster;

typedef short delta_t;
typedef short2 delta2_t;

typedef struct ChargePos_s
{
    global_pad_t gpad;
    timestamp    time;
} ChargePos;

typedef short2 local_id;

constant charge_t CHARGE_THRESHOLD = 2;
constant charge_t OUTER_CHARGE_THRESHOLD = 0;

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


void toNative(const PartialCluster *cluster, charge_t qmax, uchar flags, ClusterNative *cn)
{
    cn->qmax = qmax;
    cn->qtot = cluster->Q;
    cnSetTimeFlags(cn, cluster->timeMean, flags);
    cnSetPad(cn, cluster->padMean);
    cnSetSigmaTime(cn, cluster->timeSigma);
    cnSetSigmaPad(cn, cluster->padSigma);
}

void toRegular(
               const PartialCluster *cluster, 
               const Digit          *myDigit,
        global const int            *globalToLocalRow,
        global const int            *globalRowToCru,
                     Cluster        *out)
{
    out->Q         = cluster->Q;
    out->QMax      = round(myDigit->charge);
    out->padMean   = cluster->padMean;
    out->timeMean  = cluster->timeMean;
    out->timeSigma = cluster->timeSigma;
    out->padSigma  = cluster->padSigma;

    out->cru = globalRowToCru[myDigit->row];
    out->row = globalToLocalRow[myDigit->row];

}

void updateCluster(PartialCluster *cluster, charge_t charge, delta_t dp, delta_t dt)
{
    cluster->Q         += charge;
    cluster->padMean   += charge*dp;
    cluster->timeMean  += charge*dt;
    cluster->padSigma  += charge*dp*dp;
    cluster->timeSigma += charge*dt*dt;
}

bool isAtEdge(const Digit *d)
{
    return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW-2);
}

void addOuterCharge(
        global const charge_t       *chargeMap,
                     PartialCluster *cluster, 
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t outerCharge = CHARGE(chargeMap, gpad+dp, time+dt);

#if defined(SPLIT_CHARGES)
    outerCharge = (outerCharge < 0) ? -outerCharge : 0.f;
#else
    outerCharge = (outerCharge > OUTER_CHARGE_THRESHOLD) ? outerCharge : 0;
#endif

    updateCluster(cluster, outerCharge, dp, dt);
}

charge_t addInnerCharge(
        global const charge_t       *chargeMap,
                     PartialCluster *cluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q  = CHARGE(chargeMap, gpad+dp, time+dt);

    updateCluster(cluster, q, dp, dt);

    return q;
}

void addCorner(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);
    
#if !defined(SPLIT_CHARGES)
    if (q > CHARGE_THRESHOLD)
    {
#endif
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp,   dt);
        addOuterCharge(chargeMap, myCluster, gpad, time,   dp, 2*dt);
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp, 2*dt);
#if !defined(SPLIT_CHARGES)
    }
#endif
}

void addLine(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q = addInnerCharge(chargeMap, myCluster, gpad, time, dp, dt);

#if !defined(SPLIT_CHARGES)
    if (q > CHARGE_THRESHOLD)
    {
#endif
        addOuterCharge(chargeMap, myCluster, gpad, time, 2*dp, 2*dt);
#if !defined(SPLIT_CHARGES)
    }
#endif
}

void reset(PartialCluster *clus)
{
    clus->Q = 0.f;
    clus->padMean = 0.f;
    clus->timeMean = 0.f;
    clus->padSigma = 0.f;
    clus->timeSigma = 0.f;
}


void fillScratchPad(
        global   const charge_t  *chargeMap,
                       uint       wgSize,
                       local_id   lid,
                       uint       offset,
                       uint       N,
        constant       delta2_t  *neighbors,
        local          ChargePos *posBcast,
        local          charge_t  *buf)
{
    int i = 0;
	__attribute__((opencl_unroll_hint(1)))
    for (; i < wgSize-N; i += N)
    {
        ChargePos readFrom = posBcast[i + lid.y];

        delta2_t d = neighbors[lid.x + offset];
        delta_t dp = d.x;
        delta_t dt = d.y;

        /* IF_DBG_INST DBGPR_2("delta = (%d, %d)", dp, dt); */
        
        uint writeTo = N * (i + lid.y) + lid.x;

        /* IF_DBG_INST DBGPR_1("writeTo = %d", writeTo); */

        buf[writeTo] = CHARGE(chargeMap, readFrom.gpad+dp, readFrom.time+dt);
    }

    // load last batch of values seperately as wgSize might not be divisible by N
    if (i + lid.y <= wgSize)
    {
        ChargePos readFrom = posBcast[i + lid.y];

        delta2_t d = neighbors[lid.x + offset];
        delta_t dp = d.x;
        delta_t dt = d.y;

        /* IF_DBG_INST DBGPR_2("delta = (%d, %d)", dp, dt); */
        
        uint writeTo = N * (i + lid.y) + lid.x;

        /* IF_DBG_INST DBGPR_1("writeTo = %d", writeTo); */

        buf[writeTo] = CHARGE(chargeMap, readFrom.gpad+dp, readFrom.time+dt);
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
}

void updateClusterScratchpadInner(
                    ushort          lid,
                    ushort          N,
        local const charge_t       *buf,
                    PartialCluster *cluster,
        local       uchar          *innerAboveThreshold)
{
    uchar aboveThreshold = 0;

	__attribute__((opencl_unroll_hint(1)))
    for (ushort i = 0; i < N; i++)
    {
        delta2_t d = INNER_NEIGHBORS[i];

        delta_t dp = d.x;
        delta_t dt = d.y;

        charge_t q = buf[N * lid + i];

        IF_DBG_INST DBGPR_3("q = %f, dp = %d, dt = %d", q, dp, dt);

        updateCluster(cluster, q, dp, dt);

        aboveThreshold |= ((q > CHARGE_THRESHOLD) << i);
    }

    IF_DBG_INST DBGPR_1("bitset = 0x%02x", aboveThreshold);

    innerAboveThreshold[lid] = aboveThreshold;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
}


bool innerAboveThreshold(uchar aboveThreshold, ushort outerIdx)
{
    return aboveThreshold & (1 << OUTER_TO_INNER[outerIdx]);
}

void updateClusterScratchpadOuter(
                    ushort          lid,
                    ushort          N,
                    ushort          offset,
        local const charge_t       *buf,
        local const uchar          *innerAboveThresholdSet,
                    PartialCluster *cluster)
{
    uchar aboveThreshold = innerAboveThresholdSet[lid];

    IF_DBG_INST DBGPR_1("bitset = 0x%02x", aboveThreshold);
	
	__attribute__((opencl_unroll_hint(1)))
    for (ushort i = 0; i < N; i++)
    {
        charge_t q = buf[N * lid + i];
        ushort outerIdx = i + offset;

        bool contributes = (q > OUTER_CHARGE_THRESHOLD 
                && innerAboveThreshold(aboveThreshold, outerIdx));

        IF_DBG_INST DBGPR_1("q = %f", q);

        q = (contributes) ? q : 0.f;

        delta2_t d = OUTER_NEIGHBORS[outerIdx];

        delta_t dp = d.x;
        delta_t dt = d.y;
        IF_DBG_INST DBGPR_3("q = %f, dp = %d, dt = %d", q, dp, dt);
        updateCluster(cluster, q, dp, dt);
    }
}


void buildClusterScratchPad(
            global const charge_t       *chargeMap,
                         ChargePos       pos,
                         ushort          N,
            local        ChargePos      *posBcast,
            local        charge_t       *buf,
            local        uchar          *innerAboveThreshold,
                         PartialCluster *myCluster)
{
    reset(myCluster);

    ushort ll = get_local_linear_id();
    local_id lid = {ll % N, ll / N};

    posBcast[ll] = pos;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    /* IF_DBG_INST DBGPR_2("lid = (%d, %d)", lid.x, lid.y); */
    fillScratchPad(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            0,
            N,
            INNER_NEIGHBORS,
            posBcast,
            buf);

    updateClusterScratchpadInner(ll, N, buf, myCluster, innerAboveThreshold);

    fillScratchPad(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE, 
            lid, 
            0, 
            N, 
            OUTER_NEIGHBORS,
            posBcast, 
            buf);
    updateClusterScratchpadOuter(ll, N, 0, buf, innerAboveThreshold, myCluster);

    fillScratchPad(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            8,
            N,
            OUTER_NEIGHBORS,
            posBcast,
            buf);
    updateClusterScratchpadOuter(ll, N, 8, buf, innerAboveThreshold, myCluster);
}


void buildClusterNaive(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     global_pad_t    gpad,
                     timestamp       time)
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
               const Digit     *digit,
                     ushort     N,
        global const charge_t  *chargeMap,
        local        ChargePos *posBcast,
        local        charge_t  *buf)
{
    ushort ll = get_local_linear_id();
    local_id lid = {ll % N, ll / N};

    const timestamp time = digit->time;
    const row_t row = digit->row;
    const pad_t pad = digit->pad;

    const global_pad_t gpad = tpcGlobalPadIdx(row, pad);
    ChargePos pos = {gpad, time};

    posBcast[ll] = pos;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    fillScratchPad(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            0,
            N,
            INNER_NEIGHBORS,
            posBcast,
            buf);

    bool peak = true;
    for (ushort i = 0; i < N; i++)
    {
        charge_t q = buf[N * ll + i];
        peak &= (digit->charge > q) 
             || (INNER_TEST_EQ[i] && digit->charge == q);
    }
    peak &= (digit->charge > CHARGE_THRESHOLD);

    return peak;
}

bool isPeak(
               const Digit    *digit,
        global const charge_t *chargeMap)
{
    const charge_t myCharge = digit->charge;
    const timestamp time = digit->time;
    const row_t row = digit->row;
    const pad_t pad = digit->pad;

    const global_pad_t gpad = tpcGlobalPadIdx(row, pad);

    bool peak = true;

#define CMP_NEIGHBOR(dp, dt, cmpOp) \
    do \
    { \
        const charge_t otherCharge = CHARGE(chargeMap, gpad+dp, time+dt); \
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

    peak &= (myCharge > CHARGE_THRESHOLD);

    return peak;
}


void finalize(
              PartialCluster *pc,
        const Digit          *myDigit)
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
    if (isAtEdge(myDigit) 
            && fabs(pc->padMean - (charge_t)myDigit->pad) > 0.0f
            && fabs(pc->timeMean - (charge_t)myDigit->time) > 0.0f)
    {
        pc->padMean = myDigit->pad; 
        pc->timeMean = myDigit->time;
    }
#endif
}

char countPeaksAroundDigit(
               const global_pad_t  gpad,
               const timestamp     time,
        global const uchar        *peakMap)
{
    char peakCount = 0;

    for (uchar i = 0; i < 8; i++)
    {
        delta2_t d = INNER_NEIGHBORS[i];
        delta_t dp = d.x;
        delta_t dt = d.y;
        peakCount += IS_PEAK(peakMap, gpad+dp, time+dt);
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
        peakCount -= IS_PEAK(peakMap, gpad+dp, time+dt);
    }

    return peakCount;
}


ushort partition(
                    ushort     ll, 
                    bool       pred, 
                    ushort     partSize,
        local const charge_t  *chargeBcastIn,
        local const ChargePos *posBcastIn,
        local       charge_t  *chargeBcastOut,
        local       ChargePos *posBcastOut)
{
    bool participates = ll < partSize;

    ushort lpos = work_group_scan_inclusive_add( pred && participates);
    ushort rpos = work_group_scan_inclusive_add(!pred && participates);

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE-1);

    lpos -= 1;
    rpos += part-1;
    ushort pos = (participates) ? ((pred) ? rpos : lpos) : ll;

    chargeBcastOut[pos] = chargeBcastIn[ll];
    posBcastOut[pos]    = posBcastIn[ll];

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    return part;
}


kernel
void fillChargeMap(
        global const Digit    *digits,
        global       charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = myDigit.charge;
}


kernel
void resetMaps(
        global const Digit    *digits,
        global       charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = 0.f;
}


kernel
void findPeaks(
        global const charge_t *chargeMap,
        global const Digit    *digits,
                     uint      digitnum,
        global       uchar    *isPeakPredicate,
        global       uchar    *peakMap)
{
    size_t idx = get_global_linear_id();

    // For certain configurations dummy work items are added, so the total 
    // number of work items is dividable by 64.
    // These dummy items also compute the last digit but discard the result.
    Digit myDigit = digits[min(idx, (size_t)(digitnum-1) )];

    bool peak;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    const ushort N = 8;
    local ChargePos posBcast[N];
    local charge_t  buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];
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

#if defined(SPLIT_CHARGES)
    const global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    IS_PEAK(peakMap, gpad, myDigit.time) = peak;
#endif
}


kernel
void countPeaks(
        global const uchar    *peakMap,
        global const Digit    *digits,
               const uint      digitnum,
        global const uchar    *isPeak,
        global       charge_t *chargeMap)
{
    size_t idx = get_global_linear_id();

    bool iamDummy = (idx >= digitnum);
    /* idx = select(idx, (size_t)(digitnum-1), (size_t)iamDummy); */
    idx = iamDummy ? digitnum-1 : idx;

    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    bool iamPeak  = isPeak[idx];

    char peakCount;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    ushort ll = get_local_linear_id();

    local ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    local ChargePos posBcast2[SCRATCH_PAD_WORK_GROUP_SIZE];
    local charge_t  chargeBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    local charge_t  chargeBcast2[SCRATCH_PAD_WORK_GROUP_SIZE];
    local uchar     buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];

    posBcast1[ll]    = {gpad, myDigit.time};
    chargeBcast1[ll] = myDigit.charge;

    // TODO find number of dummies

    ushort in3x3 = partition(
            ll, 
            iamPeak,
            SCRATCH_PAD_WORK_GROUP_SIZE - numOfDummies, 
            chargeBcast1, 
            posBcast1, 
            chargeBcast2, 
            posBcast2);

    fillScratchPad(isPeak, in3x3, lid, 0, N, INNER_NEIGHBORS, posBcast2, buf);
    if (ll < in3x3)
    {
        peakCount = countPeaksScratchPadInner();
        chargeBcast2[ll] /= peakCount;
    }

    ushort in5x5 = partition(
            ll,
            peakCount > 0,
            in3x3,
            chargeBcast2,
            posBcast2,
            chargeBcast1,
            posBcast1);

    fillScratchPad(isPeak, in5x5, lid, 0, N, OUTER_NEIGHBORS, posBcast1, buf);
    if (ll < in5x5)
    {
        peakCount = countPeaksScratchPad();
    }

    fillScratchPad(isPeak, in5x5, lid, 8, N, OUTER_NEIGHBORS, posBcast1, buf);
    if (ll < in5x5)
    {
        peakCount += countPeaksScratchPad();
        chargeBcast1[ll] /= peakCount;
    }

    ChargePos pos = posBcast1[ll];
    CHARGE(chargeMap, pos.pad, pos.time) = chargeBcast[ll];

#else
    peakCount = countPeaksAroundDigit(gpad, myDigit.time, peakMap);

    if (iamDummy)
    {
        return;
    }

    /* peakCount = select(peakCount, (uchar) (PCMask_Has3x3Peak | 1), (uchar)iamPeak); */
    peakCount = iamPeak ? 1 : peakCount;

    CHARGE(chargeMap, gpad, myDigit.time) = myDigit.charge / peakCount;
#endif
}


kernel
void computeClusters(
        global const charge_t      *chargeMap,
        global const Digit         *digits,
        global const int           *globalToLocalRow,
        global const int           *globalRowToCru,
                     uint           clusternum,
#if defined(CLUSTER_NATIVE_OUTPUT)
        global       ClusterNative *clusters,
#else
        global       Cluster       *clusters,
#endif
        global       uchar         *peakMap)
{
    uint idx = get_global_linear_id();

    // For certain configurations dummy work items are added, so the total 
    // number of work items is dividable by 64.
    // These dummy items also compute the last cluster but discard the result.
    Digit myDigit = digits[min(idx, clusternum-1)];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    PartialCluster pc;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    const ushort N = 8;
    local ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    local charge_t  buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];
    local uchar     innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];

    buildClusterScratchPad(
            chargeMap,
            (ChargePos){gpad, myDigit.time},
            N,
            posBcast,
            buf,
            innerAboveThreshold,
            &pc);
#else
    buildClusterNaive(chargeMap, &pc, gpad, myDigit.time);
#endif

    finalize(&pc, &myDigit);

#if defined(CLUSTER_NATIVE_OUTPUT)
    bool isEdgeCluster = isAtEdge(&myDigit);

    uchar flags = isEdgeCluster;
    ClusterNative myCluster;
    toNative(&pc, myDigit.charge, flags, &myCluster);
#else
    Cluster myCluster;
    toRegular(
            &pc, 
            &myDigit,
            globalToLocalRow, 
            globalRowToCru, 
            &myCluster);
#endif

    clusters[idx] = myCluster;

#if defined(SPLIT_CHARGES)
    IS_PEAK(peakMap, gpad, myDigit.time) = 0;
#endif
}


kernel
void nativeToRegular(
        global const ClusterNative *native,
        global const Digit         *peaks,
        global const int           *globalToLocalRow,
        global const int           *globalRowToCru,
        global       Cluster       *regular)
{
    size_t idx = get_global_linear_id();

    ClusterNative cn = native[idx];
    Digit peak       = peaks[idx];

    Cluster c;
    c.Q    = cn.qtot;
    c.QMax = cn.qmax;

    c.padMean = cnGetPad(&cn);
    c.timeMean = cnGetTime(&cn);

    c.padSigma = cnGetSigmaPad(&cn);
    c.timeSigma = cnGetSigmaTime(&cn);

    c.cru = globalRowToCru[peak.row];
    c.row = globalToLocalRow[peak.row];

    regular[idx] = c;
}

