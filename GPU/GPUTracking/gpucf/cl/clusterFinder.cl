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


typedef struct PartialCluster_s
{
    charge_t Q;
    charge_t padMean;
    charge_t padSigma;
    charge_t timeMean;
    charge_t timeSigma;
    uchar    splitInTime;
    uchar    splitInPad;
} PartialCluster;

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


bool isAtEdge(const Digit *d)
{
    return (d->pad < 2 || d->pad >= TPC_PADS_PER_ROW-2);
}



void toNative(const PartialCluster *cluster, const Digit *d, ClusterNative *cn)
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
        PartialCluster *cluster,
        charge_t splitCharge,
        delta_t dp,
        delta_t dt)
{
    cluster->Q         += splitCharge;
    cluster->padMean   += splitCharge*dp;
    cluster->timeMean  += splitCharge*dt;
    cluster->padSigma  += splitCharge*dp*dp;
    cluster->timeSigma += splitCharge*dt*dt;
}


charge_t updateClusterInner(
        PartialCluster *cluster, 
        charge_t charge, 
        char peakCount, 
        delta_t dp, 
        delta_t dt)
{
    charge /= peakCount;

    collectCharge(cluster, charge, dp, dt);

    cluster->splitInTime += (dt != 0 && peakCount > 1);
    cluster->splitInPad  += (dp != 0 && peakCount > 1);

    return charge;
}

void updateClusterOuter(
        PartialCluster *cluster,
        charge_t charge,
        char peakCount,
        delta_t dp,
        delta_t dt)
{
    charge = (peakCount < 0) ? charge / -peakCount : 0.f;
    /* charge = (charge > OUTER_CHARGE_THRESHOLD) ? charge : 0; */

    collectCharge(cluster, charge, dp, dt);

    cluster->splitInTime += (dt != 0 && peakCount < -1);
    cluster->splitInPad  += (dp != 0 && peakCount < -1);
}


void addOuterCharge(
        global const charge_t       *chargeMap,
        global const char           *peakCountMap,
                     PartialCluster *cluster, 
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t outerCharge = CHARGE(chargeMap, gpad+dp, time+dt);
    char     peakCount   = PEAK_COUNT(peakCountMap, gpad+dp, time+dt);

    updateClusterOuter(cluster, outerCharge, peakCount, dp, dt);
}

charge_t addInnerCharge(
        global const charge_t       *chargeMap,
        global const char           *peakCountMap,
                     PartialCluster *cluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q  = CHARGE(chargeMap, gpad+dp, time+dt);
    char peakCount   = PEAK_COUNT(peakCountMap, gpad+dp, time+dt);

    return updateClusterInner(cluster, q, peakCount, dp, dt);
}

void addCorner(
        global const charge_t       *chargeMap,
        global const char           *peakCountMap,
                     PartialCluster *myCluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q = addInnerCharge(chargeMap, peakCountMap, myCluster, gpad, time, dp, dt);
    
    /* if (q > CHARGE_THRESHOLD) */
    /* { */
        addOuterCharge(chargeMap, peakCountMap, myCluster, gpad, time, 2*dp,   dt);
        addOuterCharge(chargeMap, peakCountMap, myCluster, gpad, time,   dp, 2*dt);
        addOuterCharge(chargeMap, peakCountMap, myCluster, gpad, time, 2*dp, 2*dt);
    /* } */
}

void addLine(
        global const charge_t       *chargeMap,
        global const char           *peakCountMap,
                     PartialCluster *myCluster,
                     global_pad_t    gpad,
                     timestamp       time,
                     delta_t         dp,
                     delta_t         dt)
{
    charge_t q = addInnerCharge(chargeMap, peakCountMap, myCluster, gpad, time, dp, dt);

    /* if (q > CHARGE_THRESHOLD) */
    /* { */
        addOuterCharge(chargeMap, peakCountMap, myCluster, gpad, time, 2*dp, 2*dt);
    /* } */
}

void reset(PartialCluster *clus)
{
    clus->Q           = 0.f;
    clus->padMean     = 0.f;
    clus->timeMean    = 0.f;
    clus->padSigma    = 0.f;
    clus->timeSigma   = 0.f;
    clus->splitInTime = 0;
    clus->splitInPad  = 0;
}


#define DECL_FILL_SCRATCH_PAD(type, accessFunc) \
    void fillScratchPad_##type( \
            global   const type      *chargeMap, \
                           uint       wgSize, \
                           local_id   lid, \
                           uint       offset, \
                           uint       N, \
            constant       delta2_t  *neighbors, \
            local    const ChargePos *posBcast, \
            local          type      *buf) \
    { \
        int i = lid.y; \
        __attribute__((opencl_unroll_hint(1))) \
        for (; i < wgSize; i += N) \
        { \
            ChargePos readFrom = posBcast[i]; \
            delta2_t d = neighbors[lid.x + offset]; \
            delta_t dp = d.x; \
            delta_t dt = d.y; \
            \
            uint writeTo = N * i + lid.x; \
            \
            buf[writeTo] = accessFunc(chargeMap, readFrom.gpad+dp, readFrom.time+dt); \
        } \
        work_group_barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    void anonymousFunction()

DECL_FILL_SCRATCH_PAD(charge_t, CHARGE);
DECL_FILL_SCRATCH_PAD(uchar, IS_PEAK);
DECL_FILL_SCRATCH_PAD(char, PEAK_COUNT);

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
                    ushort          lid,
                    ushort          N,
        local const charge_t       *buf,
        local const char           *peakCount,
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

        // FIXME pass peakCount
        char pc = peakCount[N * lid + i];
        updateClusterInner(cluster, q, pc, dp, dt);

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
        local const char           *peakCount,
        local const uchar          *innerAboveThresholdSet,
                    PartialCluster *cluster)
{
    uchar aboveThreshold = innerAboveThresholdSet[lid];

    IF_DBG_INST DBGPR_1("bitset = 0x%02x", aboveThreshold);
	
	__attribute__((opencl_unroll_hint(1)))
    for (ushort i = 0; i < N; i++)
    {
        charge_t q = buf[N * lid + i];
        char    pc = peakCount[N * lid + i];

        /* q = (q < 0) ? -q : 0.f; */
        /* bool contributes = (q > OUTER_CHARGE_THRESHOLD */ 
        /*         && innerAboveThreshold(aboveThreshold, outerIdx)); */

        /* q = (contributes) ? q : 0.f; */

        ushort outerIdx = i + offset;
        delta2_t d = OUTER_NEIGHBORS[outerIdx];
        delta_t dp = d.x;
        delta_t dt = d.y;

        IF_DBG_INST DBGPR_3("q = %f, dp = %d, dt = %d", q, dp, dt);
        updateClusterOuter(cluster, q, pc, dp, dt);
    }
}


void buildClusterScratchPad(
            global const charge_t       *chargeMap,
            global const char           *peakCountMap,
                         ChargePos       pos,
                         ushort          N,
            local        ChargePos      *posBcast,
            local        charge_t       *buf,
            local        char           *bufPeakCount,
            local        uchar          *innerAboveThreshold,
                         PartialCluster *myCluster)
{
    reset(myCluster);

    ushort ll = get_local_linear_id();
    local_id lid = {ll % N, ll / N};

    posBcast[ll] = pos;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    /* IF_DBG_INST DBGPR_2("lid = (%d, %d)", lid.x, lid.y); */
    fillScratchPad_charge_t(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            0,
            N,
            INNER_NEIGHBORS,
            posBcast,
            buf);
    fillScratchPad_char(
            peakCountMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            0,
            N,
            INNER_NEIGHBORS,
            posBcast,
            bufPeakCount);
    updateClusterScratchpadInner(
            ll, 
            N, 
            buf, 
            bufPeakCount, 
            myCluster, 
            innerAboveThreshold);

    fillScratchPad_charge_t(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE, 
            lid, 
            0, 
            N, 
            OUTER_NEIGHBORS,
            posBcast, 
            buf);
    fillScratchPad_char(
            peakCountMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            0,
            N,
            OUTER_NEIGHBORS,
            posBcast,
            bufPeakCount);
    updateClusterScratchpadOuter(
            ll, 
            N, 
            0, 
            buf, 
            bufPeakCount, 
            innerAboveThreshold, 
            myCluster);

    fillScratchPad_charge_t(
            chargeMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            8,
            N,
            OUTER_NEIGHBORS,
            posBcast,
            buf);
    fillScratchPad_char(
            peakCountMap,
            SCRATCH_PAD_WORK_GROUP_SIZE,
            lid,
            8,
            N,
            OUTER_NEIGHBORS,
            posBcast,
            bufPeakCount);
    updateClusterScratchpadOuter(
            ll, 
            N, 
            8, 
            buf, 
            bufPeakCount, 
            innerAboveThreshold, 
            myCluster);
}


void buildClusterNaive(
        global const charge_t       *chargeMap,
        global const char           *peakCountMap,
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
    addCorner(chargeMap, peakCountMap, myCluster, gpad, time, -1, -1);

    // Add upper charges
    // o o O o o
    // o i I i o
    // o i c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, peakCountMap, myCluster, gpad, time,  0, -1);

    // Add charges in top right corner:
    // o o o O O
    // o i i I O
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, peakCountMap, myCluster, gpad, time, 1, -1);


    // Add left charges
    // o o o o o
    // o i i i o
    // O I c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, peakCountMap, myCluster, gpad, time, -1,  0);

    // Add right charges
    // o o o o o
    // o i i i o
    // o i c I O
    // o i i i o
    // o o o o o
    addLine(chargeMap, peakCountMap, myCluster, gpad, time,  1,  0);


    // Add charges in bottom left corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // O I i i o
    // O O o o o
    addCorner(chargeMap, peakCountMap, myCluster, gpad, time, -1, 1);

    // Add bottom charges
    // o o o o o
    // o i i i o
    // o i c i o
    // o i I i o
    // o o O o o
    addLine(chargeMap, peakCountMap, myCluster, gpad, time,  0,  1); 
    // Add charges in bottom right corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // o i i I O
    // o o o O O
    addCorner(chargeMap, peakCountMap, myCluster, gpad, time, 1, 1);
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

    IF_DBG_INST DBGPR_2("pos = %d, %d", pos.gpad, pos.time);

    posBcast[ll] = pos;
    IF_DBG_INST DBGPR_2("pos = %d, %d", posBcast[ll].gpad, posBcast[ll].time);
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    fillScratchPad_charge_t(
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

char countPeaksScratchpad(
                    ushort  ll,
        local const uchar  *isPeak)
{
    char peaks = 0;
    for (uchar i = 0; i < 8; i++)
    {
        peaks += isPeak[ll * 8 + i];
    }

    return peaks;
}

ushort partition(
                    ushort     ll, 
                    bool       pred, 
                    ushort     partSize,
        local const char      *pcBcastIn,
        local const ChargePos *posBcastIn,
        local       char      *pcBcastOut,
        local       ChargePos *posBcastOut)
{
    bool participates = ll < partSize;

    IF_DBG_INST DBGPR_1("partSize = %d", partSize);

    ushort lpos = work_group_scan_inclusive_add(!pred && participates);
    IF_DBG_GROUP DBGPR_3("ll = %d, pred = %d, lpos = %d", ll, pred, lpos);

    ushort rpos = work_group_scan_inclusive_add( pred && participates);
    IF_DBG_GROUP DBGPR_3("ll = %d, pred = %d, rpos = %d", ll, pred, rpos);

    ushort part = work_group_broadcast(lpos, SCRATCH_PAD_WORK_GROUP_SIZE-1);

    IF_DBG_INST DBGPR_1("part = %d", part);

    lpos -= 1;
    rpos += part-1;
    ushort pos = (participates) ? ((pred) ? rpos : lpos) : ll;

    IF_DBG_GROUP DBGPR_2("ll = %d, pos = %d", ll, pos);

    pcBcastOut[pos]  = pcBcastIn[ll];
    posBcastOut[pos] = posBcastIn[ll];

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
        global       charge_t *chargeMap,
        global       char     *peakCountMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);

    CHARGE(chargeMap, gpad, myDigit.time) = 0.f;
    PEAK_COUNT(peakCountMap, gpad, myDigit.time) = 1;
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
    IF_DBG_INST printf("Looking for peaks (using LDS)\n");
    const ushort N = 8;
    local ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
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

    const global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    IS_PEAK(peakMap, gpad, myDigit.time) = peak;
}


kernel
void countPeaks(
        global const uchar    *peakMap,
        global const Digit    *digits,
               const uint      digitnum,
        global const uchar    *isPeak,
        global       char     *peakCountMap)
{
    size_t idx = get_global_linear_id();

    bool iamDummy = (idx >= digitnum);
    /* idx = select(idx, (size_t)(digitnum-1), (size_t)iamDummy); */
    idx = iamDummy ? digitnum-1 : idx;

    Digit myDigit = digits[idx];

    global_pad_t gpad = tpcGlobalPadIdx(myDigit.row, myDigit.pad);
    bool iamPeak  = isPeak[idx];

    char peakCount = 0;
#if defined(BUILD_CLUSTER_SCRATCH_PAD)
    ushort ll = get_local_linear_id();

    const ushort N = 8;

    local ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    local ChargePos posBcast2[SCRATCH_PAD_WORK_GROUP_SIZE];
    local char      pcBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    local char      pcBcast2[SCRATCH_PAD_WORK_GROUP_SIZE];
    local uchar     buf[SCRATCH_PAD_WORK_GROUP_SIZE * N];

    posBcast1[ll]    = (ChargePos){gpad, myDigit.time};
    pcBcast1[ll] = 0;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    int numOfDummies = work_group_reduce_add((int) iamDummy);

    IF_DBG_INST DBGPR_1("dummies = %d", numOfDummies);

    ushort in3x3 = partition(
            ll, 
            iamPeak,
            SCRATCH_PAD_WORK_GROUP_SIZE - numOfDummies, 
            pcBcast1, 
            posBcast1, 
            pcBcast2, 
            posBcast2);

    IF_DBG_INST DBGPR_1("in3x3 = %d", in3x3);

    local_id lid = {ll % N, ll / N};
    IF_DBG_INST DBGPR_0("Fill LDS 1.");
    fillScratchPad_uchar(peakMap, in3x3, lid, 0, N, INNER_NEIGHBORS, posBcast2, buf);
    /* fillScratchPadNaive(peakMap, in3x3, ll, 0, N, INNER_NEIGHBORS, posBcast2, buf); */
    if (ll < in3x3)
    {
        IF_DBG_INST DBGPR_0("Counting peaks in LDS.");
        peakCount = countPeaksScratchpad(ll, buf);

        if (peakCount > 0)
        {
            pcBcast2[ll] = peakCount;
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    IF_DBG_GROUP DBGPR_2("ll = %d, peakCount = %d", ll, peakCount);


    IF_DBG_INST DBGPR_0("Partition 2.");
    ushort in5x5 = partition(
            ll,
            peakCount > 0,
            in3x3,
            pcBcast2,
            posBcast2,
            pcBcast1,
            posBcast1);

    IF_DBG_INST DBGPR_0("Fill LDS 2.");
    fillScratchPad_uchar(peakMap, in5x5, lid, 0, N, OUTER_NEIGHBORS, posBcast1, buf);
    /* fillScratchPadNaive(peakMap, in5x5, ll, 0, N, OUTER_NEIGHBORS, posBcast1, buf); */
    if (ll < in5x5)
    {
        IF_DBG_INST DBGPR_0("Counting peaks in LDS.");
        peakCount = countPeaksScratchpad(ll, buf);
    }

    IF_DBG_INST DBGPR_0("Fill LDS 3.");
    fillScratchPad_uchar(peakMap, in5x5, lid, 8, N, OUTER_NEIGHBORS, posBcast1, buf);
    /* fillScratchPadNaive(peakMap, in5x5, ll, 8, N, OUTER_NEIGHBORS, posBcast1, buf); */
    if (ll < in5x5)
    {
        IF_DBG_INST DBGPR_0("Counting peaks in LDS.");
        peakCount += countPeaksScratchpad(ll, buf);
        peakCount *= -1;
        pcBcast1[ll] = peakCount;
    }

    IF_DBG_GROUP DBGPR_2("ll = %d, peakCount = %d", ll, peakCount);

    IF_DBG_INST DBGPR_0("Write results back.");

    if (iamDummy)
    {
        return;
    }

    char pc = pcBcast1[ll];
    ChargePos pos = posBcast1[ll];
    PEAK_COUNT(peakCountMap, pos.gpad, pos.time) = pc;
#else
    peakCount = countPeaksAroundDigit(gpad, myDigit.time, peakMap);

    if (iamDummy)
    {
        return;
    }

    /* peakCount = select(peakCount, (uchar) (PCMask_Has3x3Peak | 1), (uchar)iamPeak); */
    peakCount = iamPeak ? 1 : peakCount;

    PEAK_COUNT(peakCountMap, gpad, myDigit.time) = peakCount;
#endif
}


kernel
void computeClusters(
        global const charge_t      *chargeMap,
        global const char          *peakCountMap,
        global const Digit         *digits,
                     uint           clusternum,
        global       ClusterNative *clusters,
        global       row_t         *rows,
        global       uchar         *aboveQTotCutoff,
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
    local char      bufPeakCount[SCRATCH_PAD_WORK_GROUP_SIZE * N];
    local uchar     innerAboveThreshold[SCRATCH_PAD_WORK_GROUP_SIZE];

    buildClusterScratchPad(
            chargeMap,
            peakCountMap,
            (ChargePos){gpad, myDigit.time},
            N,
            posBcast,
            buf,
            bufPeakCount,
            innerAboveThreshold,
            &pc);
#else
    buildClusterNaive(chargeMap, peakCountMap, &pc, gpad, myDigit.time);
#endif

    finalize(&pc, &myDigit);


    ClusterNative myCluster;
    toNative(&pc, &myDigit, &myCluster);

    clusters[idx] = myCluster;
    rows[idx] = myDigit.row;

    IS_PEAK(peakMap, gpad, myDigit.time) = 0;

#if defined(CUT_QTOT)
    aboveQTotCutoff[idx] = (pc.Q > QTOT_THRESHOLD);
#else
    aboveQTotCutoff[idx] = true;
#endif
}
