#include "config.h"

#include "shared/tpc.h"



constant charge_t CHARGE_THRESHOLD = 2;
constant charge_t OUTER_CHARGE_THRESHOLD = 0;


typedef struct PartialCluster_s
{
    charge_t Q;
    charge_t padMean;
    charge_t padSigma;
    charge_t timeMean;
    charge_t timeSigma;
} PartialCluster;


void updateCluster(PartialCluster *cluster, charge_t charge, int dp, int dt)
{
    cluster->Q         += charge;
    cluster->padMean   += charge*dp;
    cluster->timeMean  += charge*dt;
    cluster->padSigma  += charge*dp*dp;
    cluster->timeSigma += charge*dt*dt;
}

void updateClusterOuter(
        global const charge_t       *chargeMap,
                     PartialCluster *cluster, 
                     row_t           row,
                     pad_t           pad,
                     timestamp       time,
                     int             dpOut,
                     int             dtOut)
{
    charge_t outerCharge = CHARGE(chargeMap, row, pad+dpOut, time+dtOut);

    outerCharge = (outerCharge > OUTER_CHARGE_THRESHOLD) ? outerCharge : 0;
    updateCluster(cluster, outerCharge, dpOut, dtOut);
}

void addCorner(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     row_t           row,
                     pad_t           pad,
                     timestamp       time,
                     int             dp,
                     int             dt)
{
    charge_t innerCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
    updateCluster(myCluster, innerCharge, dp, dt);
    
    if (innerCharge > CHARGE_THRESHOLD)
    {
        updateClusterOuter(chargeMap, myCluster, row, pad, time, 2*dp,   dt);
        updateClusterOuter(chargeMap, myCluster, row, pad, time,   dp, 2*dt);
        updateClusterOuter(chargeMap, myCluster, row, pad, time, 2*dp, 2*dt);
    }
}

void addLine(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     row_t           row,
                     pad_t           pad,
                     timestamp       time,
                     int             dp,
                     int             dt)
{
    charge_t innerCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
    updateCluster(myCluster, innerCharge, dp, dt);

    if (innerCharge > CHARGE_THRESHOLD)
    {
        updateClusterOuter(chargeMap, myCluster, row, pad, time, 2*dp, 2*dt);
    }
}

void buildCluster(
        global const charge_t       *chargeMap,
                     PartialCluster *myCluster,
                     row_t           row,
                     pad_t           pad,
                     timestamp       time)
{
    myCluster->Q = 0.f;
    myCluster->padMean = 0.f;
    myCluster->timeMean = 0.f;
    myCluster->padSigma = 0.f;
    myCluster->timeSigma = 0.f;

    // Add charges in top left corner:
    // O O o o o
    // O I i i o
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, row, pad, time, -1, -1);

    // Add upper charges
    // o o O o o
    // o i I i o
    // o i c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, row, pad, time,  0, -1);

    // Add charges in top right corner:
    // o o o O O
    // o i i I O
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, row, pad, time, 1, -1);


    // Add left charges
    // o o o o o
    // o i i i o
    // O I c i o
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, row, pad, time, -1,  0);

    // Add right charges
    // o o o o o
    // o i i i o
    // o i c I O
    // o i i i o
    // o o o o o
    addLine(chargeMap, myCluster, row, pad, time,  1,  0);


    // Add charges in bottom left corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // O I i i o
    // O O o o o
    addCorner(chargeMap, myCluster, row, pad, time, -1, 1);

    // Add bottom charges
    // o o o o o
    // o i i i o
    // o i c i o
    // o i I i o
    // o o O o o
    addLine(chargeMap, myCluster, row, pad, time,  0,  1);

    // Add charges in bottom right corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // o i i I O
    // o o o O O
    addCorner(chargeMap, myCluster, row, pad, time, 1, 1);
}


#define HALF_NEIGHBORS_NUM 4

constant int2 LEQ_NEIGHBORS[HALF_NEIGHBORS_NUM] = 
{
    (int2)(-1, -1), 
    (int2)(-1, 0), 
    (int2)(-1, 1),
    (int2)(0, -1)
};
constant int2 LQ_NEIGHBORS[HALF_NEIGHBORS_NUM]  = 
{
    (int2)(0, 1),
    (int2)(1, -1),
    (int2)(1, 0), 
    (int2)(1, 1)
};



bool isPeak(
               const Digit    *digit,
        global const charge_t *chargeMap)
{
    const charge_t myCharge = digit->charge;
    short time = digit->time;
    uchar row = digit->row;
    uchar pad = digit->pad;

    bool peak = true;

#define CMP_NEIGHBOR(dp, dt, cmpOp) \
    do \
    { \
        const charge_t otherCharge = CHARGE(chargeMap, row, pad+dp, time+dt); \
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


void finalizeCluster(
               const PartialCluster *pc,
               const Digit          *myDigit, 
        global const int            *globalToLocalRow,
        global const int            *globalRowToCru,
                     Cluster        *outCluster)
{
    charge_t totalCharge = pc->Q + myDigit->charge;
    charge_t padMean     = pc->padMean;
    charge_t timeMean    = pc->timeMean;
    charge_t padSigma    = pc->padSigma;
    charge_t timeSigma   = pc->timeSigma;

    padMean   /= totalCharge;
    timeMean  /= totalCharge;
    padSigma  /= totalCharge;
    timeSigma /= totalCharge;
    
    padSigma  = sqrt(padSigma  - padMean*padMean);

    timeSigma = sqrt(timeSigma - timeMean*timeMean);

    padMean  += myDigit->pad;
    timeMean += myDigit->time;


    outCluster->Q         = totalCharge;
    outCluster->QMax      = round(myDigit->charge);
    outCluster->padMean   = padMean;
    outCluster->timeMean  = timeMean;
    outCluster->timeSigma = timeSigma;
    outCluster->padSigma  = padSigma;

    outCluster->cru = globalRowToCru[myDigit->row];
    outCluster->row = globalToLocalRow[myDigit->row];
}


kernel
void fillChargeMap(
       global const Digit    *digits,
       global       charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    DIGIT_CHARGE(chargeMap, myDigit) = myDigit.charge;
}


kernel
void resetChargeMap(
        global const Digit    *digits,
        global       charge_t *chargeMap)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    DIGIT_CHARGE(chargeMap, myDigit) = 0.0f;
}


kernel
void findPeaks(
         global const charge_t *chargeMap,
         global const Digit    *digits,
         global       uchar    *isPeakPredicate)
{
    size_t idx = get_global_id(0);
    Digit myDigit = digits[idx];

    bool peak = isPeak(&myDigit, chargeMap);

    isPeakPredicate[idx] = peak;
}


kernel
void computeClusters(
        global const charge_t *chargeMap,
        global const Digit    *digits,
        global const int  *globalToLocalRow,
        global const int  *globalRowToCru,
        global       Cluster  *clusters)
{
    size_t idx = get_global_id(0);

    Digit myDigit = digits[idx];

    PartialCluster pc;
    buildCluster(chargeMap, &pc, myDigit.row, myDigit.pad, myDigit.time);

    Cluster myCluster;
    finalizeCluster(
            &pc, &myDigit, globalToLocalRow, globalRowToCru, &myCluster);

    clusters[idx] = myCluster;
}
