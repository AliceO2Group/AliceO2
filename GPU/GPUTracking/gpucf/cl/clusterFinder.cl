#include "config.h"

#include "shared/tpc.h"


#define CHARGE(map, row, pad, time) \
    map[TPC_PADS_PER_ROW_PADDED*TPC_MAX_TIME_PADDED*(row) \
         +TPC_MAX_TIME_PADDED*((pad)+PADDING)+(time)+PADDING]

#define DIGIT_CHARGE(map, digit) CHARGE(map, digit.row, digit.pad, digit.time)


constant float CHARGE_THRESHOLD = 2;
constant float OUTER_CHARGE_THRESHOLD = 0;

constant int HALF_NEIGHBORS_NUM = 4;
constant int2 LEQ_NEIGHBORS[HALF_NEIGHBORS_NUM] = {(int2)(-1, -1), (int2)(-1, 0), (int2)(0, -1), (int2)(1, -1)};
constant int2 LQ_NEIGHBORS[HALF_NEIGHBORS_NUM]  = {(int2)(-1, 1), (int2)(1, 0), (int2)(1, 1), (int2)(0, 1)};


Cluster newCluster()
{
    Cluster c = {0, 0, 0, 0, 0, 0, 0, 0};
    return c;
}

void updateCluster(Cluster *cluster, float charge, int dp, int dt)
{
    cluster->Q         += charge;
    cluster->padMean   += charge*dp;
    cluster->timeMean  += charge*dt;
    cluster->padSigma  += charge*dp*dp;
    cluster->timeSigma += charge*dt*dt;
}

void updateClusterOuter(
        global const float   *chargeMap,
                     Cluster *cluster, 
                     int      row,
                     int      pad,
                     int      time,
                     int      dpIn, 
                     int      dtIn,
                     int      dpOut,
                     int      dtOut)
{
    float innerCharge = CHARGE(chargeMap, row, pad+dpIn, time+dtIn);
    float outerCharge = CHARGE(chargeMap, row, pad+dpOut, time+dtOut);

    if (   innerCharge >       CHARGE_THRESHOLD 
        && outerCharge > OUTER_CHARGE_THRESHOLD) 
    {
        updateCluster(cluster, outerCharge, dpOut, dtOut);
    }
}

void addCorner(
        global const float   *chargeMap,
                     Cluster *myCluster,
                     int      row,
                     int      pad,
                     int      time,
                     int      dp,
                     int      dt)
{
    float innerCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
    updateCluster(myCluster, innerCharge, dp, dt);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, dp, dt, 2*dp,   dt);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, dp, dt,   dp, 2*dt);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, dp, dt, 2*dp, 2*dt);
}

void addLine(
        global const float   *chargeMap,
                     Cluster *myCluster,
                     int      row,
                     int      pad,
                     int      time,
                     int      dp,
                     int      dt)
{
    float innerCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
    updateCluster(myCluster, innerCharge, dp, dt);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, dp, dt, 2*dp, 2*dt);
}

void buildCluster(
        global const float   *chargeMap,
                     Cluster *myCluster,
                     int      row,
                     int      pad,
                     int      time)
{
    myCluster->Q = 0;
    myCluster->QMax = 0;
    myCluster->padMean = 0;
    myCluster->timeMean = 0;
    myCluster->padSigma = 0;
    myCluster->timeSigma = 0;

    // Add charges in top left corner:
    // O O o o o
    // O I i i o
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, row, pad, time, -1, -1);

    // Add charges in top right corner:
    // o o o O O
    // o i i I O
    // o i c i o
    // o i i i o
    // o o o o o
    addCorner(chargeMap, myCluster, row, pad, time, 1, -1);

    // Add charges in bottom right corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // o i i I O
    // o o o O O
    addCorner(chargeMap, myCluster, row, pad, time, 1, 1);

    // Add charges in bottom left corner:
    // o o o o o
    // o i i i o
    // o i c i o
    // O I i i o
    // O O o o o
    addCorner(chargeMap, myCluster, row, pad, time, -1, 1);

    // Add remaining charges:
    // o o O o o
    // o i I i o
    // O I c I O
    // o i I i o
    // o o O o o
    addLine(chargeMap, myCluster, row, pad, time,  0, -1);
    addLine(chargeMap, myCluster, row, pad, time,  1,  0);
    addLine(chargeMap, myCluster, row, pad, time,  0,  1);
    addLine(chargeMap, myCluster, row, pad, time, -1,  0);
}


bool isPeak(
               const Digit *digit,
        global const float *chargeMap)
{
    float myCharge = digit->charge;
    short time = digit->time;
    uchar row = digit->row;
    uchar pad = digit->pad;

    bool peak = true;

    for (int i = 0; i < HALF_NEIGHBORS_NUM; i++)
    {
        int dp = LEQ_NEIGHBORS[i].x;
        int dt = LEQ_NEIGHBORS[i].y;
        float otherCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
        peak &= (otherCharge <= myCharge);
    }

    for (int i = 0; i < HALF_NEIGHBORS_NUM; i++)
    {
        int dp = LQ_NEIGHBORS[i].x;
        int dt = LQ_NEIGHBORS[i].y;
        float otherCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
        peak &= (otherCharge < myCharge);
    }

    peak &= (myCharge > CHARGE_THRESHOLD);

    return peak;
}


void finalizeCluster(
                     Cluster *myCluster, 
               const Digit   *myDigit, 
        global const int     *globalToLocalRow,
        global const int     *globalRowToCru)
{
    myCluster->Q += myDigit->charge;

    float totalCharge = myCluster->Q;
    float padMean     = myCluster->padMean;
    float timeMean    = myCluster->timeMean;
    float padSigma    = myCluster->padSigma;
    float timeSigma   = myCluster->timeSigma;

    padMean   /= totalCharge;
    timeMean  /= totalCharge;
    padSigma  /= totalCharge;
    timeSigma /= totalCharge;
    
    padSigma  = sqrt(padSigma  - padMean*padMean);

    timeSigma = sqrt(timeSigma - timeMean*timeMean);

    padMean  += myDigit->pad;
    timeMean += myDigit->time;

    myCluster->QMax      = round(myDigit->charge);
    myCluster->padMean   = padMean;
    myCluster->timeMean  = timeMean;
    myCluster->timeSigma = timeSigma;
    myCluster->padSigma  = padSigma;

    myCluster->cru = globalRowToCru[myDigit->row];
    myCluster->row = globalToLocalRow[myDigit->row];
}


kernel
void fillChargeMap(
       global const Digit *digits,
       global       float *chargeMap)
{
    int idx = get_global_id(0);
    Digit myDigit = digits[idx];

    DIGIT_CHARGE(chargeMap, myDigit) = myDigit.charge;
}

kernel
void findPeaks(
         global const float *chargeMap,
         global const Digit *digits,
         global       int   *isPeakPredicate)
{
    int idx = get_global_id(0);
    Digit myDigit = digits[idx];

    bool peak = isPeak(&myDigit, chargeMap);

    isPeakPredicate[idx] = peak;
}

kernel
void computeClusters(
        global const float   *chargeMap,
        global const Digit   *digits,
        global const int     *globalToLocalRow,
        global const int     *globalRowToCru,
        global       Cluster *clusters)
{
    int idx = get_global_id(0);

    Digit myDigit = digits[idx];

    Cluster myCluster;
    buildCluster(chargeMap, &myCluster, myDigit.row, myDigit.pad, myDigit.time);
    finalizeCluster(&myCluster, &myDigit, globalToLocalRow, globalRowToCru);

    clusters[idx] = myCluster;
}
