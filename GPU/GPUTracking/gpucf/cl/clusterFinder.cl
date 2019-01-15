#include "shared/Cluster.h"
#include "shared/Digit.h"
#include "shared/tpc.h"


#define CHARGE(map, row, pad, time) \
    map[TPC_PADS_PER_ROW_PADDED*TPC_MAX_TIME_PADDED*row \
         +TPC_MAX_TIME_PADDED*(pad+PADDING)+time+PADDING]

#define DIGIT_CHARGE(map, digit) CHARGE(map, digit.row, digit.pad, digit.time)

constant float CHARGE_THRESHOLD = 2;
constant float OUTER_CHARGE_THRESHOLD = 0;

constant int2 LEQ_NEIGHBORS[4] = {(int2)(-1, -1), (int2)(-1, 0), (int2)(0, -1), (int2)(1, -1)};
constant int2 LQ_NEIGHBORS[4]  = {(int2)(-1, 1), (int2)(0, 1), (int2)(1, 1), (int2)(0, 1)};


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

void extendTo5x5Cluster(
        global const float   *chargeMap,
                     Cluster *myCluster,
                     int      row,
                     int      pad,
                     int      time)
{
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, -1, -2, -1);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, -1, -2, -2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, -1, -1, -2);

    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, -1, 1, -2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, -1, 2, -2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, -1, 2, -1);

    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, 1, -2, 1);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, 1, -2, 2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1, 1, -1, 2);

    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, 1, -2, 1);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, 1, -2, 2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1, 1, -1, 2);

    updateClusterOuter(chargeMap, myCluster, row, pad, time, 0, -1, 0, -2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 1,  0, 2, 0);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, 0,  1, 0, 2);
    updateClusterOuter(chargeMap, myCluster, row, pad, time, -1,  0, -2, 0);
}

bool tryBuild3x3Cluster(
        global const float   *chargeMap, 
                     int      row, 
                     int      pad, 
                     int      time, 
                     Cluster *cluster)
{
    bool isClusterCenter = true; 
    float myCharge = CHARGE(chargeMap, row, pad, time);

    for (int i = 0; i < sizeof(LEQ_NEIGHBORS)/sizeof(int2); i++)
    {
        int dp = LEQ_NEIGHBORS[i].x;
        int dt = LEQ_NEIGHBORS[i].y;
        float otherCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
        isClusterCenter &= (otherCharge <= myCharge);
        updateCluster(cluster, otherCharge, dp, dt);
    }

    for (int i = 0; i < sizeof(LQ_NEIGHBORS)/sizeof(int2); i++)
    {
        int dp = LQ_NEIGHBORS[i].x;
        int dt = LQ_NEIGHBORS[i].y;
        float otherCharge = CHARGE(chargeMap, row, pad+dp, time+dt);
        isClusterCenter &= (otherCharge < myCharge);
        updateCluster(cluster, otherCharge, dp, dt);
    }

    return isClusterCenter;
}

void finalizeCluster(Cluster *myCluster, const Digit *myDigit)
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

    myCluster->QMax      = myDigit->charge;
    myCluster->padMean   = padMean;
    myCluster->timeMean  = timeMean;
    myCluster->timeSigma = timeSigma;
    myCluster->padSigma  = padSigma;

    myCluster->cru = myDigit->cru;
    myCluster->row = myDigit->row;
}


kernel
void testStructCopy(
        global const Digit   *digits,
        global const float   *chargeMap,
        global       Cluster *clusters)
{
    int idx = get_global_id(0);

    Digit myDigit = digits[idx]; 

    Cluster myCluster;

    myCluster.cru = myDigit.cru;
    myCluster.row = myDigit.row;
    myCluster.Q        = DIGIT_CHARGE(chargeMap, myDigit);
    myCluster.QMax     = myDigit.charge;
    myCluster.padMean  = myDigit.pad;
    myCluster.timeMean = myDigit.time;
    myCluster.padSigma = myDigit.pad;
    myCluster.timeMean = myDigit.time;

    clusters[idx] = myCluster;
}


kernel
void digitsToChargeMap(
       global const Digit *digits,
       global       float *chargeMap)
{
    int idx = get_global_id(0);
    Digit myDigit = digits[idx];

    DIGIT_CHARGE(chargeMap, myDigit) = myDigit.charge;
}

kernel
void findClusters(
         global const float   *chargeMap,
         global const Digit   *digits,
         global       int     *digitIsClusterCenter,
         global       Cluster *clusters)
{
    int idx = get_global_id(0);
    Digit myDigit = digits[idx];

    if (myDigit.charge <= CHARGE_THRESHOLD)
    {
        digitIsClusterCenter[idx] = false;
        return;
    }

    Cluster myCluster = newCluster();
    bool isClusterCenter = tryBuild3x3Cluster(chargeMap, 
                                              myDigit.row, 
                                              myDigit.pad,
                                              myDigit.time,
                                              &myCluster);

    digitIsClusterCenter[idx] = isClusterCenter;

    if (!isClusterCenter)
    {
        return;
    }

    extendTo5x5Cluster(chargeMap,
                       &myCluster,
                       myDigit.row,
                       myDigit.pad,
                       myDigit.time);

    finalizeCluster(&myCluster, &myDigit);


    clusters[idx] = myCluster;
}
