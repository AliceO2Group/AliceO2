#include "shared/Cluster.h"
#include "shared/Digit.h"
#include "shared/tpc.h"


#define CHARGE(map, row, pad, time) \
    map[TPC_PADS_PER_ROW*TPC_MAX_TIME*row+TPC_MAX_TIME*(pad+EMPTY_SPACE)+time+EMPTY_SPACE]

#define DIGIT_CHARGE(map, digit) CHARGE(map, digit->row, digit->pad, digit->time)

constant int CHARGE_THRESHOLD = 2;
constant int2 LEQ_NEIGHBORS[4] = {(int2)(-1, -1), (int2)(-1, 0), (int2)(0, -1), (int2)(1, -1)};
constant int2 LQ_NEIGHBORS[4]  = {(int2)(-1, 1), (int2)(0, 1), (int2)(1, 1), (int2)(0, 1)};


void updateCluster(Cluster *cluster, float charge, int dp, int dt)
{
    cluster->Q         += charge;
    cluster->padMean   += charge*dp;
    cluster->timeMean  += charge*dt;
    cluster->padSigma  += charge*dp*dp;
    cluster->timeSigma += charge*dt*dt;
}

bool tryBuild3x3Cluster(const float *chargeMap, int row, int pad, int time, Cluster *cluster)
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

    cluster->Q += myCharge;
    float totalCharge = cluster->Q;
    float padMean     = cluster->padMean;
    float timeMean    = cluster->timeMean;
    float padSigma    = cluster->padSigma;
    float timeSigma   = cluster->timeSigma;

    padMean   /= totalCharge;
    timeMean  /= totalCharge;
    padSigma  /= totalCharge;
    timeSigma /= totalCharge;
    
    padSigma  = sqrt(padSigma  - padMean*padMean);
    timeSigma = sqrt(timeSigma - timeSigma*timeSigma);

    padMean  += pad;
    timeMean += time;

    cluster->QMax = myCharge;
    cluster->padMean   = padMean;
    cluster->timeMean  = timeMean;
    cluster->timeSigma = timeSigma;
    cluster->padSigma  = padSigma;

    return isClusterCenter;
}

kernel
void digitsToChargeMap(global const Digit *digits,
                       global float *chargeMap)
{
    int idx = get_global_id(0);
    const Digit *myDigit = &digits[idx];

    DIGIT_CHARGE(chargeMap, myDigit) = myDigit->charge;
}


kernel
void find3x3Clusters(global const float *chargeMap,
                     global const Digit *digits,
                     global int *digitIsClusterCenter,
                     global Cluster *clusters)
{
    int idx = get_global_id(0);
    const Digit *myDigit = &digits[idx];

    if (myDigit->charge <= CHARGE_THRESHOLD)
    {
        return;
    }

    Cluster *myCluster = &clusters[idx];
    bool isClusterCenter = tryBuild3x3Cluster(
                            chargeMap, 
                            myDigit->row, 
                            myDigit->pad,
                            myDigit->time,
                            myCluster);

    digitIsClusterCenter[idx] = isClusterCenter;
}
