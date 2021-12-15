// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _PADPROCESSING_H
#define _PADPROCESSING_H

#include "dataStructure.h"
#include "PadsPEM.h"

typedef struct {
  PadIdx_t i;
  PadIdx_t j;
} MapKToIJ_t;

typedef short Group_t;

// 8 neigbours + the center pad itself + separator (-1)
// TODO ??? Maybe find a limit with dx0, dx1, ...
// See Run2 ev=8, cluster=1105
static const int MaxNeighbors = 25 + 13; // 5x5 neighbours + 50 %

inline static const PadIdx_t* getNeighborsOf(const PadIdx_t* neigh, PadIdx_t i) { return &neigh[MaxNeighbors * i]; };
inline static PadIdx_t* getNeighborsOf(PadIdx_t* neigh, PadIdx_t i) { return &neigh[MaxNeighbors * i]; };

inline static PadIdx_t getTheFirstNeighborOf(PadIdx_t* neigh, PadIdx_t i) { return neigh[MaxNeighbors * i]; };

extern "C" {

void cleanInspectPadProcess();

int collectPixels(int which, int N, double* xyDxy, double* q);

// Helper to get results from Python
int getNbrProjectedPads();
//
void setNbrProjectedPads(int n);

void copyProjectedPads(double* xyDxy, double* chA, double* chB);

void collectProjectedMinMax(double* chMin, double* chMax);

PadIdx_t* getFirstNeighbors(const double* xyDxy, int N, int allocatedN, int verbose);

// Used when there is 1 cathode
// The neighbors list must be computed for the laplacian
void computeAndStoreFirstNeighbors(const double* xyDxy, int N, int allocatedN);

int projectChargeOnOnePlane(
  const double* xy0InfSup, const double* ch0,
  const double* xy1InfSup, const double* ch1,
  PadIdx_t N0, PadIdx_t N1, int includeAlonePads);

int projectChargeOnOnePlaneWithTheta(
  const double* xy0InfSup, const double* ch0,
  const double* xy1InfSup, const double* ch1,
  const double* chTheta0, const double* ch1Theta0,
  PadIdx_t N0, PadIdx_t N1, int includeAlonePads, double* qProj);

void buildProjectedSaturatedPads(const Mask_t* saturated0, const Mask_t* saturated1, Mask_t* saturatedProj);

int getConnectedComponentsOfProjPads(short* padGrp);
int getConnectedComponentsOfProjPadsWOIsolatedPads(short* padGrp);

int findLocalMaxWithLaplacian(const double* xyDxy, const double* z,
                              Group_t* padToGrp,
                              int nGroups,
                              int N, int K, double* laplacian, double* theta, PadIdx_t* thetaIndexes,
                              Group_t* thetaToGrp);

int findLocalMaxWithBothCathodes(const double* xyDxy0, const double* q0, int N0,
                                 const double* xyDxy1, const double* q1, int N1, const double* xyDxyProj, int NProj, int chId,
                                 const PadIdx_t* mapGrpIdxToI, const PadIdx_t* mapGrpIdxToJ, int nbrCath0, int nbrCath1,
                                 double* thetaOut, int kMax);

int findLocalMaxWithPET(double* xyDxyGrp0, double* qGrp0, Mask_t* saturateGrp0, int nbrOfPadsInTheGroupCath0,
                        double* xyDxyGrp1, double* qGrp1, Mask_t* saturateGrp1, int nbrOfPadsInTheGroupCath1,
                        double* xyDxyProj, double* qProj, int nProjPads, int chId,
                        // const PadIdx_t *mapGrpIdxToI, const PadIdx_t *mapGrpIdxToJ, int nbrCath0, int nbrCath1,
                        double* thetaL, int nbrOfPadsInTheGroupCath);

void inspectSavePixels(int which, o2::mch::Pads& pixels);

// ??? To remove
int findLocalMaxWithLaplacianV0(const double* xyDxy, const double* z, const PadIdx_t* grpIdxToProjIdx, int N, int xyDxyAllocated, double* laplacian, double* theta);

void assignPadsToGroupFromProjAndProjCharge(short* projPadGroup, double* chProj, int nProjPads,
                                            const PadIdx_t* cath0ToPadIdx, const PadIdx_t* cath1ToPadIdx,
                                            int nGrp, int nPads, short* padCathGrp);

void assignOneCathPadsToGroup(short* padGroup, int nPads, int nGrp, int nCath0, int nCath1, short* wellSplitGroup);

void assignCathPadsToGroupFromProj(short* padGroup, int nPads, int nGrp, int nCath0, int nCath1,
                                   short* wellSplitGroup, short* matGrpGrp);

int assignGroupToCathPads(short* projPadGroup, int nPads, int nGrp, int nCath0, int nCath1,
                          short* cath0ToGrp, short* cath1ToGrp);

int assignCathPadsToGroup(short* matGrpGrp, int nGrp, int nCath0, int nCath1, short* grpToGrp);

int assignPadsToGroupFromProj(short* projPadGroup, int nProjPads, const PadIdx_t*, const PadIdx_t*, int nGrp, int nPads,
                              short* padGrp);

int renumberGroups(short* grpToGrp, int nGrp);
int renumberGroupsV2(Mask_t* cath0Grp, int nbrCath0, Mask_t* cath1Grp, int nbrCath1, Mask_t* grpToGrp, int nGrp);

void storeProjectedPads(const double* xyDxyProj, const double* z, int nPads);
//
void copyCathToGrpFromProj(short* cath0Grp, short* cath1Grp, int nCath0, int nCath1);

void getMaskCathToGrpFromProj(short g, short* mask0, short* mask1, int nCath0, int nCath1);

int addIsolatedPadInGroups(const double* xyDxy, Mask_t* cathToGrp, int nbrCath, int cath, Mask_t* grpToGrp, int nGroups);

void updateProjectionGroups(short* projPadToGrp, int nProjPads, const short* cath0ToGrp, const short* cath1ToGrp);

void freeMemoryPadProcessing();

o2::mch::Pads* addBoundaryPads(const double* x_, const double* y_, const double* dx_, const double* dy_, const double* q_, const Mask_t* cath_, const Mask_t* sat_, int chamberId, int N);
//
void printNeighbors(const PadIdx_t* neigh, int N);
void printMatrixShort(const char* str, const short* matrix, int N, int M);
void printMatrixInt(const char* str, const int* matrix, int N, int M);
}
#endif // _PADPROCESSING_H
