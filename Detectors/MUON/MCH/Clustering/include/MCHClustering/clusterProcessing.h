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

#ifndef O2_MCH_CLUSTERPROCESSING_H
#define O2_MCH_CLUSTERPROCESSING_H

#include "MCHClustering/ClusterConfig.h"

typedef std::pair<int, const double*> DataBlock_t;

namespace o2
{
namespace mch
{
void collectGroupMapping(Mask_t* padToMGrp, int nPads);
// Store the pad/group mapping in ClusterResult
/*
    void storeGroupMapping( const Groups_t *cath0Grp,
                        const PadIdx_t *mapCath0PadIdxToPadIdx, int nCath0,
                        const Groups_t *cath1Grp,
                        const PadIdx_t *mapCath1PadIdxToPadIdx, int nCath1);
*/
void collectSeeds(double* theta, o2::mch::Groups_t* thetaToGroup, int K);

} // namespace mch
} // namespace o2

extern "C" {
void setMathiesonVarianceApprox(int chId, double* theta, int K);

int clusterProcess(const double* xyDxyi, const o2::mch::Mask_t* cathi,
                   const o2::mch::Mask_t* saturated, const double* zi, int chId,
                   int nPads);

void collectTheta(double* theta, o2::mch::Groups_t* thetaToGroup, int N);

int getNbrOfPadsInGroups();

int getNbrOfProjPads();

// Inv ??? void collectPadsAndCharges(double* xyDxy, double* z, Groups_t*
// padToGroup, int nTot);

// Inv ??? void collectPadToCathGroup(Mask_t* padToMGrp, int nPads);

/*
// Store the pad/group mapping in ClusterResult
void storeGroupMapping( const o2::mch::Groups_t *cath0Grp,
                        const o2::mch::PadIdx_t *mapCath0PadIdxToPadIdx, int nCath0,
                        const o2::mch::Groups_t *cath1Grp,
                        const o2::mch::PadIdx_t *mapCath1PadIdxToPadIdx, int nCath1);
*/

void collectLaplacian(double* laplacian, int N);

void computeResidual(const double* xyDxy, const double* zObs,
                     const double* theta, int K, int N, double* residual);

void collectResidual(double* residual, int N);

void computeMathiesonResidual(const double* xyDxy, const o2::mch::Mask_t* cath,
                              const double* zObs, const double* theta, int chId,
                              int K, int N, double* residual);

int getKThetaInit();

void collectThetaInit(double* thetai, int N);

int getNbrOfThetaEMFinal();

void collectThetaEMFinal(double* thetaEM, int K);

void cleanClusterProcessVariables();
}
#endif // O2_MCH_CLUSTERPROCESSING_H_
