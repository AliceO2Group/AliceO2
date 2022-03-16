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

#ifndef O2_MCH_CLUSTERPROCESSING_H_
#define O2_MCH_CLUSTERPROCESSING_H_

typedef std::pair<int, const double*> DataBlock_t;

extern "C" {
void setMathiesonVarianceApprox(int chId, double* theta, int K);

int clusterProcess(const double* xyDxyi, const Mask_t* cathi, const Mask_t* saturated, const double* zi,
                   int chId, int nPads);

void collectTheta(double* theta, Group_t* thetaToGroup, int N);

int getNbrOfPadsInGroups();

int getNbrOfProjPads();

void collectPadsAndCharges(double* xyDxy, double* z, Group_t* padToGroup, int nTot);

void collectPadToCathGroup(Mask_t* padToMGrp, int nPads);

void collectLaplacian(double* laplacian, int N);

void computeResidual(const double* xyDxy, const double* zObs, const double* theta, int K, int N, double* residual);

void collectResidual(double* residual, int N);

void computeMathiesonResidual(const double* xyDxy, const Mask_t* cath, const double* zObs, const double* theta, int chId, int K, int N, double* residual);

int getKThetaInit();

void collectThetaInit(double* thetai, int N);

int getNbrOfThetaEMFinal();

void collectThetaEMFinal(double* thetaEM, int K);

void cleanClusterProcessVariables(int uniqueCath);
}
#endif
