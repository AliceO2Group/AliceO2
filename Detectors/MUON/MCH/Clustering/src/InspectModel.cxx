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

/// \file Cluster.h
/// \brief Definition of a class to reconstruct clusters with the gem MLEM
/// algorithm
///
/// \author Gilles Grasseau, Subatech

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <vector>

#include "InspectModel.h"
#include "MCHClustering/PadsPEM.h"
#include "mathUtil.h"
#include "mathieson.h"

/* Inv
static InspectModel inspectModel={.nbrOfProjPads=0, .projectedPads=0,
.projGroups=0, .thetaInit=0, .kThetaInit=0, .totalNbrOfSubClusterPads=0,
.totalNbrOfSubClusterThetaEMFinal=0, .nCathGroups=0, .padToCathGrp=0};
*/
static InspectModel inspectModel;

// Used when several sub-cluster occur in the precluster
// Append the new hits/clusters in the thetaList of the pre-cluster
void copyInGroupList(const double* values, int N, int item_size,
                     std::vector<DataBlock_t>& groupList)
{
  double* ptr = new double[N * item_size];
  // memcpy( (void *) ptr, (const void*) values, N*item_size*sizeof(double));
  o2::mch::vectorCopy(values, N * item_size, ptr);
  groupList.push_back(std::make_pair(N, ptr));
}

/* Inv
void appendInThetaList( const double *values, int N, std::vector< DataBlock_t >
&groupList) {
  // double *ptr = new double[N];
  // memcpy( (void *) ptr, (const void*) theta, N*sizeof(double));
  groupList.push_back( std::make_pair(N, values));
}
*/

void saveThetaEMInGroupList(const double* thetaEM, int K)
{
  int element_size = 5;
  copyInGroupList(thetaEM, K, element_size,
                  inspectModel.subClusterThetaEMFinal);
}

void saveThetaExtraInGroupList(const double* thetaExtra, int K)
{
  int element_size = 5;
  copyInGroupList(thetaExtra, K, element_size,
                  inspectModel.subClusterThetaExtra);
}

void saveThetaFitInGroupList(const double* thetaFit, int K)
{
  int element_size = 5;
  copyInGroupList(thetaFit, K, element_size,
                  inspectModel.subClusterThetaFitList);
}

void collectTheta(double* theta, o2::mch::Groups_t* thetaToGroup, int K)
{
  int sumK = 0;

  if (o2::mch::ClusterConfig::inspectModelLog >= o2::mch::ClusterConfig::info) {
    printf("collectTheta : nbrOfGroups with clusters = %lu\n", inspectModel.subClusterThetaFitList.size());
  }
  for (int h = 0; h < inspectModel.subClusterThetaFitList.size(); h++) {
    int k = inspectModel.subClusterThetaFitList[h].first;
    if (o2::mch::ClusterConfig::inspectModelLog >= o2::mch::ClusterConfig::info) {
      o2::mch::printTheta("  ",
                          inspectModel.subClusterThetaFitList[h].second,
                          inspectModel.subClusterThetaFitList[h].first);
    }
    o2::mch::copyTheta(inspectModel.subClusterThetaFitList[h].second, k,
                       &theta[sumK], K, k);
    if (thetaToGroup) {
      o2::mch::vectorSetShort(&thetaToGroup[sumK], h + 1, k);
    }
    sumK += k;
    if (o2::mch::ClusterConfig::inspectModelLog >= o2::mch::ClusterConfig::info) {
      printf("collect theta grp=%d,  grpSize=%d, adress=%p\n", h, k,
             inspectModel.subClusterThetaFitList[h].second);
    }
    delete[] inspectModel.subClusterThetaFitList[h].second;
  }
  if (sumK > K) {
    printf("Bad allocation for collectTheta sumK=%d greater than K=%d\n", sumK,
           K);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterThetaFitList.clear();
}

void savePadsOfSubCluster(const double* xyDxy, const double* q, int n)
{
  int element_size = 4;
  copyInGroupList(xyDxy, n, element_size, inspectModel.subClusterPadList);
  element_size = 1;
  copyInGroupList(q, n, element_size, inspectModel.subClusterChargeList);
}

void cleanInspectModel()
{
  //
  for (int i = 0; i < inspectModel.subClusterPadList.size(); i++) {
    delete[] inspectModel.subClusterPadList[i].second;
  }
  inspectModel.subClusterPadList.clear();
  //
  for (int i = 0; i < inspectModel.subClusterChargeList.size(); i++) {
    delete[] inspectModel.subClusterChargeList[i].second;
  }
  inspectModel.subClusterChargeList.clear();
  //
  for (int i = 0; i < inspectModel.subClusterThetaEMFinal.size(); i++) {
    delete[] inspectModel.subClusterThetaEMFinal[i].second;
  }
  inspectModel.subClusterThetaEMFinal.clear();
  //
  for (int i = 0; i < inspectModel.subClusterThetaExtra.size(); i++) {
    delete[] inspectModel.subClusterThetaExtra[i].second;
  }
  inspectModel.subClusterThetaExtra.clear();
  //
  for (int i = 0; i < inspectModel.subClusterThetaFitList.size(); i++) {
    delete[] inspectModel.subClusterThetaFitList[i].second;
  }
  inspectModel.subClusterThetaFitList.clear();
  //
  if (inspectModel.projectedPads != nullptr) {
    delete[] inspectModel.projectedPads;
    inspectModel.projectedPads = nullptr;
  }
  if (inspectModel.qProj != nullptr) {
    delete[] inspectModel.qProj;
    inspectModel.qProj = nullptr;
  }
  if (inspectModel.projGroups != nullptr) {
    delete[] inspectModel.projGroups;
    inspectModel.projGroups = nullptr;
  }
  if (inspectModel.thetaInit != nullptr) {
    delete[] inspectModel.thetaInit;
    inspectModel.thetaInit = nullptr;
  }
  //
  inspectModel.totalNbrOfSubClusterPads = 0;
  inspectModel.totalNbrOfSubClusterThetaEMFinal = 0;
  inspectModel.totalNbrOfSubClusterThetaExtra = 0;

  cleanPixels();
  // Cath group
  delete[] inspectModel.padToCathGrp;
  inspectModel.padToCathGrp = nullptr;
  inspectModel.nCathGroups = 0;
}

void finalizeInspectModel()
{
  int sumN = 0;
  for (int h = 0; h < inspectModel.subClusterPadList.size(); h++) {
    int n = inspectModel.subClusterPadList[h].first;
    sumN += n;
  }
  inspectModel.totalNbrOfSubClusterPads = sumN;
  //
  int sumK = 0;
  for (int h = 0; h < inspectModel.subClusterThetaEMFinal.size(); h++) {
    int k = inspectModel.subClusterThetaEMFinal[h].first;
    sumK += k;
  }
  inspectModel.totalNbrOfSubClusterThetaEMFinal = sumK;
  //
  sumK = 0;
  for (int h = 0; h < inspectModel.subClusterThetaExtra.size(); h++) {
    int k = inspectModel.subClusterThetaExtra[h].first;
    sumK += k;
  }
  inspectModel.totalNbrOfSubClusterThetaExtra = sumK;
}

int getNbrOfProjPads() { return inspectModel.nbrOfProjPads; }

int getNbrOfPadsInGroups() { return inspectModel.totalNbrOfSubClusterPads; }

int getNbrOfThetaEMFinal()
{
  return inspectModel.totalNbrOfSubClusterThetaEMFinal;
}

int getNbrOfThetaExtra() { return inspectModel.totalNbrOfSubClusterThetaExtra; }

void saveProjectedPads(const o2::mch::Pads* pads, double* qProj)
{
  int nbrOfProjPads = pads->getNbrOfPads();
  inspectModel.nbrOfProjPads = nbrOfProjPads;
  inspectModel.projectedPads = new double[nbrOfProjPads * 4];
  o2::mch::vectorCopy(pads->getX(), nbrOfProjPads,
                      &inspectModel.projectedPads[0]);
  o2::mch::vectorCopy(pads->getY(), nbrOfProjPads,
                      &inspectModel.projectedPads[1 * nbrOfProjPads]);
  o2::mch::vectorCopy(pads->getDX(), nbrOfProjPads,
                      &inspectModel.projectedPads[2 * nbrOfProjPads]);
  o2::mch::vectorCopy(pads->getDY(), nbrOfProjPads,
                      &inspectModel.projectedPads[3 * nbrOfProjPads]);
  inspectModel.qProj = qProj;
}

void collectProjectedPads(double* xyDxy, double* chA, double* chB)
{

  int nbrOfProjPads = inspectModel.nbrOfProjPads;
  for (int i = 0; i < 4; i++) {
    for (int k = 0; k < nbrOfProjPads; k++) {
      xyDxy[i * nbrOfProjPads + k] =
        inspectModel.projectedPads[i * nbrOfProjPads + k];
    }
  }
  for (int k = 0; k < nbrOfProjPads; k++) {
    chA[k] = inspectModel.qProj[k];
  }
  for (int k = 0; k < nbrOfProjPads; k++) {
    chB[k] = inspectModel.qProj[k];
  }
  // printf("collectProjectedPads nbrOfProjPads=%d\n", nbrOfProjPads);
  // o2::mch::vectorPrint( "  qProj=", chA, nbrOfProjPads);
}

void saveProjPadToGroups(o2::mch::Groups_t* projPadToGrp, int N)
{
  inspectModel.projGroups = new o2::mch::Groups_t[N];
  o2::mch::vectorCopyShort(projPadToGrp, N, inspectModel.projGroups);
}

int collectProjGroups(o2::mch::Groups_t* projPadToGrp)
{
  int N = inspectModel.nbrOfProjPads;
  o2::mch::vectorCopyShort(inspectModel.projGroups, N, projPadToGrp);
  return o2::mch::vectorMaxShort(projPadToGrp, N);
}
// ???
// Optim collectXXX can be replaced by getConstPtrXXX
void savePadToCathGroup(const o2::mch::Groups_t* cath0Grp,
                        const o2::mch::PadIdx_t* mapCath0PadIdxToPadIdx, int nCath0,
                        const o2::mch::Groups_t* cath1Grp,
                        const o2::mch::PadIdx_t* mapCath1PadIdxToPadIdx, int nCath1)
{
  inspectModel.padToCathGrp = new o2::mch::Groups_t[nCath0 + nCath1];
  if (cath0Grp != nullptr) {
    for (int p = 0; p < nCath0; p++) {
      inspectModel.padToCathGrp[mapCath0PadIdxToPadIdx[p]] = cath0Grp[p];
    }
  }
  if (cath1Grp != nullptr) {
    for (int p = 0; p < nCath1; p++) {
      // printf("savePadToCathGroup p[cath1 idx]=%d mapCath1PadIdxToPadIdx[p]=
      // %d, grp=%d\n", p, mapCath1PadIdxToPadIdx[p], cath1Grp[p]);
      inspectModel.padToCathGrp[mapCath1PadIdxToPadIdx[p]] = cath1Grp[p];
    }
  }
}

void collectPadToCathGroup(o2::mch::Mask_t* padToMGrp, int nPads)
{
  if (o2::mch::ClusterConfig::inspectModelLog >= o2::mch::ClusterConfig::info) {
    printf("collectPadToCathGroup nPads=%d\n", nPads);
  }
  o2::mch::vectorCopyShort(inspectModel.padToCathGrp, nPads, padToMGrp);
}

/// ???
void collectPadsAndCharges(double* xyDxy, double* z, o2::mch::Groups_t* padToGroup,
                           int N)
{
  int sumN = 0;
  for (int h = 0; h < inspectModel.subClusterPadList.size(); h++) {
    int n = inspectModel.subClusterPadList[h].first;
    o2::mch::copyXYdXY(inspectModel.subClusterPadList[h].second, n,
                       &xyDxy[sumN], N, n);
    o2::mch::vectorCopy(inspectModel.subClusterChargeList[h].second, n,
                        &z[sumN]);
    if (padToGroup) {
      o2::mch::vectorSetShort(&padToGroup[sumN], h + 1, n);
    }
    sumN += n;
    delete[] inspectModel.subClusterPadList[h].second;
    delete[] inspectModel.subClusterChargeList[h].second;
  }
  if (sumN > N) {
    printf("Bad allocation for collectTheta sumN=%d greater than N=%d\n", sumN,
           N);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterPadList.clear();
  inspectModel.subClusterChargeList.clear();
  inspectModel.totalNbrOfSubClusterPads = 0;
}

// ??? Unused
/*
void collectLaplacian( double *laplacian, int N) {
  o2::mch::vectorCopy( inspectModel.laplacian, N, laplacian );
}
 */

/* Unused
void collectResidual( double *residual, int N) {
  o2::mch::vectorCopy( inspectModel.residualProj, N, residual );
}
*/

int getKThetaInit() { return inspectModel.kThetaInit; }

void collectThetaInit(double* thetai, int N)
{
  o2::mch::vectorCopy(inspectModel.thetaInit, 5 * N, thetai);
}

/* Invalid
void collectThetaFit( double *thetaFit, int K) {
  int sumK=0;
  for (int h=0; h < inspectModel.subClusterThetaFitList.size(); h++) {
    int k = inspectModel.subClusterThetaFitList[h].first;
    o2::mch::copyTheta( inspectModel.subClusterThetaFitList[h].second, k,
&thetaFit[sumK], K, k ); sumK += k; delete[]
inspectModel.subClusterThetaFitList[h].second;
  }
  if ( sumK > K) {
    printf("Bad allocation for collectThetaFitList sumN=%d greater than N=%d\n",
sumK, K); throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterThetaFitList.clear();
  // ??? inspectModel.totalNbrOfSubClusterThetaEMFinal = 0;
}
*/

void collectThetaEMFinal(double* thetaEM, int K)
{
  int sumK = 0;
  for (int h = 0; h < inspectModel.subClusterThetaEMFinal.size(); h++) {
    int k = inspectModel.subClusterThetaEMFinal[h].first;
    o2::mch::copyTheta(inspectModel.subClusterThetaEMFinal[h].second, k,
                       &thetaEM[sumK], K, k);
    sumK += k;
    delete[] inspectModel.subClusterThetaEMFinal[h].second;
  }
  if (sumK > K) {
    printf("Bad allocation for collectThetaEMFinal sumN=%d greater than N=%d\n",
           sumK, K);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterThetaEMFinal.clear();
  inspectModel.totalNbrOfSubClusterThetaEMFinal = 0;
}

void collectThetaExtra(double* thetaExtra, int K)
{
  int sumK = 0;
  for (int h = 0; h < inspectModel.subClusterThetaExtra.size(); h++) {
    int k = inspectModel.subClusterThetaExtra[h].first;
    o2::mch::copyTheta(inspectModel.subClusterThetaExtra[h].second, k,
                       &thetaExtra[sumK], K, k);
    sumK += k;
    delete[] inspectModel.subClusterThetaExtra[h].second;
  }
  if (sumK > K) {
    printf("Bad allocation for collectThetaEMFinal sumN=%d greater than N=%d\n",
           sumK, K);
    throw std::overflow_error("Bad Allocation");
  }
  inspectModel.subClusterThetaExtra.clear();
  inspectModel.totalNbrOfSubClusterThetaExtra = 0;
}

// PadProcess
//

static InspectPadProcessing_t
  inspectPadProcess; //={.xyDxyQPixels ={{0,nullptr}, {0,nullptr},
                     //{0,nullptr},  {0,nullptr}}};
//.laplacian=0, .residualProj=0, .thetaInit=0, .kThetaInit=0,
//  .totalNbrOfSubClusterPads=0, .totalNbrOfSubClusterThetaEMFinal=0,
//  .nCathGroups=0, .padToCathGrp=0};

void cleanPixels()
{
  for (int i = 0; i < inspectPadProcess.nPixelStorage; i++) {
    int G = inspectPadProcess.xyDxyQPixels[i].size();
    for (int g = 0; g < G; g++) {
      if (inspectPadProcess.xyDxyQPixels[i][g].first != 0) {
        delete[] inspectPadProcess.xyDxyQPixels[i][g].second;
      }
      inspectPadProcess.xyDxyQPixels[i][g].first = 0;
    }
    inspectPadProcess.xyDxyQPixels[i].clear();
  }
}

int collectPixels(int which, int N, double* xyDxy, double* q)
{
  // which : select the pixel data
  // N : if N = 0, return the nbr of Pixels
  // xyDxy : 4N allocated array
  // q : N allocated array

  int nSrc = 0;
  const double* xyDxySrc;
  const double* qSrc;
  int G = inspectPadProcess.xyDxyQPixels[which].size();

  for (int g = 0; g < G; g++) {
    nSrc += inspectPadProcess.xyDxyQPixels[which][g].first;
  }
  if (N != nSrc) {
    N = 0;
  }

  if (N != 0) {
    int shift = 0;
    for (int g = 0; g < G; g++) {
      int n = inspectPadProcess.xyDxyQPixels[which][g].first;
      xyDxySrc = inspectPadProcess.xyDxyQPixels[which][g].second;
      qSrc = &xyDxySrc[4 * n];
      o2::mch::vectorCopy(&xyDxySrc[0 * n], n, &xyDxy[0 * N + shift]);
      o2::mch::vectorCopy(&xyDxySrc[1 * n], n, &xyDxy[1 * N + shift]);
      o2::mch::vectorCopy(&xyDxySrc[2 * n], n, &xyDxy[2 * N + shift]);
      o2::mch::vectorCopy(&xyDxySrc[3 * n], n, &xyDxy[3 * N + shift]);
      o2::mch::vectorCopy(qSrc, n, &q[shift]);
      shift += n;
    }
  }
  return nSrc;
}

void inspectSavePixels(int which, o2::mch::Pads& pixels)
{
  int N = pixels.getNbrOfPads();
  double* xyDxyQ = new double[5 * N];
  double* xyDxy = xyDxyQ;
  double* q = &xyDxyQ[4 * N];
  o2::mch::vectorCopy(pixels.getX(), N, xyDxy);
  o2::mch::vectorCopy(pixels.getY(), N, &xyDxy[N]);
  o2::mch::vectorCopy(pixels.getDX(), N, &xyDxy[2 * N]);
  o2::mch::vectorCopy(pixels.getDY(), N, &xyDxy[3 * N]);
  o2::mch::vectorCopy(pixels.getCharges(), N, q);
  DataBlock_t db = {N, xyDxyQ};
  inspectPadProcess.xyDxyQPixels[which].push_back(db);
  // printf("[inspectPadProcess], chanel=%d, nbrGrp=%ld\n", which,
  // inspectPadProcess.xyDxyQPixels[which].size() );
}

int getNbrProjectedPads() { return inspectModel.nbrOfProjPads; };

void setNbrProjectedPads(int n)
{
  inspectModel.nbrOfProjPads = n;
  // inspectModel.maxNbrOfProjPads= n;
};
