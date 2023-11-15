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
#include "mathiesonFit.h"

/* Inv
static InspectModel inspectModel={.nbrOfProjPads=0, .projectedPads=0,
.projGroups=0, .thetaInit=0, .kThetaInit=0, .totalNbrOfSubClusterPads=0,
.totalNbrOfSubClusterThetaEMFinal=0, .nCathGroups=0, .padToCathGrp=0};
*/

namespace o2
{
namespace mch
{
extern ClusterConfig clusterConfig;
}
} // namespace o2

static InspectModel inspectModel;
// Used when several sub-cluster occur in the precluster
// Append the new hits/clusters in the thetaList of the pre-cluster
void copyInGroupList(const double* values, int N, int item_size,
                     std::vector<o2::mch::DataBlock_t>& groupList)
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

  if (o2::mch::clusterConfig.inspectModelLog >= o2::mch::ClusterConfig::info) {
    printf("collectTheta : nbrOfGroups with clusters = %lu\n", inspectModel.subClusterThetaFitList.size());
  }
  for (int h = 0; h < inspectModel.subClusterThetaFitList.size(); h++) {
    int k = inspectModel.subClusterThetaFitList[h].first;
    if (o2::mch::clusterConfig.inspectModelLog >= o2::mch::ClusterConfig::info) {
      o2::mch::printTheta("  ", 1.0,
                          inspectModel.subClusterThetaFitList[h].second,
                          inspectModel.subClusterThetaFitList[h].first);
    }
    o2::mch::copyTheta(inspectModel.subClusterThetaFitList[h].second, k,
                       &theta[sumK], K, k);
    if (thetaToGroup) {
      o2::mch::vectorSetShort(&thetaToGroup[sumK], h + 1, k);
    }
    sumK += k;
    if (o2::mch::clusterConfig.inspectModelLog >= o2::mch::ClusterConfig::info) {
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

  // Timing
  for (int i = 0; i < 4; i++) {
    inspectModel.duration[i] = 0;
  }
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
  if (o2::mch::clusterConfig.inspectModelLog >= o2::mch::ClusterConfig::info) {
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

void inspectOverWriteQ(int which, const double* qPixels)
{
  int G = inspectPadProcess.xyDxyQPixels[which].size();
  /// Last Group
  int N = inspectPadProcess.xyDxyQPixels[which][G - 1].first;
  if (N != 0) {
    double* xyDxyQ = inspectPadProcess.xyDxyQPixels[which][G - 1].second;
    double* q = &xyDxyQ[4 * N];
    o2::mch::vectorCopy(qPixels, N, q);
    o2::mch::vectorPrint("inspectOverWriteQ ???", q, N);
  }
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
  o2::mch::DataBlock_t db = {N, xyDxyQ};
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

void InspectModelChrono(int type, bool end)
{
  if (type == -1) {
    // printf("Duration all=%f localMax=%f fitting=%f\n", inspectModel.duration[0], inspectModel.duration[1], inspectModel.duration[2]);
    return;
  }
  if (!end) {
    // Start
    inspectModel.startTime[type] = std::chrono::high_resolution_clock::now();
  } else {
    std::chrono::time_point<std::chrono::high_resolution_clock> tEnd;
    tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ = tEnd - inspectModel.startTime[type];
    inspectModel.duration[type] += duration_.count();
  }
}

/*
int f_ChargeIntegralMag(const gsl_vector* gslParams, void* dataFit,
                        gsl_vector* residuals)
{
  o2::mch::funcDescription_t* dataPtr = (o2::mch::funcDescription_t*)dataFit;
  int N = dataPtr->N;
  int K = dataPtr->K;
  const double* x = dataPtr->x_ptr;
  const double* y = dataPtr->y_ptr;
  const double* dx = dataPtr->dx_ptr;
  const double* dy = dataPtr->dy_ptr;
  const o2::mch::Mask_t* cath = dataPtr->cath_ptr;
  const double* zObs = dataPtr->zObs_ptr;
  o2::mch::Mask_t* notSaturated = dataPtr->notSaturated_ptr;
  int chamberId = dataPtr->chamberId;
  // double* cathWeights = dataPtr->cathWeights_ptr;
  // double* cathMax = dataPtr->cathMax_ptr;
  // double* zCathTotalCharge = dataPtr->zCathTotalCharge_ptr;

  // Parameters
  const double* params = gsl_vector_const_ptr(gslParams, 0);
  // Note:
  //  mux = mu[0:K-1]
  //  muy = mu[K:2K-1]
  const double* mu = o2::mch::getMuAndW(dataPtr->thetaInit, K);
  double* w = (double*)&params[0];

  // Set constrain: sum_(w_k) = 1
  double lastW = 1.0 - o2::mch::vectorSum(w, K - 1);
  //
  // Display paramameters (w, mu_x, mu_x
  if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::debug) {
    printf("  Function evaluation at:\n");
    for (int k = 0; k < K; k++) {
      printf("    mu_k[%d] = %g %g \n", k, mu[k], mu[K + k]);
    }
    for (int k = 0; k < K - 1; k++) {
      printf("    w_k[%d] = %g \n", k, w[k]);
    }
    // Last W
    printf("    w_k[%d] = %g \n", K - 1, lastW);
  }

  // Charge Integral on Pads
  double z[N];
  o2::mch::vectorSetZero(z, N);
  double zTmp[N];
  //
  double xyInfSup[4 * N];
  double* xInf = o2::mch::getXInf(xyInfSup, N);
  double* xSup = o2::mch::getXSup(xyInfSup, N);
  double* yInf = o2::mch::getYInf(xyInfSup, N);
  double* ySup = o2::mch::getYSup(xyInfSup, N);

  // Compute the pads charge considering the
  // Mathieson set w_k, mu_x, mu_y
  // TODO: Minor optimization  avoid to
  // compute  x[:] - dx[:]  i.E use xInf / xSup
  for (int k = 0; k < K; k++) {
    // xInf[:] = x[:] - dx[:] - muX[k]
    o2::mch::vectorAddVector(x, -1.0, dx, N, xInf);
    o2::mch::vectorAddScalar(xInf, -mu[k], N, xInf);
    // xSup = xInf + 2.0 * dxy[0]
    o2::mch::vectorAddVector(xInf, 2.0, dx, N, xSup);
    // yInf = xy[1] - dxy[1] - mu[k,1]
    // ySup = yInf + 2.0 * dxy[1]
    o2::mch::vectorAddVector(y, -1.0, dy, N, yInf);
    o2::mch::vectorAddScalar(yInf, -mu[K + k], N, yInf);
    // ySup = yInf + 2.0 * dxy[0]
    o2::mch::vectorAddVector(yInf, 2.0, dy, N, ySup);
    //
    o2::mch::compute2DPadIntegrals(xInf, xSup, yInf, ySup, N, chamberId, zTmp);
    // Multiply by the weight w[k]
    double wTmp = (k != K - 1) ? w[k] : lastW;
    o2::mch::vectorAddVector(z, wTmp, zTmp, N, z);
  }
  // ??? vectorPrint("z", z, N);
  // ??? vectorPrint("zObs", zObs, N);
  //
  // Normalisation
  //
  double totalCharge = o2::mch::vectorSum(zObs, N);
  double cathCharge[2] = {0., 0.};
  o2::mch::Mask_t mask[N];
  cathCharge[0] = o2::mch::vectorMaskedSum(zObs, mask, N);
  cathCharge[1] = totalCharge - cathCharge[0];
  for (int i = 0; i < N; i++) {
    if (mask[i] == 0) {
      z[i] = z[i] * cathCharge[0];
    } else {
      z[i] = z[i] * cathCharge[1];
    }
  }
  //
  // Compute residual
  for (int i = 0; i < N; i++) {
    // Don't consider saturated pads (notSaturated[i] = 0)
    double mask = notSaturated[i];
    if ((notSaturated[i] == 0) && (z[i] < zObs[i])) {
      // Except those charge < Observed charge
      mask = 1.0;
    }
    //
    // Residuals with penalization
    //
    gsl_vector_set(residuals, i, mask * ((zObs[i] - z[i])));
    //
    // Without penalization
    // gsl_vector_set(residuals, i, mask * (zObs[i] - z[i]) + 0 * wPenal);
    //
    // Other studied penalization
    // gsl_vector_set(residuals, i, (zObs[i] - z[i]) * (1.0 + cathPenal) +
    // wPenal);
  }
  return GSL_SUCCESS;
}

void printStateMag(int iter, gsl_multifit_fdfsolver* s, int K)
{
  printf("  Fitting iter=%3d |f(x)|=%g\n", iter, gsl_blas_dnrm2(s->f));
  printf("    w:");
  int k = 0;
  double sumW = 0;
  for (; k < K - 1; k++) {
    double w = gsl_vector_get(s->x, k);
    sumW += w;
    printf(" %7.3f", gsl_vector_get(s->x, k));
  }
  // Last w : 1.0 - sumW
  printf(" %7.3f", 1.0 - sumW);
  printf("\n");

  k = 0;
  printf("    dw:");
  for (; k < K - 1; k++) {
    printf(" % 7.3f", gsl_vector_get(s->dx, k));
  }
  printf("\n");
}

void fitMathiesonMag(const double* xyDxDy, const double* q,
                     const o2::mch::Mask_t* cath, const o2::mch::Mask_t* sat, int chId,
                     double* thetaInit, int K, int N,
                     double* thetaFinal, double* khi2)
{
  int status;
  if (K == 1) {
    o2::mch::vectorCopy(thetaInit, 5 * K, thetaFinal);
    return;
  }
  if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::info) {
    printf("\n> [fitMathiesonMag] Fitting \n");
  }
  //
  double* muAndWi = o2::mch::getMuAndW(thetaInit, K);
  //
  // Check if fitting is possible
  double* muAndWf = o2::mch::getMuAndW(thetaFinal, K);
  if (3 * K - 1 > N) {
    muAndWf[0] = NAN;
    muAndWf[K] = NAN;
    muAndWf[2 * K] = NAN;
    return;
  }

  o2::mch::funcDescription_t mathiesonData;

  if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
    o2::mch::vectorPrintShort("  cath", cath, N);
    o2::mch::vectorPrint("  q", q, N);
  }

  mathiesonData.N = N;
  mathiesonData.K = K;
  mathiesonData.x_ptr = o2::mch::getConstX(xyDxDy, N);
  mathiesonData.y_ptr = o2::mch::getConstY(xyDxDy, N);
  mathiesonData.dx_ptr = o2::mch::getConstDX(xyDxDy, N);
  mathiesonData.dy_ptr = o2::mch::getConstDY(xyDxDy, N);
  mathiesonData.cath_ptr = cath;
  mathiesonData.zObs_ptr = q;
  o2::mch::Mask_t notSaturated[N];
  o2::mch::vectorCopyShort(sat, N, notSaturated);
  o2::mch::vectorNotShort(notSaturated, N, notSaturated);
  mathiesonData.notSaturated_ptr = notSaturated;

  // Total Charge per cathode plane
  mathiesonData.cathWeights_ptr = nullptr;
  mathiesonData.cathMax_ptr = nullptr;
  mathiesonData.chamberId = chId;
  mathiesonData.zCathTotalCharge_ptr = nullptr;
  mathiesonData.verbose = 0;
  mathiesonData.thetaInit = thetaInit;
  //
  // Define Function, jacobian
  gsl_multifit_function_fdf f;
  f.f = &f_ChargeIntegralMag;
  f.df = nullptr;
  f.fdf = nullptr;
  f.n = N;
  f.p = K - 1;
  f.params = &mathiesonData;

  bool doFit = true;
  // K test
  // Sort w
  int maxIndex[K];
  for (int k = 0; k < K; k++) {
    maxIndex[k] = k;
  }
  double* w = &muAndWi[2 * K];
  std::sort(maxIndex, &maxIndex[K],
            [=](int a, int b) { return (w[a] > w[b]); });

  while (doFit) {
    // Select the best K's
    // Copy kTest max
    double wTest[K];
    // Mu part
    for (int k = 0; k < K; k++) {
      // Respecttively mux, muy, w
      wTest[k] = muAndWi[maxIndex[k] + 2 * K];
    }
    if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
      o2::mch::vectorPrint("  Selected w", wTest, K);
    }
    mathiesonData.K = K;
    f.p = K - 1;
    // Set initial parameters
    // Inv ??? gsl_vector_view params0 = gsl_vector_view_array(muAndWi, 3 * K -
    // 1);
    gsl_vector_view params0 = gsl_vector_view_array(wTest, K - 1);

    // Fitting method
    gsl_multifit_fdfsolver* s = gsl_multifit_fdfsolver_alloc(
      gsl_multifit_fdfsolver_lmsder, N, K - 1);
    // associate the fitting mode, the function, and the starting parameters
    gsl_multifit_fdfsolver_set(s, &f, &params0.vector);

    if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
      printStateMag(-1, s, K);
    }
    // double initialResidual = gsl_blas_dnrm2(s->f);
    double initialResidual = 0.0;
    // Fitting iteration
    status = GSL_CONTINUE;
    double residual = DBL_MAX;
    double prevResidual = DBL_MAX;
    double prevW[K - 1];
    for (int iter = 0; (status == GSL_CONTINUE) && (iter < 50); iter++) {
      // TODO: to speed if possible
      for (int k = 0; k < (K - 1); k++) {
        prevW[k] = gsl_vector_get(s->x, k);
      }
      // printf("  Debug Fitting iter=%3d |f(x)|=%g\n", iter,
      // gsl_blas_dnrm2(s->f));
      status = gsl_multifit_fdfsolver_iterate(s);
      if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
        printf("  Solver status = %s\n", gsl_strerror(status));
        printStateMag(iter, s, K);
      }

      status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
      if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
        printf("  Status multifit_test_delta = %d %s\n", status,
               gsl_strerror(status));
      }
      // Residu
      prevResidual = residual;
      residual = gsl_blas_dnrm2(s->f);
      // vectorPrint(" prevtheta", prevTheta, 3*K-1);
      // vectorPrint(" theta", s->dx->data, 3*K-1);
      // printf(" prevResidual, residual %f %f\n", prevResidual, residual );
      if (fabs(prevResidual - residual) < 1.0e-2) {
        // Stop iteration
        // Take the previous value of theta
        if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
          printf("  Stop iteration (dResidu~0), prevResidual=%f residual=%f\n",
                 prevResidual, residual);
        }
        for (int k = 0; k < (K - 1); k++) {
          gsl_vector_set(s->x, k, prevW[k]);
        }
        status = GSL_SUCCESS;
      }
    }
    double finalResidual = gsl_blas_dnrm2(s->f);
    bool keepInitialTheta =
      fabs(finalResidual - initialResidual) / initialResidual < 1.0e-1;

    // Khi2
    if (khi2 != nullptr) {
      // Khi2
      double chi = gsl_blas_dnrm2(s->f);
      double dof = N - (3 * K - 1);
      double c = fmax(1.0, chi / sqrt(dof));
      if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
        printf("  K=%d, chi=%f, chisq/dof = %g\n", K, chi * chi,
               chi * chi / dof);
      }
      khi2[0] = chi * chi / dof;
    }

    // ???? if (keepInitialTheta) {
    if (0) {
      // Keep the result of EM (GSL bug when no improvemebt)
      o2::mch::copyTheta(thetaInit, K, thetaFinal, K, K);
    } else {
      // Fitted parameters

      // Mu part
      for (int k = 0; k < K; k++) {
        muAndWf[k] = muAndWi[k];
        muAndWf[k + K] = muAndWi[k + K];
      }
      // w part
      double sumW = 0;
      for (int k = 0; k < K - 1; k++) {
        double w = gsl_vector_get(s->x, k);
        sumW += w;
        muAndWf[k + 2 * K] = w;
      }
      // Last w : 1.0 - sumW
      muAndWf[3 * K - 1] = 1.0 - sumW;
    }
    if (o2::mch::ClusterConfig::fittingLog >= o2::mch::ClusterConfig::detail) {
      printf("  status parameter error = %s\n", gsl_strerror(status));
    }
    gsl_multifit_fdfsolver_free(s);
    K = K - 1;
    // doFit = (K < 3) && (K > 0);
    doFit = false;
  } // while(doFit)
  // Release memory
  //
  return;
}
*/