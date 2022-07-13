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

#ifndef O2_MCH_INSPECTMODEL_H_
#define O2_MCH_INSPECTMODEL_H_

#include <vector>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_vector.h>

#include "MCHClustering/ClusterConfig.h"
#include "MCHClustering/PadsPEM.h"
#include "MCHClustering/ClusterPEM.h"
#include "MCHClustering/clusterProcessing.h"

// Inspect data
typedef struct dummy_t {
  // Data on Projected Pads
  int nbrOfProjPads = 0;
  double* projectedPads = nullptr;
  double* qProj = nullptr; // Projected charges
  o2::mch::Groups_t* projGroups = nullptr;
  // Theta init
  double* thetaInit = nullptr;
  int kThetaInit = 0;
  // Data about subGroups
  int totalNbrOfSubClusterPads = 0;
  int totalNbrOfSubClusterThetaEMFinal = 0;
  int totalNbrOfSubClusterThetaExtra = 0;
  std::vector<DataBlock_t> subClusterPadList;
  std::vector<DataBlock_t> subClusterChargeList;
  std::vector<DataBlock_t> subClusterThetaEMFinal;
  std::vector<DataBlock_t> subClusterThetaFitList;
  std::vector<DataBlock_t> subClusterThetaExtra;

  // Cath groups
  int nCathGroups = 0;
  short* padToCathGrp = nullptr;
} InspectModel;
//

// PadProcessing
typedef struct dummyPad_t {
  // Data on Pixels
  const static int nPixelStorage = 8;
  std::vector<DataBlock_t> xyDxyQPixels[nPixelStorage];
} InspectPadProcessing_t;

extern "C" {
void cleanThetaList();
void cleanInspectModel();
// ??? Internal void copyInGroupList( const double *values, int N, int
// item_size, std::vector< DataBlock_t > &groupList);
// ??? void appendInThetaList( const double *values, int N, std::vector<
// DataBlock_t > &groupList);
void saveThetaEMInGroupList(const double* thetaEM, int K);
void saveThetaExtraInGroupList(const double* thetaExtra, int K);
void saveThetaFitInGroupList(const double* thetaFit, int K);
void collectTheta(double* theta, o2::mch::Groups_t* thetaToGroup, int K);
void savePadsOfSubCluster(const double* xyDxy, const double* q, int n);
void finalizeInspectModel();
int getNbrOfProjPads();
int getNbrOfPadsInGroups();
int getNbrOfThetaEMFinal();
int getNbrOfThetaExtra();
//
void saveProjectedPads(const o2::mch::Pads* pads, double* qProj);
void collectProjectedPads(double* xyDxy, double* chA, double* chB);
void savePadToCathGroup(const o2::mch::Groups_t* cath0Grp,
                        const o2::mch::PadIdx_t* mapCath1PadIdxToPadIdx, int nCath0,
                        const o2::mch::Groups_t* cath1Grp,
                        const o2::mch::PadIdx_t* mapCath0PadIdxToPadIdx, int nCath1);
int collectProjGroups(o2::mch::Groups_t* projPadToGrp);
void saveProjPadToGroups(o2::mch::Groups_t* projPadToGrp, int N);
void collectPadToCathGroup(o2::mch::Mask_t* padToMGrp, int nPads);
void collectPadsAndCharges(double* xyDxy, double* z, o2::mch::Groups_t* padToGroup,
                           int N);
// Unused ??? void collectLaplacian( double *laplacian, int N);
void collectResidual(double* residual, int N);
int getKThetaInit();
void collectThetaInit(double* thetai, int N);
void collectThetaEMFinal(double* thetaEM, int K);
void collectThetaExtra(double* thetaExtra, int K);
void cleanPixels();
int collectPixels(int which, int N, double* xyDxy, double* q);
void inspectSavePixels(int which, o2::mch::Pads& pixels);
int getNbrProjectedPads();
void setNbrProjectedPads(int n);
// Only used for old Clusterind analysis
// Perform a fit with fix coordinates
int f_ChargeIntegralMag(const gsl_vector* gslParams, void* data,
                        gsl_vector* residual);

void fitMathiesonMag(const double* xyDxDy, const double* q,
                     const o2::mch::Mask_t* cath, const o2::mch::Mask_t* sat, int chId,
                     double* thetaInit, int K, int N,
                     double* thetaFinal, double* khi2);
}
#endif // O2_MCH_INSPECTMODEL_H_