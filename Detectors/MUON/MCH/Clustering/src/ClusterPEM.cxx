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

/// \file ClusterPEM.h
/// \brief Definition of a class to reconstruct clusters with the original MLEM
/// algorithm
///
/// \author Gilles Grasseau, Subatech

#include <cstdio>
#include <stdexcept>

#include "InspectModel.h"
#include "MCHClustering/ClusterPEM.h"
#include "MCHClustering/ClusterConfig.h"
#include "MCHClustering/PadsPEM.h"
#include "mathUtil.h"
#include "mathieson.h"
#include "mathiesonFit.h"
#include "poissonEM.h"

namespace o2
{
namespace mch
{
// Fit parameters
// doProcess = verbose + (doJacobian << 2) + ( doKhi << 3) + (doStdErr << 4)
static const int processFitVerbose = 1 + (0 << 2) + (1 << 3) + (1 << 4);
static const int processFit = 0 + (0 << 2) + (1 << 3) + (1 << 4);

double epsilonGeometry = 1.0e-4;

ClusterPEM::ClusterPEM() = default;

ClusterPEM::ClusterPEM(Pads* pads0, Pads* pads1)
{
  int nPlanes = 0;
  singleCathPlaneID = -1;
  if (pads0 != nullptr) {
    pads[nPlanes] = pads0;
    nPlanes++;
  } else {
    singleCathPlaneID = 1;
  }
  if (pads1 != nullptr) {
    pads[nPlanes] = pads1;
    nPlanes++;
  } else {
    singleCathPlaneID = 0;
  }
  nbrOfCathodePlanes = nPlanes;
}

ClusterPEM::ClusterPEM(const double* x, const double* y, const double* dx,
                       const double* dy, const double* q, const short* cathodes,
                       const short* saturated, int chId, int nPads)
{

  chamberId = chId;
  nbrSaturated = vectorSumShort(saturated, nPads);

  int nbrCath1 = vectorSumShort(cathodes, nPads);
  int nbrCath0 = nPads - nbrCath1;

  // Build the pads for each cathode
  int nCath = 0;
  if (nbrCath0 != 0) {
    mapCathPadIdxToPadIdx[0] = new PadIdx_t[nbrCath0];
    pads[0] = new Pads(x, y, dx, dy, q, cathodes, saturated, 0, chId,
                       mapCathPadIdxToPadIdx[0], nPads);
    singleCathPlaneID = 0;
    nCath += 1;
  }
  if (nbrCath1 != 0) {
    mapCathPadIdxToPadIdx[1] = new PadIdx_t[nbrCath1];
    pads[1] = new Pads(x, y, dx, dy, q, cathodes, saturated, 1, chId,
                       mapCathPadIdxToPadIdx[1], nPads);
    singleCathPlaneID = 1;
    nCath += 1;
  }
  // Number of cathodes & alone cathode
  nbrOfCathodePlanes = nCath;
  if (nbrOfCathodePlanes == 2) {
    singleCathPlaneID = -1;
  }
  // ??? To remove if default Constructor
  // Projection
  projectedPads = nullptr;
  projNeighbors = nullptr;
  projPadToGrp = nullptr;
  nbrOfProjGroups = 0;
  // Groups
  cathGroup[0] = nullptr;
  cathGroup[1] = nullptr;
  nbrOfCathGroups = 0;
  // Geometry
  IInterJ = nullptr;
  JInterI = nullptr;
  mapKToIJ = nullptr;
  mapIJToK = nullptr;
  aloneIPads = nullptr;
  aloneJPads = nullptr;
  aloneKPads = nullptr;

  //
  if (ClusterConfig::processingLog >= ClusterConfig::info) {
    printf("-----------------------------\n");
    printf("Starting CLUSTER PROCESSING\n");
    printf("# cath0=%2d, cath1=%2d\n", nbrCath0, nbrCath1);
    printf("# sum Q0=%7.3g, sum Q1=%7.3g\n",
           (pads[0]) ? pads[0]->getTotalCharge() : 0,
           (pads[1]) ? pads[1]->getTotalCharge() : 0);
    printf("# singleCathPlaneID=%2d\n", singleCathPlaneID);
  }
}

// Extract the subcluster g
ClusterPEM::ClusterPEM(ClusterPEM& cluster, Groups_t g)
{
  chamberId = cluster.chamberId;
  int nbrCath[2] = {0, 0};
  nbrCath[0] = (cluster.pads[0]) ? cluster.pads[0]->getNbrOfPads() : 0;
  nbrCath[1] = (cluster.pads[1]) ? cluster.pads[1]->getNbrOfPads() : 0;
  //
  // Extract the pads of group g
  //
  Mask_t maskGrpCath0[nbrCath[0]];
  Mask_t maskGrpCath1[nbrCath[1]];
  Mask_t* maskGrpCath[2] = {maskGrpCath0, maskGrpCath1};
  int nbrCathPlanes_ = 0;
  int singleCathPlaneID_ = -1;
  for (int c = 0; c < 2; c++) {
    // Build the mask mapping the group g
    int nbrPads = 0;
    if (nbrCath[c]) {
      nbrPads = vectorBuildMaskEqualShort(cluster.cathGroup[c], g,
                                          cluster.pads[c]->getNbrOfPads(),
                                          maskGrpCath[c]);
    }
    // inv ??? printf("??? pads in the same group cathode c=%d \n", c);
    // inv ??? vectorPrintShort("??? maskGrpCath[c]",  maskGrpCath[c], cluster.pads[c]->getNbrOfPads());
    if (nbrPads != 0) {
      // Create the pads of the group g
      pads[c] = new Pads(*cluster.pads[c], maskGrpCath[c]);
      nbrCathPlanes_++;
      singleCathPlaneID_ = c;
    }
  }
  nbrOfCathodePlanes = nbrCathPlanes_;
  if (nbrCathPlanes_ != 2) {
    singleCathPlaneID = singleCathPlaneID_;
  }

  //
  // Extract the projected pads belonging to the group g
  //

  // Build the group-mask for proj pads
  Mask_t maskProjGrp[cluster.projectedPads->getNbrOfPads()];
  int nbrOfProjPadsInTheGroup = vectorBuildMaskEqualShort(
    cluster.projPadToGrp, g, cluster.projectedPads->getNbrOfPads(),
    maskProjGrp);
  projectedPads = new Pads(*cluster.projectedPads, maskProjGrp);
}

ClusterPEM::~ClusterPEM()
{
  for (int c = 0; c < 2; c++) {
    if (pads[c] == nullptr) {
      delete pads[c];
      pads[c] = nullptr;
    }
    deleteInt(mapCathPadIdxToPadIdx[c]);
    deleteShort(cathGroup[c]);
  }
  if (projectedPads == nullptr) {
    delete projectedPads;
    projectedPads = nullptr;
  }
  deleteInt(projNeighbors);
  deleteShort(projPadToGrp);
  deleteInt(IInterJ);
  deleteInt(JInterI);
  if (mapKToIJ != nullptr) {
    delete[] mapKToIJ;
    mapKToIJ = nullptr;
  }
  deleteInt(mapIJToK);
  deleteInt(aloneIPads);
  deleteInt(aloneJPads);
  deleteInt(aloneKPads);
}

int ClusterPEM::getIndexByRow(const char* matrix, PadIdx_t N, PadIdx_t M,
                              PadIdx_t* IIdx)
{
  int k = 0;
  // printf("N=%d, M=%d\n", N, M);
  for (PadIdx_t i = 0; i < N; i++) {
    for (PadIdx_t j = 0; j < M; j++) {
      if (matrix[i * M + j] == 1) {
        IIdx[k] = j;
        k++;
        // printf("k=%d,", k);
      }
    }
    // end of row/columns
    IIdx[k] = -1;
    k++;
  }
  // printf("\n final k=%d \n", k);
  return k;
}

int ClusterPEM::getIndexByColumns(const char* matrix, PadIdx_t N, PadIdx_t M,
                                  PadIdx_t* JIdx)
{
  int k = 0;
  for (PadIdx_t j = 0; j < M; j++) {
    for (PadIdx_t i = 0; i < N; i++) {
      if (matrix[i * M + j] == 1) {
        JIdx[k] = i;
        k++;
      }
    }
    // end of row/columns
    JIdx[k] = -1;
    k++;
  }
  return k;
}

int ClusterPEM::checkConsistencyMapKToIJ(const char* intersectionMatrix,
                                         const MapKToIJ_t* mapKToIJ,
                                         const PadIdx_t* mapIJToK,
                                         const PadIdx_t* aloneIPads,
                                         const PadIdx_t* aloneJPads, int N0,
                                         int N1, int nbrOfProjPads)
{
  MapKToIJ_t ij;
  int n = 0;
  int rc = 0;
  // Consistency with intersectionMatrix
  // and aloneI/JPads
  for (PadIdx_t k = 0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if ((ij.i >= 0) && (ij.j >= 0)) {
      if (intersectionMatrix[ij.i * N1 + ij.j] != 1) {
        printf("ERROR: no intersection %d %d %d\n", ij.i, ij.j,
               intersectionMatrix[ij.i * N1 + ij.j]);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      } else {
        n++;
      }
    } else if (ij.i < 0) {
      if (aloneJPads[ij.j] != k) {
        printf("ERROR: j-pad should be alone %d %d %d %d\n", ij.i, ij.j,
               aloneIPads[ij.j], k);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      }
    } else if (ij.j < 0) {
      if (aloneIPads[ij.i] != k) {
        printf("ERROR: i-pad should be alone %d %d %d %d\n", ij.i, ij.j,
               aloneJPads[ij.j], k);
        throw std::overflow_error("Divide by zero exception");
        rc = -1;
      }
    }
  }
  // TODO : Make a test with alone pads ???
  int sum = vectorSumChar(intersectionMatrix, N0 * N1);
  if (sum != n) {
    printf("ERROR: nbr of intersection differs %d %d \n", n, sum);
    throw std::overflow_error("Divide by zero exception");
    rc = -1;
  }

  for (int i = 0; i < N0; i++) {
    for (int j = 0; j < N1; j++) {
      int k = mapIJToK[i * N1 + j];
      if (k >= 0) {
        ij = mapKToIJ[k];
        if ((ij.i != i) || (ij.j != j)) {
          throw std::overflow_error(
            "checkConsistencyMapKToIJ: MapIJToK/MapKToIJ");
          printf("ij.i=%d, ij.j=%d, i=%d, j=%d \n", ij.i, ij.j, i, j);
        }
      }
    }
  }
  // Check mapKToIJ / mapIJToK
  for (PadIdx_t k = 0; k < nbrOfProjPads; k++) {
    ij = mapKToIJ[k];
    if (ij.i < 0) {
      if (aloneJPads[ij.j] != k) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error(
          "checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneJPads");
      }
    } else if (ij.j < 0) {
      if (aloneIPads[ij.i] != k) {
        printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
        throw std::overflow_error(
          "checkConsistencyMapKToIJ: MapKToIJ/MapIJToK aloneIPads");
      }
    } else if (mapIJToK[ij.i * N1 + ij.j] != k) {
      printf("i, j, k = %d, %d %d\n", ij.i, ij.j, k);
      throw std::overflow_error("checkConsistencyMapKToIJ: MapKToIJ/MapIJToK");
    }
  }

  return rc;
}

void ClusterPEM::computeProjectedPads(const Pads& pad0InfSup,
                                      const Pads& pad1InfSup,
                                      int maxNbrProjectedPads,
                                      PadIdx_t* aloneIPads, PadIdx_t* aloneJPads,
                                      PadIdx_t* aloneKPads, int includeAlonePads)
{
  // Use positive values of the intersectionMatrix
  // negative ones are single pads
  // Compute the new location of the projected pads (projected_xyDxy)
  // and the mapping mapKToIJ which maps k (projected pads)
  // to i, j (cathode pads)
  const double* x0Inf = pad0InfSup.getXInf();
  const double* y0Inf = pad0InfSup.getYInf();
  const double* x0Sup = pad0InfSup.getXSup();
  const double* y0Sup = pad0InfSup.getYSup();
  const double* x1Inf = pad1InfSup.getXInf();
  const double* y1Inf = pad1InfSup.getYInf();
  const double* x1Sup = pad1InfSup.getXSup();
  const double* y1Sup = pad1InfSup.getYSup();
  int N0 = pad0InfSup.getNbrOfPads();
  int N1 = pad1InfSup.getNbrOfPads();
  //
  // ??? Inv
  /*
  double *projX = projectedPads->getX();
  double *projY = projectedPads->getY();
  double *projDX = projectedPads->getDX();
  double *projDY = projectedPads->getDY();
  int nProjPads = projectedPads->getNbrOfPads
  */
  double* projX = new double[maxNbrProjectedPads];
  double* projY = new double[maxNbrProjectedPads];
  double* projDX = new double[maxNbrProjectedPads];
  double* projDY = new double[maxNbrProjectedPads];
  double l, r, b, t;
  int k = 0;
  PadIdx_t* ij_ptr = IInterJ;
  double countIInterJ, countJInterI;
  PadIdx_t i, j;
  for (i = 0; i < N0; i++) {
    // Nbr of j-pads intersepting  i-pad
    for (countIInterJ = 0; *ij_ptr != -1; countIInterJ++, ij_ptr++) {
      j = *ij_ptr;
      // Debug
      // printf("X[0/1]inf/sup %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, x0Inf[i],
      // x0Sup[i], x1Inf[j], x1Sup[j]);
      l = std::fmax(x0Inf[i], x1Inf[j]);
      r = std::fmin(x0Sup[i], x1Sup[j]);
      b = std::fmax(y0Inf[i], y1Inf[j]);
      t = std::fmin(y0Sup[i], y1Sup[j]);
      projX[k] = (l + r) * 0.5;
      projY[k] = (b + t) * 0.5;
      projDX[k] = (r - l) * 0.5;
      projDY[k] = (t - b) * 0.5;
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = j;
      mapIJToK[i * N1 + j] = k;
      // Debug
      if (ClusterConfig::padMappingLog >= ClusterConfig::debug) {
        printf("newpad %d %d %d %9.3g %9.3g %9.3g %9.3g\n", i, j, k, projX[k],
               projY[k], projDX[k], projDY[k]);
      }
      k++;
    }
    // Test if there is no intercepting pads with i-pad
    if ((countIInterJ == 0) && includeAlonePads) {
      l = x0Inf[i];
      r = x0Sup[i];
      b = y0Inf[i];
      t = y0Sup[i];
      projX[k] = (l + r) * 0.5;
      projY[k] = (b + t) * 0.5;
      projDX[k] = (r - l) * 0.5;
      projDY[k] = (t - b) * 0.5;
      // printf("newpad alone cath0 %d %d %9.3g %9.3g %9.3g %9.3g\n", i, k,
      // projX[k], projY[k], projDX[k], projDY[k]); Not used ???
      mapKToIJ[k].i = i;
      mapKToIJ[k].j = -1;
      aloneIPads[i] = k;
      aloneKPads[k] = i;
      k++;
    }
    // Row change
    ij_ptr++;
  }
  // Just add alone j-pads of cathode 1
  if (includeAlonePads) {
    ij_ptr = JInterI;
    for (PadIdx_t j = 0; j < N1; j++) {
      for (countJInterI = 0; *ij_ptr != -1; countJInterI++) {
        ij_ptr++;
      }
      if (countJInterI == 0) {
        l = x1Inf[j];
        r = x1Sup[j];
        b = y1Inf[j];
        t = y1Sup[j];
        projX[k] = (l + r) * 0.5;
        projY[k] = (b + t) * 0.5;
        projDX[k] = (r - l) * 0.5;
        projDY[k] = (t - b) * 0.5;
        // Debug
        // printf("newpad alone cath1 %d %d %9.3g %9.3g %9.3g %9.3g\n", j, k,
        // projX[k], projY[k], projDX[k], projDY[k]); newCh0[k] = ch0[i]; Not
        // used ???
        mapKToIJ[k].i = -1;
        mapKToIJ[k].j = j;
        aloneJPads[j] = k;
        aloneKPads[k] = j;
        k++;
      }
      // Skip -1, row/ col change
      ij_ptr++;
    }
  }
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printf("builProjectPads mapIJToK=%p, N0=%d N1=%d\\n", mapIJToK, N0, N1);
    for (int i = 0; i < N0; i++) {
      for (int j = 0; j < N1; j++) {
        if ((mapIJToK[i * N1 + j] != -1)) {
          printf(" %d inter %d\n", i, j);
        }
      }
    }
    vectorPrintInt("builProjectPads", aloneKPads, k);
  }
  projectedPads = new Pads(projX, projY, projDX, projDY, chamberId, k);
}

int ClusterPEM::buildProjectedGeometry(int includeSingleCathodePads)
{

  // Single cathode
  if (nbrOfCathodePlanes == 1) {
    // One Cathode case
    // Pad Projection is the cluster itself
    projectedPads = new Pads(*pads[singleCathPlaneID], Pads::xydxdyMode);
    projNeighbors = projectedPads->buildFirstNeighbors();
    return projectedPads->getNbrOfPads();
  }

  int N0 = pads[0]->getNbrOfPads();
  int N1 = pads[1]->getNbrOfPads();
  char intersectionMatrix[N0 * N1];
  vectorSetZeroChar(intersectionMatrix, N0 * N1);

  // Get the pad limits
  Pads padInfSup0(*pads[0], Pads::xyInfSupMode);
  Pads padInfSup1(*pads[1], Pads::xyInfSupMode);
  mapIJToK = new PadIdx_t[N0 * N1];
  vectorSetInt(mapIJToK, -1, N0 * N1);

  //
  // Build the intersection matrix
  // Looking for j pads, intercepting pad i
  //
  double xmin, xmax, ymin, ymax;
  PadIdx_t xInter, yInter;
  const double* x0Inf = padInfSup0.getXInf();
  const double* x0Sup = padInfSup0.getXSup();
  const double* y0Inf = padInfSup0.getYInf();
  const double* y0Sup = padInfSup0.getYSup();
  const double* x1Inf = padInfSup1.getXInf();
  const double* x1Sup = padInfSup1.getXSup();
  const double* y1Inf = padInfSup1.getYInf();
  const double* y1Sup = padInfSup1.getYSup();
  for (PadIdx_t i = 0; i < N0; i++) {
    for (PadIdx_t j = 0; j < N1; j++) {
      xmin = std::fmax(x0Inf[i], x1Inf[j]);
      xmax = std::fmin(x0Sup[i], x1Sup[j]);
      xInter = (xmin <= (xmax - epsilonGeometry));
      if (xInter) {
        ymin = std::fmax(y0Inf[i], y1Inf[j]);
        ymax = std::fmin(y0Sup[i], y1Sup[j]);
        yInter = (ymin <= (ymax - epsilonGeometry));
        intersectionMatrix[i * N1 + j] = yInter;
        // printf("inter i=%3d, j=%3d,  x0=%8.5f y0=%8.5f, x1=%8.5f y1=%8.5f\n",
        // i, j, pads[0]->x[i], pads[0]->y[i], pads[1]->x[i], pads[1]->y[i]);
      }
    }
  }
  //
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printMatrixChar("  Intersection Matrix", intersectionMatrix, N0, N1);
  }
  //
  // Compute the max number of projected pads to make
  // memory allocations
  //
  int maxNbrOfProjPads = vectorSumChar(intersectionMatrix, N0 * N1);
  int nbrOfSinglePads = 0;
  if (includeSingleCathodePads) {
    // Add alone cath0-pads
    for (PadIdx_t i = 0; i < N0; i++) {
      if (vectorSumRowChar(&intersectionMatrix[i * N1], N0, N1) == 0) {
        nbrOfSinglePads++;
      }
    }
    // Add alone cath1-pads
    for (PadIdx_t j = 0; j < N1; j++) {
      if (vectorSumColumnChar(&intersectionMatrix[j], N0, N1) == 0) {
        nbrOfSinglePads++;
      }
    }
  }
  // Add alone pas and row/column separators
  maxNbrOfProjPads += nbrOfSinglePads + fmax(N0, N1);
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printf("  maxNbrOfProjPads %d\n", maxNbrOfProjPads);
  }
  //
  //
  // Intersection Matrix Sparse representation
  //
  /// To Save ???
  IInterJ = new PadIdx_t[maxNbrOfProjPads];
  JInterI = new PadIdx_t[maxNbrOfProjPads];
  int checkr = getIndexByRow(intersectionMatrix, N0, N1, IInterJ);
  int checkc = getIndexByColumns(intersectionMatrix, N0, N1, JInterI);
  if (ClusterConfig::padMappingCheck) {
    if ((checkr > maxNbrOfProjPads) || (checkc > maxNbrOfProjPads)) {
      printf(
        "Allocation pb for  IInterJ or JInterI: allocated=%d, needed for "
        "row=%d, for col=%d \n",
        maxNbrOfProjPads, checkr, checkc);
      throw std::overflow_error("Allocation pb for  IInterJ or JInterI");
    }
  }
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printInterMap("  IInterJ", IInterJ, N0);
    printInterMap("  JInterI", JInterI, N1);
  }

  //
  // Remaining allocation
  //
  aloneIPads = new PadIdx_t[N0];
  aloneJPads = new PadIdx_t[N1];
  // PadIdx_t *aloneKPads = new PadIdx_t[N0*N1];
  mapKToIJ = new MapKToIJ_t[maxNbrOfProjPads];
  aloneKPads = new PadIdx_t[maxNbrOfProjPads];
  vectorSetInt(aloneIPads, -1, N0);
  vectorSetInt(aloneJPads, -1, N1);
  vectorSetInt(aloneKPads, -1, maxNbrOfProjPads);

  //
  // Build the projected pads
  //
  computeProjectedPads(padInfSup0, padInfSup1, maxNbrOfProjPads, aloneIPads,
                       aloneJPads, aloneKPads, includeSingleCathodePads);

  if (ClusterConfig::padMappingCheck) {
    checkConsistencyMapKToIJ(intersectionMatrix, mapKToIJ, mapIJToK, aloneIPads,
                             aloneJPads, N0, N1, projectedPads->getNbrOfPads());
  }
  //
  // Get the isolated new pads
  // (they have no neighboring)
  //
  int thereAreIsolatedPads = 0;
  projNeighbors = projectedPads->buildFirstNeighbors();
  // Pads::printPads("Projected Pads:", *projectedPads);
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printf("  Neighbors of the projected geometry\n");
    Pads::printNeighbors(projNeighbors, projectedPads->getNbrOfPads());
  }
  int nbrOfProjPads = projectedPads->getNbrOfPads();
  MapKToIJ_t ij;
  for (PadIdx_t k = 0; k < nbrOfProjPads; k++) {
    if (getTheFirstNeighborOf(projNeighbors, k) == -1) {
      // pad k is isolated
      thereAreIsolatedPads = 1;
      ij = mapKToIJ[k];
      if ((ij.i >= 0) && (ij.j >= 0)) {
        if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
          printf(" Isolated pad: nul intersection i,j = %d %d\n", ij.i, ij.j);
        }
        intersectionMatrix[ij.i * N1 + ij.j] = 0;
      } else {
        throw std::overflow_error("I/j negative (alone pad)");
      }
    }
  }
  if ((ClusterConfig::padMappingLog >= ClusterConfig::detail) && thereAreIsolatedPads) {
    printf("There are isolated pads %d\n", thereAreIsolatedPads);
  }
  //
  if (thereAreIsolatedPads == 1) {
    // Recompute all
    // Why ???
    getIndexByRow(intersectionMatrix, N0, N1, IInterJ);
    getIndexByColumns(intersectionMatrix, N0, N1, JInterI);
    //
    // Build the new pads
    //
    delete projectedPads;
    computeProjectedPads(padInfSup0, padInfSup1, maxNbrOfProjPads, aloneIPads,
                         aloneJPads, aloneKPads, includeSingleCathodePads);
    delete[] projNeighbors;
    projNeighbors = projectedPads->buildFirstNeighbors();
  }
  return projectedPads->getNbrOfPads();
}

double* ClusterPEM::projectChargeOnProjGeometry(int includeAlonePads)
{

  double* qProj;
  if (nbrOfCathodePlanes == 1) {
    Pads* sPads = pads[singleCathPlaneID];
    qProj = new double[sPads->getNbrOfPads()];
    vectorCopy(sPads->getCharges(), sPads->getNbrOfPads(), qProj);
    return qProj;
  }
  int nbrOfProjPads = projectedPads->getNbrOfPads();
  const double* ch0 = pads[0]->getCharges();
  const double* ch1 = pads[1]->getCharges();
  //
  // Computing charges of the projected pads
  // Ch0 part
  //
  double minProj[nbrOfProjPads];
  double maxProj[nbrOfProjPads];
  double projCh0[nbrOfProjPads];
  double projCh1[nbrOfProjPads];
  int N0 = pads[0]->getNbrOfPads();
  int N1 = pads[1]->getNbrOfPads();
  PadIdx_t k = 0;
  double sumCh1ByRow;
  PadIdx_t* ij_ptr = IInterJ;
  PadIdx_t* rowStart;
  for (PadIdx_t i = 0; i < N0; i++) {
    // Save the starting index of the begining of the row
    rowStart = ij_ptr;
    // sum of charge with intercepting j-pad
    for (sumCh1ByRow = 0.0; *ij_ptr != -1; ij_ptr++) {
      sumCh1ByRow += ch1[*ij_ptr];
    }
    double ch0_i = ch0[i];
    if (sumCh1ByRow != 0.0) {
      double cst = ch0[i] / sumCh1ByRow;
      for (ij_ptr = rowStart; *ij_ptr != -1; ij_ptr++) {
        projCh0[k] = ch1[*ij_ptr] * cst;
        minProj[k] = fmin(ch1[*ij_ptr], ch0_i);
        maxProj[k] = fmax(ch1[*ij_ptr], ch0_i);
        // Debug
        // printf(" i=%d, j=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", i,
        // *ij_ptr, k, sumCh1ByRow, projCh0[k]);
        k++;
      }
    } else if (includeAlonePads) {
      // Alone i-pad
      projCh0[k] = ch0[i];
      minProj[k] = ch0[i];
      maxProj[k] = ch0[i];
      k++;
    }
    // Move on a new row
    ij_ptr++;
  }
  // Just add alone pads of cathode 1
  if (includeAlonePads) {
    for (PadIdx_t j = 0; j < N1; j++) {
      k = aloneJPads[j];
      if (k >= 0) {
        projCh0[k] = ch1[j];
        minProj[k] = ch1[j];
        maxProj[k] = ch1[j];
      }
    }
  }
  //
  // Computing charges of the projected pads
  // Ch1 part
  //
  k = 0;
  double sumCh0ByCol;
  ij_ptr = JInterI;
  PadIdx_t* colStart;
  for (PadIdx_t j = 0; j < N1; j++) {
    // Save the starting index of the beginnig of the column
    colStart = ij_ptr;
    // sum of charge intercepting i-pad
    for (sumCh0ByCol = 0.0; *ij_ptr != -1; ij_ptr++) {
      sumCh0ByCol += ch0[*ij_ptr];
    }
    if (sumCh0ByCol != 0.0) {
      double cst = ch1[j] / sumCh0ByCol;
      for (ij_ptr = colStart; *ij_ptr != -1; ij_ptr++) {
        PadIdx_t i = *ij_ptr;
        k = mapIJToK[i * N1 + j];
        projCh1[k] = ch0[i] * cst;
        // Debug
        // printf(" j=%d, i=%d, k=%d, sumCh0ByCol = %g, projCh1[k]= %g \n", j,
        // i, k, sumCh0ByCol, projCh1[k]);
      }
    } else if (includeAlonePads) {
      // Alone j-pad
      k = aloneJPads[j];
      if (ClusterConfig::padMappingCheck && (k < 0)) {
        printf("ERROR: Alone j-pad with negative index j=%d\n", j);
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
      }
      projCh1[k] = ch1[j];
    }
    ij_ptr++;
  }

  // Just add alone pads of cathode 0
  if (includeAlonePads) {
    ij_ptr = IInterJ;
    for (PadIdx_t i = 0; i < N0; i++) {
      k = aloneIPads[i];
      if (k >= 0) {
        // printf("Alone i-pad  i=%d, k=%d\n", i, k);
        projCh1[k] = ch0[i];
      }
    }
  }
  // Charge Result
  // Do the mean
  qProj = new double[nbrOfProjPads];
  vectorAddVector(projCh0, 1.0, projCh1, nbrOfProjPads, qProj);
  vectorMultScalar(qProj, 0.5, nbrOfProjPads, qProj);
  return qProj;
}

int ClusterPEM::buildGroupOfPads()
{

  // Having one cathode plane (cathO, cath1 or projected cathodes)
  // Extract the sub-clusters
  int nProjPads = projectedPads->getNbrOfPads();
  projPadToGrp = new Groups_t[nProjPads];
  // Set to no Group (zero)
  vectorSetShort(projPadToGrp, 0, nProjPads);

  //
  //  Part I - Extract groups from projection plane
  //
  int nCathGroups = 0;
  int nGroups = 0;
  int nbrCath0 = (pads[0]) ? pads[0]->getNbrOfPads() : 0;
  int nbrCath1 = (pads[1]) ? pads[1]->getNbrOfPads() : 0;

  if (ClusterConfig::groupsLog >= ClusterConfig::info || ClusterConfig::processingLog >= ClusterConfig::info) {
    printf("\n");
    printf("[buildGroupOfPads] Group processing\n");
    printf("----------------\n");
  }
  //
  // Build the "proj-groups"
  // The pads which have no recovery with the other cathode plane
  // are not considered. They are named 'single-pads'
  nbrOfProjGroups = getConnectedComponentsOfProjPadsWOSinglePads();

  if (ClusterConfig::inspectModel >= ClusterConfig::active) {
    saveProjPadToGroups(projPadToGrp, projectedPads->getNbrOfPads());
  }

  // Single cathode case
  // The cath-group & proj-groups are the same
  Groups_t grpToCathGrp[nbrOfProjGroups + 1];
  if (nbrOfCathodePlanes == 1) {
    // Copy the projPadGroup to cathGroup[0/1]
    int nCathPads = pads[singleCathPlaneID]->getNbrOfPads();
    cathGroup[singleCathPlaneID] = new Groups_t[nCathPads];
    vectorCopyShort(projPadToGrp, nCathPads, cathGroup[singleCathPlaneID]);
    nCathGroups = nbrOfProjGroups;
    nGroups = nCathGroups;
    // Identity mapping
    for (int g = 0; g < (nbrOfProjGroups + 1); g++) {
      grpToCathGrp[g] = g;
    }
    // Add the pads (not present in the projected plane)
    int nNewGroups = pads[singleCathPlaneID]->addIsolatedPadInGroups(
      cathGroup[singleCathPlaneID], grpToCathGrp, nGroups);
    // ??? Check if some time
    if (nNewGroups > 0) {
      throw std::overflow_error("New group with one cathode plane ?????");
    }
    nGroups += nNewGroups;
    //
  } else {
    //
    //  Part I - Extract groups from cathode planes
    //
    // 2 cathode planes & projected cathode plane
    // Init. the new groups 'cath-groups'
    int nCath0 = pads[0]->getNbrOfPads();
    int nCath1 = pads[1]->getNbrOfPads();
    cathGroup[0] = new Groups_t[nCath0];
    cathGroup[1] = new Groups_t[nCath1];
    vectorSetZeroShort(cathGroup[0], nCath0);
    vectorSetZeroShort(cathGroup[1], nCath1);
    if (ClusterConfig::groupsLog >= ClusterConfig::info) {
      printf("> Projected Groups nbrOfProjGroups=%d\n", nbrOfProjGroups);
      vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    }

    // Build cathode groups (merge the projected groups)
    //
    int nPads = pads[0]->getNbrOfPads() + pads[1]->getNbrOfPads();
    // Propagate proj-groups on the cathode pads
    nGroups = assignPadsToGroupFromProj(nbrOfProjGroups);
    // nGroups = assignGroupToCathPads( );
    if (ClusterConfig::groupsLog >= ClusterConfig::info) {
      printf("> Groups after cathodes propagation nCathGroups=%d\n", nGroups);
    }

    // Compute the max allocation for the new cath-groups
    // Have to add single pads
    int nbrSinglePads = 0;
    for (int c = 0; c < 2; c++) {
      for (int p = 0; p < pads[c]->getNbrOfPads(); p++) {
        if (cathGroup[c][p] == 0) {
          nbrSinglePads += 1;
        }
      }
    }
    int nbrMaxGroups = nGroups + nbrSinglePads;
    // mapGrpToGrp is use to map oldGroups (proj-groups) to
    // newGroups (cath-groups) when
    // occurs
    Mask_t mapGrpToGrp[nbrMaxGroups + 1];
    for (int g = 0; g < (nbrMaxGroups + 1); g++) {
      mapGrpToGrp[g] = g;
    }

    //
    // Add single pads of cath-0 and modyfy the groups
    //
    int nNewGrpCath0 =
      pads[0]->addIsolatedPadInGroups(cathGroup[0], mapGrpToGrp, nGroups);
    nGroups += nNewGrpCath0;

    // Apply the new Groups on cath1
    for (int p = 0; p < nbrCath1; p++) {
      cathGroup[1][p] = mapGrpToGrp[cathGroup[1][p]];
    }
    // ... and on proj-pads
    for (int p = 0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[projPadToGrp[p]];
    }
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("> addIsolatedPadInGroups in cath-0 nNewGroups =%d\n", nGroups);
      vectorPrintShort("  mapGrpToGrp", mapGrpToGrp, nGroups + 1);
    }

    //
    // Do the same on cath1
    // Add single pads of cath-1 and modify the groups
    //
    int nNewGrpCath1 =
      pads[1]->addIsolatedPadInGroups(cathGroup[1], mapGrpToGrp, nGroups);
    nGroups += nNewGrpCath1;
    // Apply the new Groups on cath1
    for (int p = 0; p < nbrCath0; p++) {
      cathGroup[0][p] = mapGrpToGrp[cathGroup[0][p]];
    }
    // ... and on proj-pads
    for (int p = 0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[projPadToGrp[p]];
    }
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("> addIsolatedPadInGroups in cath-1 nNewGroups =%d\n", nGroups);
      vectorPrintShort("  mapGrpToGrp", mapGrpToGrp, nGroups + 1);
    }
    // Remove low charged groups
    removeLowChargedGroups(nGroups);

    // Some groups may be merged, others groups may diseappear
    // So the final groups must be renumbered
    int nNewGroups = renumberGroups(mapGrpToGrp, nGroups);
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("> Groups after renumbering nGroups=%d\n", nGroups);
      vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
      printf("  nNewGrpCath0=%d, nNewGrpCath1=%d, nGroups=%d\n", nNewGrpCath0,
             nNewGrpCath1, nGroups);
      vectorPrintShort("  cath0ToGrp  ", cathGroup[0], nbrCath0);
      vectorPrintShort("  cath1ToGrp  ", cathGroup[1], nbrCath1);
      vectorPrintShort("   mapGrpToGrp ", mapGrpToGrp, nNewGroups);
    }
    // Apply this renumbering on projection-pads
    for (int p = 0; p < nProjPads; p++) {
      projPadToGrp[p] = mapGrpToGrp[projPadToGrp[p]];
    }

    nGroups = nNewGroups;
    // Propagate the cath-groups to the projected pads
    updateProjectionGroups();
  }

  if (ClusterConfig::groupsLog >= ClusterConfig::info || ClusterConfig::processingLog >= ClusterConfig::info) {
    printf("  > Final Groups %d\n", nGroups);
    vectorPrintShort("  cathToGrp[0]", cathGroup[0], nbrCath0);
    vectorPrintShort("  cathToGrp[1]", cathGroup[1], nbrCath1);
  }
  return nGroups;
}

int ClusterPEM::getConnectedComponentsOfProjPadsWOSinglePads()
{
  // Class from neighbors list of projected pads, the pads in groups (connected
  // components) projPadToGrp is set to the group Id of the pad. If the group Id
  // is zero, the the pad is unclassified Return the number of groups
  int N = projectedPads->getNbrOfPads();
  projPadToGrp = new Groups_t[N];
  PadIdx_t* neigh = projNeighbors;
  PadIdx_t neighToDo[N];
  vectorSetZeroShort(projPadToGrp, N);
  // Nbr of pads alrready proccessed
  int nbrOfPadSetInGrp = 0;
  // Last projPadToGrp to process
  short* curPadGrp = projPadToGrp;
  short currentGrpId = 0;
  //
  int i, j, k;
  // printNeighbors();

  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    printf(
      "> Extract connected components "
      "[getConnectedComponentsOfProjPadsWOIsolatedPads]\n");
  }
  while (nbrOfPadSetInGrp < N) {
    // Seeking the first unclassed pad (projPadToGrp[k]=0)
    for (; (curPadGrp < &projPadToGrp[N]) && *curPadGrp != 0;) {
      curPadGrp++;
    }
    k = curPadGrp - projPadToGrp;
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("    k=%d, nbrOfPadSetInGrp g=%d: n=%d\n", k, currentGrpId,
             nbrOfPadSetInGrp);
    }
    //
    // New group for pad k - then search all neighbours of k
    // aloneKPads = 0 if only one cathode
    if (aloneKPads && (aloneKPads[k] != -1)) {
      // Alone Pad no group at the moment
      if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
        printf("    isolated pad %d\n", k);
      }
      projPadToGrp[k] = -1;
      nbrOfPadSetInGrp++;
      continue;
    }
    currentGrpId++;
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("    New Grp, pad k=%d in new grp=%d\n", k, currentGrpId);
    }
    projPadToGrp[k] = currentGrpId;
    nbrOfPadSetInGrp++;
    PadIdx_t startIdx = 0, endIdx = 1;
    neighToDo[startIdx] = k;
    // Labels k neighbors
    // Propagation of the group in all neighbour list
    for (; startIdx < endIdx; startIdx++) {
      i = neighToDo[startIdx];
      if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
        printf("    propagate grp to neighbours of i=%d ", i);
      }
      //
      // Scan i neighbors
      for (PadIdx_t* neigh_ptr = getTheFirtsNeighborOf(neigh, i);
           *neigh_ptr != -1; neigh_ptr++) {
        j = *neigh_ptr;
        // printf("    neigh j %d\n, \n", j);
        if (projPadToGrp[j] == 0) {
          // Add the neighbors in the currentgroup
          //
          // aloneKPads = 0 if only one cathode
          if (aloneKPads && (aloneKPads[j] != -1)) {
            if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
              printf("    isolated pad %d, ", j);
            }
            projPadToGrp[j] = -1;
            nbrOfPadSetInGrp++;
            continue;
          }
          if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
            printf("%d, ", j);
          }
          projPadToGrp[j] = currentGrpId;
          nbrOfPadSetInGrp++;
          // Append in the neighbor list to search
          neighToDo[endIdx] = j;
          endIdx++;
        }
      }
      if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
        printf("\n");
      }
    }
    // printf("make groups grpId=%d, nbrOfPadSetInGrp=%d\n", currentGrpId,
    // nbrOfPadSetInGrp);
  }
  for (int k = 0; k < N; k++) {
    if (projPadToGrp[k] == -1) {
      projPadToGrp[k] = 0;
    }
  }
  // return tne number of Grp
  return currentGrpId;
}

///??????????????????
/*
void Cluster::assignSingleCathPadsToGroup( short *padGroup, int nPads, int nGrp,
int nCath0, int nCath1) { Groups_t cath0ToGrpFromProj[nCath0]; Groups_t
cath1ToGrpFromProj[nCath1]; cath1ToGrpFromProj = 0; if ( nCath0 != 0) {
    cath0ToGrpFromProj = new short[nCath0];
    vectorCopyShort( padGroup, nCath0, cath0ToGrpFromProj);
  } else {
    cath1ToGrpFromProj = new short[nCath1];
    vectorCopyShort( padGroup, nCath1, cath1ToGrpFromProj);
  }
  vectorSetShort( wellSplitGroup, 1, nGrp+1);
}
*/

// Assign a group to the original pads
// Update the pad group and projected-pads group
int ClusterPEM::assignPadsToGroupFromProj(int nGrp)
{
  // Matrix for the mapping olg-group -> new group
  short matGrpGrp[(nGrp + 1) * (nGrp + 1)];
  vectorSetZeroShort(matGrpGrp, (nGrp + 1) * (nGrp + 1));
  //
  PadIdx_t i, j;
  short g, prevGroup;
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    printf("> Assign cath-grp from proj-grp [AssignPadsToGroupFromProj]\n");
  }
  // Expand the projected Groups
  // 'projPadToGrp' to the pad groups 'padToGrp'
  // If there are conflicts, fuse the groups
  // Build the Group-to-Group matrix matGrpGrp
  // which describe how to fuse Groups
  // with the projected Groups
  // projPadToGrp
  int nProjPads = projectedPads->getNbrOfPads();
  for (int k = 0; k < nProjPads; k++) {
    g = projPadToGrp[k];
    // Give the indexes of overlapping pad k
    i = mapKToIJ[k].i;
    j = mapKToIJ[k].j;
    //
    // Cathode 0
    //
    if (i >= 0) {
      // Remark: if i is an alone pad (j<0)
      // i is processed as well
      //
      prevGroup = cathGroup[0][i];
      if ((prevGroup == 0) || (prevGroup == g)) {
        // Case: no group before or same group
        //
        cathGroup[0][i] = g;
        matGrpGrp[g * (nGrp + 1) + g] = 1;
      } else {
        // Already a grp (Conflict)
        // Group to fuse
        // Store the grp into grp matrix
        cathGroup[0][i] = g;
        matGrpGrp[g * (nGrp + 1) + prevGroup] = 1;
        matGrpGrp[prevGroup * (nGrp + 1) + g] = 1;
      }
    }
    //
    // Cathode 1
    //
    if ((j >= 0)) {
      // Remark: if j is an alone pad (j<0)
      // j is processed as well
      //
      prevGroup = cathGroup[1][j];

      if ((prevGroup == 0) || (prevGroup == g)) {
        // No group before
        cathGroup[1][j] = g;
        matGrpGrp[g * (nGrp + 1) + g] = 1;
      } else {
        // Already a Group (Conflict)
        cathGroup[1][j] = g;
        matGrpGrp[g * (nGrp + 1) + prevGroup] = 1;
        matGrpGrp[prevGroup * (nGrp + 1) + g] = 1;
      }
    }
  }
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    printMatrixShort("  Group/Group matrix", matGrpGrp, nGrp + 1, nGrp + 1);
    vectorPrintShort("  cathToGrp[0]", cathGroup[0], pads[0]->getNbrOfPads());
    vectorPrintShort("  cathToGrp[1]", cathGroup[1], pads[1]->getNbrOfPads());
  }
  //
  // Merge the groups (build the mapping grpToMergedGrp)
  //
  Groups_t grpToMergedGrp[nGrp + 1]; // Mapping old groups to new merged groups
  vectorSetZeroShort(grpToMergedGrp, nGrp + 1);
  //
  int iGroup = 1; // Describe the current group
  int curGroup;   // Describe the mapping grpToMergedGrp[iGroup]
  while (iGroup < (nGrp + 1)) {
    // Define the new group to process
    if (grpToMergedGrp[iGroup] == 0) {
      // No group before
      grpToMergedGrp[iGroup] = iGroup;
    }
    curGroup = grpToMergedGrp[iGroup];
    // printf( "  current iGroup=%d -> grp=%d \n", iGroup, curGroup);
    // Look for other groups in matGrpGrp
    int ishift = iGroup * (nGrp + 1);
    // Check if there is an overlaping group
    for (int j = iGroup + 1; j < (nGrp + 1); j++) {
      if (matGrpGrp[ishift + j]) {
        // Merge the groups with the current one
        if (grpToMergedGrp[j] == 0) {
          // No group assign before, merge the groups with the current one
          // printf( "    newg merge grp=%d -> grp=%d\n", j, curGroup);
          grpToMergedGrp[j] = curGroup;
        } else {
          // A group is already assigned,
          // Merge the 2 groups curGroup and grpToMergedGrp[j]
          // printf( "    oldg merge grp=%d -> grp=%d\n", curGroup,
          // grpToMergedGrp[j]); Remark : curGroup < j Fuse and propagate the
          // groups
          grpToMergedGrp[curGroup] = grpToMergedGrp[j];
          for (int g = 1; g < nGrp + 1; g++) {
            if (grpToMergedGrp[g] == curGroup) {
              grpToMergedGrp[g] = grpToMergedGrp[j];
            }
          }
        }
      }
    }
    iGroup++;
  }

  // Perform the mapping group -> mergedGroups
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    vectorPrintShort("  Mapping grpToMergedGrp", grpToMergedGrp, nGrp + 1);
  }
  //
  // Renumber the fused groups
  //
  int newGroupID = 0;
  Mask_t map[nGrp + 1];
  vectorSetZeroShort(map, (nGrp + 1));
  for (int g = 1; g < (nGrp + 1); g++) {
    int gm = grpToMergedGrp[g];
    if (map[gm] == 0) {
      newGroupID++;
      map[gm] = newGroupID;
    }
  }
  // Apply the renumbering
  for (int g = 1; g < (nGrp + 1); g++) {
    grpToMergedGrp[g] = map[grpToMergedGrp[g]];
  }

  // Perform the mapping grpToMergedGrp to the cath-groups
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    vectorPrintShort("  Mapping renumbered grpToMergedGrp", grpToMergedGrp,
                     nGrp + 1);
  }
  for (int c = 0; c < 2; c++) {
    for (int p = 0; p < pads[c]->getNbrOfPads(); p++) {
      // ??? Why abs() ... explain
      cathGroup[c][p] = grpToMergedGrp[std::abs(cathGroup[c][p])];
    }
  }

  if (ClusterConfig::groupsCheck) {
    for (int c = 0; c < 2; c++) {
      for (int p = 0; p < pads[c]->getNbrOfPads(); p++) {
        if (ClusterConfig::groupsLog >= ClusterConfig::info && cathGroup[c][p] == 0) {
          printf("  [assignPadsToGroupFromProj] pad %d with no group\n", p);
        }
      }
    }
  }

  // Perform the mapping grpToMergedGrp to the projected pads
  vectorMapShort(projPadToGrp, grpToMergedGrp, nProjPads);

  //
  return newGroupID;
}

// Add boundary pads with q charge equal 0
void ClusterPEM::addBoundaryPads()
{
  for (int c = 0; c < 2; c++) {
    if (pads[c]) {
      Pads* bPads = pads[c]->addBoundaryPads();
      delete pads[c];
      pads[c] = bPads;
    }
  }
}

// Propagate the proj-groups to the cath-pads
// Not used
int ClusterPEM::assignGroupToCathPads()
{
  //
  // From the cathode group found with the projection,
  int nCath0 = (pads[0]) ? pads[0]->getNbrOfPads() : 0;
  int nCath1 = (pads[1]) ? pads[1]->getNbrOfPads() : 0;
  int nGrp = nbrOfProjGroups;
  // Groups obtain with the projection
  /*
  Groups_t cath0ToGrpFromProj[nCath0];
  Groups_t cath1ToGrpFromProj[nCath1];
  vectorSetZeroShort( cath0ToGrpFromProj, nCath0);
  vectorSetZeroShort( cath1ToGrpFromProj, nCath1);
  */
  Groups_t* cath0ToGrpFromProj = cathGroup[0];
  Groups_t* cath1ToGrpFromProj = cathGroup[1];
  vectorSetZeroShort(cathGroup[0], nCath0);
  vectorSetZeroShort(cathGroup[1], nCath1);
  // Mapping proj-groups to cath-groups
  Groups_t projGrpToCathGrp[nGrp + 1];
  vectorSetZeroShort(projGrpToCathGrp, nGrp + 1);
  int nCathGrp = 0;
  //
  if (ClusterConfig::groupsLog >= ClusterConfig::info) {
    printf("  [assignGroupToCathPads]\n");
  }
  //
  PadIdx_t i, j;
  short g, prevGroup0, prevGroup1;
  if (nbrOfCathodePlanes == 1) {
    // Single cathode plane
    vectorCopyShort(projPadToGrp, pads[singleCathPlaneID]->getNbrOfPads(),
                    cathGroup[singleCathPlaneID]);
    return nGrp;
  }
  int nProjPads = projectedPads->getNbrOfPads();
  for (int k = 0; k < nProjPads; k++) {
    // Group of the projection pad k
    g = projPadToGrp[k];
    // Intersection indexes of the 2 cath
    i = mapKToIJ[k].i;
    j = mapKToIJ[k].j;
    if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
      printf("map k=%d g=%d to i=%d/%d, j=%d/%d\n", k, g, i, nCath0, j, nCath1);
    }
    //
    // Cathode 0
    //
    if ((i >= 0) && (nCath0 != 0)) {
      // if the pad has already been set
      prevGroup0 = cath0ToGrpFromProj[i];
      if (prevGroup0 == 0) {
        if ((projGrpToCathGrp[g] == 0) && (g != 0)) {
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        }
        cath0ToGrpFromProj[i] = projGrpToCathGrp[g];

      } else if (prevGroup0 != projGrpToCathGrp[g]) {
        // The previous cath-group of pad i differs from
        // those of g:
        // -> Fuse the 2 cath-groups g and prevGroup0
        printf(">>> Fuse group g=%d prevGrp0=%d, projGrpToCathGrp[g]=%d\n", g,
               prevGroup0, projGrpToCathGrp[g]);
        // projGrpToCathGrp[g] = prevGroup0;
        //
        Groups_t minGroup = (projGrpToCathGrp[g] != 0)
                              ? std::min(projGrpToCathGrp[g], prevGroup0)
                              : prevGroup0;
        projGrpToCathGrp[g] = minGroup;
        // projGrpToCathGrp[prevGroup0] = minGroup;
      }
    }
    //
    // Cathode 1
    //
    if ((j >= 0) && (nCath1 != 0)) {
      prevGroup1 = cath1ToGrpFromProj[j];
      if (prevGroup1 == 0) {
        if ((projGrpToCathGrp[g] == 0) && (g != 0)) {
          nCathGrp++;
          projGrpToCathGrp[g] = nCathGrp;
        }
        cath1ToGrpFromProj[j] = projGrpToCathGrp[g];
      } else if (prevGroup1 != projGrpToCathGrp[g]) {
        printf(">>> Fuse group g=%d prevGrp1=%d, projGrpToCathGrp[g]=%d\n", g,
               prevGroup1, projGrpToCathGrp[g]);
        // projGrpToCathGrp[g] = prevGroup1;
        // Groups_t minGroup = std::min( projGrpToCathGrp[g], prevGroup1);
        Groups_t minGroup = (projGrpToCathGrp[g] != 0)
                              ? std::min(projGrpToCathGrp[g], prevGroup1)
                              : prevGroup1;
        projGrpToCathGrp[g] = minGroup;
        // projGrpToCathGrp[prevGroup1] = minGroup;
      }
    }
  }

  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    printf("  [assignGroupToCathPads] before renumbering nCathGrp=%d\n", nCathGrp);
    vectorPrintShort("    cath0ToGrpFromProj", cath0ToGrpFromProj, nCath0);
    vectorPrintShort("    cath1ToGrpFromProj", cath1ToGrpFromProj, nCath1);
    vectorPrintShort("    projGrpToCathGrp", projGrpToCathGrp, nGrp + 1);
  }
  //
  // Renumering cathodes groups
  // Desactivated ???
  int nNewGrp = renumberGroupsFromMap(projGrpToCathGrp, nGrp);
  // Test if renumbering is necessary
  if (nNewGrp != nGrp) {
    // Perform the renumbering
    vectorMapShort(cath0ToGrpFromProj, projGrpToCathGrp, nCath0);
    vectorMapShort(cath1ToGrpFromProj, projGrpToCathGrp, nCath1);
  }
  //
  //
  // int nNewGrp = nGrp;
  //
  // Set/update the cath/proj Groups
  // vectorCopyShort( cath0ToGrpFromProj, nCath0, cathGroup[0]);
  // vectorCopyShort( cath1ToGrpFromProj, nCath1, cathGroup[1]);
  //
  for (i = 0; i < nProjPads; i++) {
    projPadToGrp[i] = projGrpToCathGrp[projPadToGrp[i]];
  }

  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    vectorPrintShort("  projPadToGrp", projPadToGrp, nProjPads);
    vectorPrintShort("  cath0ToGrp", cathGroup[0], nCath0);
    vectorPrintShort("  cath1ToGrp", cathGroup[1], nCath1);
  }
  //
  return nNewGrp;
}

int ClusterPEM::getNbrOfPadsInGroup(int g)
{
  int nbrOfPads = 0;
  for (int c = 0; c < 2; c++) {
    int nbrPads = getNbrOfPads(c);
    for (int p = 0; p < nbrPads; p++) {
      if (cathGroup[c][p] == g) {
        nbrOfPads++;
      }
    }
  }
  return nbrOfPads;
}

void ClusterPEM::removeLowChargedGroups(int nGroups)
{
  int nbrPadsInGroup[2][nGroups + 1];
  double chargeInGroup[2][nGroups + 1];
  vectorSetInt(nbrPadsInGroup[0], 0, nGroups);
  vectorSetInt(nbrPadsInGroup[1], 0, nGroups);
  vectorSet(chargeInGroup[0], 0, nGroups);
  vectorSet(chargeInGroup[1], 0, nGroups);
  int nbrCath = 0;
  //
  // Compute the total charge of a group
  for (int c = 0; c < 2; c++) {
    int nbrPads = pads[c]->getNbrOfPads();
    const double* q = pads[c]->getCharges();
    nbrCath += (nbrPads != 0) ? 1 : 0;
    for (int p = 0; p < nbrPads; p++) {
      nbrPadsInGroup[c][cathGroup[c][p]]++;
      chargeInGroup[c][cathGroup[c][p]] += q[p];
    }
  }

  char str[256];
  for (Groups_t g = 1; g < nGroups + 1; g++) {
    double charge = chargeInGroup[0][g] + chargeInGroup[1][g];
    charge = charge / nbrCath;
    int nbrPads = nbrPadsInGroup[0][g] + nbrPadsInGroup[1][g];
    if ((charge < ClusterConfig::minChargeOfClusterPerCathode) &&
        (nbrPads > 0)) {
      // Remove groups
      // printf("  Remove group %d, charge=%f\n", g, charge);
      // scanf("%s", str);
      // Suppress the pads
      for (int c = 0; c < 2; c++) {
        int nbrPads = pads[c]->getNbrOfPads();
        for (int p = 0; p < nbrPads; p++) {
          if (cathGroup[c][p] == g) {
            cathGroup[c][p] = 0;
          }
        }
      }
      if (ClusterConfig::groupsLog >= ClusterConfig::detail || ClusterConfig::processingLog >= ClusterConfig::info) {
        int nbrPads = chargeInGroup[0][g] + chargeInGroup[1][g];
        printf("> [removeLowChargedGroups] Remove low charge group g=%d, # pads=%d\n", g, nbrPads);
      }
    }
  }
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    vectorPrintShort("  cathToGrp[0]", cathGroup[0], pads[0]->getNbrOfPads());
    vectorPrintShort("  cathToGrp[1]", cathGroup[1], pads[1]->getNbrOfPads());
  }
}

// Remove the seeds outside of the frame delimiting the cluster.
int ClusterPEM::filterFitModelOnClusterRegion(Pads& pads, double* theta, int K,
                                              Mask_t* maskFilteredTheta)
{
  //
  // Spatial filter
  //
  const double* x = pads.getX();
  const double* y = pads.getY();
  const double* dx = pads.getDX();
  const double* dy = pads.getDY();
  int N = pads.getNbrOfPads();
  // compute the frame enclosing the pads Min/Max x/y
  double xyTmp[N];
  int kSpacialFilter = 0;
  vectorAddVector(x, -1.0, dx, N, xyTmp);
  double xMin = vectorMin(xyTmp, N);
  vectorAddVector(x, +1.0, dx, N, xyTmp);
  double xMax = vectorMax(xyTmp, N);
  vectorAddVector(y, -1.0, dy, N, xyTmp);
  double yMin = vectorMin(xyTmp, N);
  vectorAddVector(y, +1.0, dy, N, xyTmp);
  double yMax = vectorMax(xyTmp, N);
  double* muX = getMuX(theta, K);
  double* muY = getMuY(theta, K);
  for (int k = 0; k < K; k++) {
    maskFilteredTheta[k] = 0;
    if ((muX[k] > xMin) && (muX[k] < xMax)) {
      if ((muY[k] > yMin) && (muY[k] < yMax)) {
        maskFilteredTheta[k] = 1;
        kSpacialFilter++;
      }
    }
  }

  if ((ClusterConfig::fittingLog >= ClusterConfig::info) && (kSpacialFilter != K)) {
    printf("[filterFitModelOnClusterRegion] ---> Spacial Filter; removing %d hit\n", K - kSpacialFilter);
  }
  //
  // W filter
  // w cut-off
  double cutOff = 0.02 / kSpacialFilter;
  //
  double* w_ = getW(theta, K);
  double w[K];
  double wSum = 0.0;
  // Normalize new w
  for (int k = 0; k < K; k++) {
    wSum += (maskFilteredTheta[k] * w_[k]);
  }
  int kWFilter = 0;
  double norm = 1.0 / wSum;
  for (int k = 0; k < K; k++) {
    w[k] = maskFilteredTheta[k] * w_[k] * norm;
    maskFilteredTheta[k] = maskFilteredTheta[k] && (w[k] > cutOff);
    kWFilter += (maskFilteredTheta[k] && (w[k] > cutOff));
  }
  if ((ClusterConfig::fittingLog >= ClusterConfig::detail) && (kSpacialFilter > kWFilter)) {
    printf(
      "[filterFitModelOnClusterRegion] At least one hit such as w[k] < "
      "(0.05 / K) = %8.4f) -> removing %d hit\n",
      cutOff, kSpacialFilter - kWFilter);
  }
  return kWFilter;
}

// Remove seeds which drift from the EM solutions
// And remove close seeds
int ClusterPEM::filterFitModelOnSpaceVariations(const double* thetaEM, int kEM,
                                                double* thetaFit, int kFit,
                                                Mask_t* maskFilteredTheta)
{
  // Rq: kFit is the same for thetaEM & theta
  vectorSetShort(maskFilteredTheta, 0, kFit);
  int kSpacialFilter = 0;
  //
  // Spatial filter on the theta deplacements
  //
  const double* muEMX = getConstMuX(thetaEM, kEM);
  const double* muEMY = getConstMuY(thetaEM, kEM);
  const double* muEMDx = getConstVarX(thetaEM, kEM);
  const double* muEMDy = getConstVarY(thetaEM, kEM);
  double* muX = getMuX(thetaFit, kFit);
  double* muY = getMuY(thetaFit, kFit);
  double xTmp[kEM], yTmp[kEM];

  // The order of thetaEM and thetaFit can be change
  // So take the the min distance
  for (int k = 0; k < kFit; k++) {
    // Find the the nearest seed mu[k](fit) of muEM[k]
    // Compute the distances to mu[k]
    vectorAddScalar(muEMX, -muX[k], kEM, xTmp);
    vectorMultVector(xTmp, xTmp, kEM, xTmp);
    vectorAddScalar(muEMY, -muY[k], kEM, yTmp);
    vectorMultVector(yTmp, yTmp, kEM, yTmp);
    vectorAddVector(xTmp, 1.0, yTmp, kEM, xTmp);
    // muEM[kMin] is the nearest seeds of mu[k]
    int kMin = vectorArgMin(xTmp, kEM);
    // printf("??? kMin=%d, dx=%f, dy=%f\n", kMin, muEMDx[k], muEMDy[k]);
    //
    // Build the frame around muEM[kMin] with a border
    // of 1.5 times the pixel size
    double xMin = muEMX[kMin] - 3 * muEMDx[kMin];
    double xMax = muEMX[kMin] + 3 * muEMDx[kMin];
    double yMin = muEMY[kMin] - 3 * muEMDy[kMin];
    double yMax = muEMY[kMin] + 3 * muEMDy[kMin];
    // Select Seeds which didn't move with the fitting
    if (((muX[k] > xMin) && (muX[k] < xMax)) &&
        ((muY[k] > yMin) && (muY[k] < yMax))) {
      maskFilteredTheta[k] = 1;
      kSpacialFilter++;
    } else {
      if (ClusterConfig::processingLog >= ClusterConfig::info) {
        printf("[filterFitModelOnSpaceVariations] ---> too much drift; deltaX/Y=(%6.2f,%6.2f) ---> k=%3d removed\n",
               muEMX[k] - muX[k], muEMY[k] - muY[k], k);
        printf("     ??? muEMDx[kMin], muEMDy[kMin] = %f, %f\n", muEMDx[kMin], muEMDy[kMin]);
      }
    }
  }
  if ((ClusterConfig::processingLog >= ClusterConfig::info) && (kSpacialFilter != kFit)) {
    printf("[filterFitModelOnSpaceVariations] ---> %d hit(s) removed\n", kFit - kSpacialFilter);
  }
  //
  // Suppress close seeds ~< 0.5 pad size
  // ??? inv double* muDX = getVarX(thetaFit, kFit);
  // ??? double* muDY = getVarY(thetaFit, kFit);
  double* w = getW(thetaFit, kFit);
  for (int k = 0; k < kFit; k++) {
    if (maskFilteredTheta[k]) {
      for (int l = k + 1; l < kFit; l++) {
        if (maskFilteredTheta[l]) {
          // Errror X/Y is the size of a projected pad
          double maxErrorX = 2.0 * std::fmin(muEMDx[k], muEMDx[l]);
          double maxErrorY = 2.0 * std::fmin(muEMDy[k], muEMDy[l]);
          bool xClose = std::fabs(muX[k] - muX[l]) < maxErrorX;
          bool yClose = std::fabs(muY[k] - muY[l]) < maxErrorY;
          // printf(" ??? muX k/l= %f, %f, muDX K/l= %f, %f\n",  muX[k], muX[l], muEMDx[k], muEMDx[l]);
          // printf(" ??? muY k/l= %f, %f, muDY K/l= %f, %f\n",  muY[k], muY[l], muEMDy[k], muEMDy[l]);
          if (xClose && yClose) {
            // Supress the weakest weight
            if (w[k] > w[l]) {
              maskFilteredTheta[l] = 0;
              w[k] += w[l];
            } else {
              maskFilteredTheta[k] = 0;
              w[l] += w[k];
            }
          }
        }
      }
    }
  }
  int kCloseFilter = vectorSumShort(maskFilteredTheta, kFit);
  if (ClusterConfig::processingLog >= ClusterConfig::info && (kSpacialFilter > kCloseFilter)) {
    printf(
      "[filterFitModelOnSpaceVariations] ---> removing %d close seeds\n",
      kSpacialFilter - kCloseFilter);
  }
  return kCloseFilter;
}

DataBlock_t ClusterPEM::fit(double* thetaInit, int kInit)
{
  int nbrCath0 = getNbrOfPads(0);
  int nbrCath1 = getNbrOfPads(1);
  /*
  // ??? (111) Invalid fiting
  // Build the mask to handle pads with the g-group
  Mask_t maskFit0[nbrCath0];
  Mask_t maskFit1[nbrCath1];
  Mask_t *maskFit[2] = {maskFit0, maskFit1};
  // printf(" ???? nbrCath0=%d, nbrCath1=%d\n", nbrCath0, nbrCath1);
  // Ne laplacian ??? getMaskCathToGrpFromProj( g, maskFit0, maskFit1, nbrCath0,
  nbrCath1); vectorBuildMaskEqualShort( pads[0]->cath, g, nbrCath0, maskFit0);
  vectorBuildMaskEqualShort( pads[1]->cath, g, nbrCath1, maskFit1);
  // vectorPrintShort("maskFit0", maskFit0, nbrCath0);
  // vectorPrintShort("maskFit1", maskFit1, nbrCath1);
  int nFits[2];
  nFits[1] = vectorSumShort( maskFit1, nbrCath1);
  nFits[0] = vectorSumShort( maskFit0, nbrCath0);
  */
  int nFit = nbrCath0 + nbrCath1;
  // double *xyDxyFit;
  // double *qFit;
  int filteredK = 0;
  int finalK = 0;
  // ThetaFit (output)
  double* thetaFit = new double[kInit * 5];
  vectorSet(thetaFit, 0, kInit * 5);
  if (nFit < ClusterConfig::nbrOfPadsLimitForTheFitting) {
    //
    // Preparing the fitting
    //
    /*
    xyDxyFit = new double[nFit*4];
    qFit = new double[nFit];
    Mask_t cathFit[nFit];
    Mask_t notSaturatedFit[nFit];
    */
    //

    // Concatenate the 2 planes of the subCluster For the fitting
    Pads* fitPads = new Pads(pads[0], pads[1], Pads::xydxdyMode);
    // khi2 (output)
    double khi2[1];
    // pError (output)
    double pError[3 * kInit * 3 * kInit];
    if (ClusterConfig::fittingLog >= ClusterConfig::detail) {
      printf("Starting the fitting\n");
      printf("- # cath0, cath1 for fitting: %2d %2d\n", getNbrOfPads(0),
             getNbrOfPads(1));
      printTheta("- thetaInit", 1.0, thetaInit, kInit);
    }
    // Fit
    if ((kInit * 3 - 1) <= nFit) {
      /*
      fitMathieson( thetaInit, xyDxyFit, qFit, cathFit, notSaturatedFit,
      zCathTotalCharge, K, nFit, chamberId, processFitVerbose, thetaFit, khi2,
      pError
                );
      */
      fitMathieson(*fitPads, thetaInit, kInit, processFitVerbose, thetaFit,
                   khi2, pError);
    } else {
      printf("---> Fitting parameters to large : k=%d, 3k-1=%d, nFit=%d\n",
             kInit, kInit * 3 - 1, nFit);
      printf("     keep the EM solution\n");
      vectorCopy(thetaInit, kInit * 5, thetaFit);
    }
    if (ClusterConfig::fittingLog >= ClusterConfig::info) {
      printTheta("- thetaFit", 1.0, thetaFit, kInit);
    }
    // Filter Fitting solution
    Mask_t maskFilterFit[kInit];
    // filteredK =
    //   filterFitModelOnClusterRegion(*fitPads, thetaFit, kInit, maskFilterFit);
    int filteredK = filterFitModelOnSpaceVariations(thetaInit, kInit,
                                                    thetaFit, kInit, maskFilterFit);
    double filteredTheta[5 * filteredK];
    if ((filteredK != kInit) && (nFit >= filteredK)) {
      if (ClusterConfig::fittingLog >= ClusterConfig::info) {
        printf("Filtering the fitting K=%d >= K=%d\n", nFit, filteredK);
        // ??? Inv printTheta("- filteredTheta", filteredTheta, filteredK);
      }
      if (filteredK > 0) {
        maskedCopyTheta(thetaFit, kInit, maskFilterFit, kInit, filteredTheta,
                        filteredK);
        /*
        fitMathieson( filteredTheta, xyDxyFit, qFit, cathFit, notSaturatedFit,
                    zCathTotalCharge, filteredK, nFit,
                    chamberId, processFit,
                    filteredTheta, khi2, pError
                  );
        */
        fitMathieson(*fitPads, filteredTheta, filteredK, processFitVerbose,
                     filteredTheta, khi2, pError);
        delete[] thetaFit;
        thetaFit = new double[filteredK * 5];
        copyTheta(filteredTheta, filteredK, thetaFit, filteredK, filteredK);
        finalK = filteredK;
      } else {
        // No hit with the fitting
        vectorCopy(thetaInit, kInit * 5, thetaFit);
        finalK = kInit;
      }
    } else {
      // ??? InvvectorCopy( thetaFit, K*5, thetaFitFinal);
      // Don't Filter, theta resul in "thetaFit"
      finalK = kInit;
    }
  } else {
    // Keep "thetaInit
    if (ClusterConfig::fittingLog >= ClusterConfig::info) {
      printf(
        "[Cluster.fit] Keep the EM Result: nFit=%d >= "
        "nbrOfPadsLimitForTheFitting=%d\n",
        nFit, ClusterConfig::nbrOfPadsLimitForTheFitting);
    }
    vectorCopy(thetaInit, kInit * 5, thetaFit);
    finalK = kInit;
  }
  return std::make_pair(finalK, thetaFit);
}

// Old release
// Invalid, Should be removed
int ClusterPEM::renumberGroupsFromMap(short* grpToGrp, int nGrp)
{
  // short renumber[nGrp+1];
  // vectorSetShort( renumber, 0, nGrp+1 );
  int maxIdx = vectorMaxShort(grpToGrp, nGrp + 1);
  std::vector<short> counters(maxIdx + 1);
  vectorSetShort(counters.data(), 0, maxIdx + 1);

  for (int g = 1; g <= nGrp; g++) {
    if (grpToGrp[g] != 0) {
      counters[grpToGrp[g]]++;
    }
  }
  int curGrp = 0;
  for (int g = 1; g <= maxIdx; g++) {
    if (counters[g] != 0) {
      curGrp++;
      counters[g] = curGrp;
    }
  }
  // Now counters contains the mapping oldGrp -> newGrp
  for (int g = 1; g <= nGrp; g++) {
    grpToGrp[g] = counters[grpToGrp[g]];
  }
  return curGrp;
}

int ClusterPEM::renumberGroups(Mask_t* grpToGrp, int nGrp)
{
  int currentGrp = 0;
  for (int g = 0; g < (nGrp + 1); g++) {
    grpToGrp[g] = 0;
  }
  for (int c = 0; c < 2; c++) {
    Mask_t* cathToGrp = cathGroup[c];
    int nbrCath = pads[c]->getNbrOfPads();
    for (int p = 0; p < nbrCath; p++) {
      int g = cathToGrp[p];
      if (g != 0) {
        // Not the background group
        // ??? printf(" p=%d, g[p]=%d, grpToGrp[g]=%d\n", p, g, grpToGrp[g]);
        if (grpToGrp[g] == 0) {
          // It's a new Group
          currentGrp++;
          // Update the map and the cath-group
          grpToGrp[g] = currentGrp;
          cathToGrp[p] = currentGrp;
        } else {
          // The cath-pad takes the group of the map (new numbering)
          cathToGrp[p] = grpToGrp[g];
        }
      }
    }
  }
  int newNbrGroups = currentGrp;
  if (ClusterConfig::groupsLog >= ClusterConfig::info) {
    printf("> Groups renumbering [renumberGroups] newNbrGroups=%d\n",
           newNbrGroups);
    vectorPrintShort("  cath0ToGrp", cathGroup[0], pads[0]->getNbrOfPads());
    vectorPrintShort("  cath1ToGrp", cathGroup[1], pads[1]->getNbrOfPads());
  }
  return newNbrGroups;
}

int ClusterPEM::findLocalMaxWithPEM(double* thetaL, int nbrOfPadsInTheGroupCath)
{

  /// ??? Verify if not already done
  // Already done if 1 group
  Pads* cath0 = pads[0];
  Pads* cath1 = pads[1];
  Pads* projPads = projectedPads;
  // Compute the charge weight of each cathode
  double cWeight0 = getTotalCharge(0);
  double cWeight1 = getTotalCharge(1);
  double preClusterCharge = cWeight0 + cWeight1;
  cWeight0 /= preClusterCharge;
  cWeight1 /= preClusterCharge;
  //
  int chId = chamberId;
  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info || ClusterConfig::processingLog >= ClusterConfig::info) {
    printf("  - [findLocalMaxWithPEM]\n");
  }
  // Trivial cluster : only 1 pads
  if (projPads->getNbrOfPads() == 1) {
    // Return the unique local maximum : the center of the pads
    double* w = getW(thetaL, nbrOfPadsInTheGroupCath);
    double* muX = getMuX(thetaL, nbrOfPadsInTheGroupCath);
    double* muY = getMuY(thetaL, nbrOfPadsInTheGroupCath);
    double* muDX = getVarX(thetaL, nbrOfPadsInTheGroupCath);
    double* muDY = getVarY(thetaL, nbrOfPadsInTheGroupCath);
    w[0] = 1.0;
    muX[0] = projPads->getX()[0];
    muY[0] = projPads->getY()[0];
    muDX[0] = projPads->getDX()[0] * 0.5;
    muDY[0] = projPads->getDY()[0] * 0.5;
    return 1;
  }
  int nMaxPads = std::fmax(getNbrOfPads(0), getNbrOfPads(1));
  Pads* pixels = projPads->refinePads();
  int nPixels = pixels->getNbrOfPads();
  // Merge pads of the 2 cathodes
  // TODO ??? : see if it can be once with Fitting (see fitPads)
  Pads* mPads = new Pads(pads[0], pads[1], Pads::xyInfSupMode);
  int nPads = mPads->getNbrOfPads();
  Pads* localMax = nullptr;
  Pads* saveLocalMax = nullptr;
  std::pair<double, double> chi2;
  int dof, nParameters;

  // Pixel initilization
  // Rq: the charge projection is not used
  pixels->setCharges(1.0);
  // Init Cij
  double* Cij = new double[nPads * nPixels];
  // Compute pad charge xyInfSup induiced by a set of charge (the pixels)
  computeFastCij(*mPads, *pixels, Cij);
  //
  // Debug computeFastCij
  /*
  double *CijTmp = new double[nPads*nPixels];
  computeCij( *mPads, *pixels, CijTmp);
  vectorAddVector( Cij, -1, CijTmp, nPads*nPixels, CijTmp);
  vectorAbs( CijTmp, nPads*nPixels, CijTmp);
  double minDiff = vectorMin(CijTmp, nPads*nPixels);
  double maxDiff = vectorMax(CijTmp, nPads*nPixels);
  int argMax = vectorArgMax(CijTmp, nPads*nPixels);
  printf("\n\n nPads, nPixels %d %d\n", nPads, nPixels);
  printf("\n\n min/max(FastCij-Cij)=%f %f nPads*i+j %d %d\n", minDiff, maxDiff,
  argMax / nPads, argMax % nPads); delete [] CijTmp;
  */
  // MaskCij: Used to disable Cij contribution (disable pixels)
  Mask_t* maskCij = new Mask_t[nPads * nPixels];
  // Init loop

  double previousCriteriom = DBL_MAX;
  double criteriom = DBL_MAX;
  bool goon = true;
  int macroIt = 0;
  if (ClusterConfig::processingLog >= ClusterConfig::info) {
    printf(
      "    Macro  nParam    chi2 chi2/ndof chi2/ndof chi2/ndof chi2/ndof  ch2/ndof\n"
      "     It.      -       -       -       cath-0    cath-1   sum 0/1  weight-sum\n");
  }
  while (goon) {
    if (localMax != nullptr) {
      saveLocalMax = new Pads(*localMax, o2::mch::Pads::xydxdyMode);
    }
    previousCriteriom = criteriom;
    if (0) {
      vectorSet(Cij, 0.0, nPads * nPixels);
      for (int j = 0; j < nPads; j++) {
        Cij[nPads * j + j] = 1.0;
        // for (int i = 0; i < nPixels; i++) {
        //   qPadPrediction[j] += Cij[nPads * i + j] * qPixels[i];
        // }
      }
    }
    chi2 =
      PoissonEMLoop(*mPads, *pixels, Cij, maskCij, 0, minPadResidues[macroIt],
                    nIterations[macroIt], getNbrOfPads(0));
    // Obsolete
    // localMax = pixels->clipOnLocalMax(true);
    localMax = pixels->extractLocalMax();
    nParameters = localMax->getNbrOfPads();
    dof = nMaxPads - 3 * nParameters + 1;
    double chi20 = chi2.first;
    double chi21 = chi2.second;
    int ndof0 = getNbrOfPads(0) - 3 * nParameters + 1;
    if (ndof0 <= 0) {
      ndof0 = 1;
    }
    int ndof1 = getNbrOfPads(1) - 3 * nParameters + 1;
    if (ndof1 <= 0.) {
      ndof1 = 1;
    }
    if (dof == 0) {
      dof = 1;
    }
    // Model selection criteriom (nbre of parameters/seeds)
    // criteriom0 = fabs( (chi20+chi21 / dof));
    // criteriom1 = fabs(sqrt(chi20 / ndof0) + sqrt(chi21 / ndof1));
    // printf( "??? cWeight0=%f, cWeight1=%f\n", cWeight0, cWeight1);
    criteriom = cWeight0 * sqrt(chi20 / ndof0) + cWeight1 * sqrt(chi21 / ndof1);

    if (ClusterConfig::processingLog >= ClusterConfig::info) {
      printf(
        "     %2d     %3d   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f\n",
        macroIt, nParameters, chi20 + chi21, sqrt((chi20 + chi21) / dof),
        sqrt(chi20 / ndof0), sqrt(chi21 / ndof1),
        sqrt(chi20 / ndof0) + sqrt(chi21 / ndof1),
        cWeight0 * sqrt(chi20 / ndof0) + cWeight1 * sqrt(chi21 / ndof1));
    }
    if (ClusterConfig::inspectModel >= ClusterConfig::active) {
      inspectSavePixels(macroIt, *pixels);
    }
    macroIt++;
    goon =
      (criteriom < 1.01 * previousCriteriom) && (macroIt < nMacroIterations);
  }
  delete pixels;
  if (criteriom < 1.01 * previousCriteriom) {
    delete saveLocalMax;
  } else {
    delete localMax;
    localMax = saveLocalMax;
  }

  //
  // Select local Max
  // Remove local Max < 0.01 * max(LocalMax)
  //
  double cutRatio = 0.01;
  double qCut =
    cutRatio * vectorMax(localMax->getCharges(), localMax->getNbrOfPads());
  int k = 0;
  double qSum = 0.0;
  // Remove the last hits if > (nMaxPads +1) / 3
  int nMaxSolutions =
    int((std::max(getNbrOfPads(0), getNbrOfPads(1)) + 1.0) / 3.0);
  // if (nMaxSolutions < 1) {
  //     nMaxSolutions = 1;
  //}
  // To avoid 0 possibility and give more inputs to the fitting
  nMaxSolutions += 1;

  int removedLocMax = localMax->getNbrOfPads();

  if (localMax->getNbrOfPads() > nMaxSolutions) {
    if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
      printf("seed selection: nbr Max parameters =%d, nLocMax=%d\n",
             nMaxSolutions, localMax->getNbrOfPads());
      printf(
        "seed selection: Reduce the nbr of solutions to fit: Take %d/%d "
        "solutions\n",
        nMaxSolutions, localMax->getNbrOfPads());
    }
    int index[localMax->getNbrOfPads()];
    for (int k = 0; k < localMax->getNbrOfPads(); k++) {
      index[k] = k;
    }
    const double* qLocalMax = localMax->getCharges();
    std::sort(index, &index[localMax->getNbrOfPads()],
              [=](int a, int b) { return (qLocalMax[a] > qLocalMax[b]); });
    // Reoder
    qCut = qLocalMax[index[nMaxSolutions - 1]] - 1.e-03;
  }
  k = localMax->removePads(qCut);
  localMax->normalizeCharges();
  removedLocMax -= k;

  if (ClusterConfig::processingLog >= ClusterConfig::info && removedLocMax != 0) {
    printf(
      "    > seed selection: Final cut -> %d percent (qcut=%8.2f), number of "
      "local max removed = %d\n",
      int(cutRatio * 100), qCut, removedLocMax);
  }

  // Store the
  int K0 = localMax->getNbrOfPads();
  int K = std::min(K0, nbrOfPadsInTheGroupCath);
  double* w = getW(thetaL, nbrOfPadsInTheGroupCath);
  double* muX = getMuX(thetaL, nbrOfPadsInTheGroupCath);
  double* muY = getMuY(thetaL, nbrOfPadsInTheGroupCath);
  double* varX = getVarX(thetaL, nbrOfPadsInTheGroupCath);
  double* varY = getVarY(thetaL, nbrOfPadsInTheGroupCath);
  const double* ql = localMax->getCharges();
  const double* xl = localMax->getX();
  const double* yl = localMax->getY();
  const double* dxl = localMax->getDX();
  const double* dyl = localMax->getDY();
  for (int k = 0; k < K; k++) {
    w[k] = ql[k];
    muX[k] = xl[k];
    muY[k] = yl[k];
    varX[k] = dxl[k];
    varY[k] = dyl[k];
  }
  delete localMax;
  delete[] Cij;
  delete[] maskCij;
  return K;
}

// Propagate back cath-group to projection pads
void ClusterPEM::updateProjectionGroups()
{
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    printf("> Update projected Groups [updateProjectionGroups]\n");
  }
  int nProjPads = projectedPads->getNbrOfPads();
  Groups_t* cath0ToGrp = cathGroup[0];
  Groups_t* cath1ToGrp = cathGroup[1];

  // Save projPadToGrp to Check
  Groups_t savePadGrp[nProjPads];
  if (ClusterConfig::groupsCheck) {
    vectorCopyShort(projPadToGrp, nProjPads, savePadGrp);
  }
  for (int k = 0; k < nProjPads; k++) {
    MapKToIJ_t ij = mapKToIJ[k];
    PadIdx_t i = ij.i;
    PadIdx_t j = ij.j;
    if ((i > -1) && (j == -1)) {
      // int cath0Idx = mapPadToCathIdx[ i ];
      projPadToGrp[k] = cath0ToGrp[i];
      // printf("  projPadToGrp[k] = cath0ToGrp[cath0Idx], i=%d, j=%d,
      // cath0Idx=%d, cath0ToGrp[cath0Idx]=%d\n", i, j, cath0Idx,
      // cath0ToGrp[cath0Idx]);
    } else if ((i == -1) && (j > -1)) {
      // int cath1Idx = mapPadToCathIdx[ j ];
      projPadToGrp[k] = cath1ToGrp[j];
      // printf("  projPadToGrp[k] = cath1ToGrp[cath1Idx], i=%d, j=%d,
      // cath1Idx=%d, cath1ToGrp[cath1Idx]=%d\n", i, j, cath1Idx,
      // cath1ToGrp[cath1Idx]);
    } else if ((i > -1) && (j > -1)) {
      // projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ];
      projPadToGrp[k] = cath0ToGrp[i];
      // ??? if (ClusterConfig::groupsCheck && (cath0ToGrp[i] != cath1ToGrp[j])) {
      if (0) {
        printf(
          "  [updateProjectionGroups] i, cath0ToGrp[i]=(%d, %d); j, "
          "cath1ToGrp[j]=(%d, %d)\n",
          i, cath0ToGrp[i], j, cath1ToGrp[j]);
        throw std::overflow_error(
          "updateProjectionGroups cath0ToGrp[i] != cath1ToGrp[j]");
      }
      // printf("  projPadToGrp[k] = grpToGrp[ projPadToGrp[k] ], i=%d, j=%d,
      // k=%d \n", i, j, k);
    } else {
      throw std::overflow_error("updateProjectionGroups i,j=-1");
    }
  }
  if (ClusterConfig::groupsLog >= ClusterConfig::detail) {
    vectorPrintShort("  updated projGrp", projPadToGrp, nProjPads);
  }
  if (0) {
    bool same = true;
    for (int p = 0; p < nProjPads; p++) {
      same = same && (projPadToGrp[p] == savePadGrp[p]);
    }
    if (same == false) {
      vectorPrintShort("  WARNING: old projPadToGrp", savePadGrp, nProjPads);
      vectorPrintShort("  WARNING: new projPadToGrp", projPadToGrp, nProjPads);
      // throw std::overflow_error("updateProjectionGroups projection has
      // changed");
    }
  }
}

// Not used in the Clustering/fitting
// Just to check hit results
int ClusterPEM::laplacian2D(const Pads& pads_, PadIdx_t* neigh, int chId,
                            PadIdx_t* sortedLocalMax, int kMax, double* smoothQ)
{
  // ??? Place somewhere
  double eps = 1.0e-7;
  double noise = 4. * 0.22875;
  double laplacianCutOff = noise;
  // ??? Inv int atLeastOneMax = -1;
  //
  int N = pads_.getNbrOfPads();
  const double* x = pads_.getX();
  const double* y = pads_.getY();
  const double* dx = pads_.getDX();
  const double* dy = pads_.getDY();
  const double* q = pads_.getCharges();
  //
  // Laplacian allocation
  double lapl[N];
  // Locations not used as local max
  Mask_t unselected[N];
  vectorSetShort(unselected, 1, N);
  // printNeighbors(neigh, N);
  for (int i = 0; i < N; i++) {
    int nNeigh = 0;
    double sumNeigh = 0;
    int nNeighSmaller = 0;
    // printf("  Neighbors of i=%d [", i);
    //
    // For all neighbours of i
    for (PadIdx_t* neigh_ptr = getTheFirtsNeighborOf(neigh, i);
         *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      // printf("%d ,", j);
      // nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[i]));
      nNeighSmaller += (q[j] <= ((q[i] + noise) * unselected[j]));
      nNeigh++;
      sumNeigh += q[j];
    }
    // printf("]");
    // printf(" nNeighSmaller %d / nNeigh %d \n", nNeighSmaller, nNeigh);
    lapl[i] = float(nNeighSmaller) / nNeigh;
    if (lapl[i] < laplacianCutOff) {
      lapl[i] = 0.0;
    }
    unselected[i] = (lapl[i] != 1.0);
    smoothQ[i] = sumNeigh / nNeigh;
    if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
      printf(
        "Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, "
        "smoothQ[i]=%6.3f, lapl[i]=%6.3f\n",
        i, x[i], y[i], q[i], smoothQ[i], lapl[i]);
    }
  }
  //
  // Get local maxima
  Mask_t localMaxMask[N];
  vectorBuildMaskEqual(lapl, 1.0, N, localMaxMask);
  // Get the location in lapl[]
  // Inv ??? int nSortedIdx = vectorSumShort( localMaxMask, N );
  // ??? Inv int sortPadIdx[nSortedIdx];
  int nSortedIdx = vectorGetIndexFromMask(localMaxMask, N, sortedLocalMax);
  // Sort the slected laplacian (index sorting)
  // Indexes for sorting
  // Rq: Sometimes chage the order of max
  // std::sort( sortedLocalMax, &sortedLocalMax[nSortedIdx], [=](int a, int b){
  // return smoothQ[a] > smoothQ[b]; });
  std::sort(sortedLocalMax, &sortedLocalMax[nSortedIdx],
            [=](int a, int b) { return q[a] > q[b]; });
  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
    vectorPrint("  sort w", q, N);
    vectorPrintInt("  sorted q-indexes", sortedLocalMax, nSortedIdx);
  }

  ////
  // Filtering local max
  ////

  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
    printf("  filtering Local Max\n");
  }
  // At Least one locMax
  if ((nSortedIdx == 0) && (N != 0)) {
    // Take the first pad
    if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
      printf("-> No local Max, take the highest value < 1\n");
    }
    sortedLocalMax[0] = 0;
    nSortedIdx = 1;
    return nSortedIdx;
  }

  // For a small number of pads
  // limit the number of max to 1 local max
  // if the aspect ratio of the cluster
  // is close to 1
  double aspectRatio = 0;
  if ((N > 0) && (N < 6) && (chId <= 6)) {
    double xInf = DBL_MAX, xSup = DBL_MIN, yInf = DBL_MAX, ySup = DBL_MIN;
    // Compute aspect ratio of the cluster
    for (int i = 0; i < N; i++) {
      xInf = fmin(xInf, x[i] - dx[i]);
      xSup = fmax(xSup, x[i] + dx[i]);
      yInf = fmin(yInf, y[i] - dy[i]);
      ySup = fmax(ySup, y[i] + dy[i]);
    }
    // Nbr of pads in x-direction
    int nX = int((xSup - xInf) / dx[0] + eps);
    // Nbr of pads in y-direction
    int nY = int((ySup - yInf) / dy[0] + eps);
    aspectRatio = fmin(nX, nY) / fmax(nX, nY);
    if (aspectRatio > 0.6) {
      // Take the max
      nSortedIdx = 1;
      if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
        printf(
          "  -> Limit to one local Max, nPads=%d, chId=%d, aspect ratio=%6.3f\n",
          N, chId, aspectRatio);
      }
    }
  }

  // Suppress noisy peaks  when at least 1 peak
  // is bigger than
  if ((N > 0) && (q[sortedLocalMax[0]] > 2 * noise)) {
    int trunkIdx = nSortedIdx;
    for (int ik = 0; ik < nSortedIdx; ik++) {
      if (q[sortedLocalMax[ik]] <= 2 * noise) {
        trunkIdx = ik;
      }
    }
    nSortedIdx = std::max(trunkIdx, 1);
    if ((ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) && (trunkIdx != nSortedIdx)) {
      printf("-> Suppress %d local Max. too noisy (q < %6.3f),\n",
             nSortedIdx - trunkIdx, 2 * noise);
    }
  }
  // At most
  // int nbrOfLocalMax = floor( (N + 1) / 3.0 );
  // if  ( nSortedIdx > nbrOfLocalMax) {
  //  printf("Suppress %d local Max. the limit of number of local max %d is
  //  reached (< %d)\n", nSortedIdx-nbrOfLocalMax, nSortedIdx, nbrOfLocalMax);
  //  nSortedIdx = nbrOfLocalMax;
  //}

  return nSortedIdx;
}

// Not used in the Clustering/fitting
// Just to check hit results
int ClusterPEM::findLocalMaxWithBothCathodes(double* thetaOut, int kMax)
{

  int N0 = 0, N1 = 0;
  if (nbrOfCathodePlanes != 2) {
    if (singleCathPlaneID == 0) {
      N0 = pads[0]->getNbrOfPads();
    } else {
      N1 = pads[1]->getNbrOfPads();
    }
  } else {
    N0 = pads[0]->getNbrOfPads();
    N1 = pads[1]->getNbrOfPads();
  }
  int kMax0 = N0;
  int kMax1 = N1;

  // Number of seeds founds
  int k = 0;
  //
  // Pad indexes of local max. allocation per cathode
  PadIdx_t localMax0[kMax0];
  PadIdx_t localMax1[kMax1];
  // Smoothed values of q[0/1] with neighbours
  double smoothQ0[N0];
  double smoothQ1[N1];
  // Local Maximum for each cathodes
  // There are sorted with the lissed q[O/1] values
  if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
    printf("> [findLocalMaxWithBothCathodes] N0=%d N1=%d\n", N0, N1);
  }
  PadIdx_t* grpNeighborsCath0 = nullptr;
  PadIdx_t* grpNeighborsCath1 = nullptr;
  // Number of local max
  int K0 = 0, K1 = 0;
  if (N0) {
    grpNeighborsCath0 = pads[0]->buildFirstNeighbors();
    K0 = laplacian2D(*pads[0], grpNeighborsCath0, chamberId, localMax0, kMax0,
                     smoothQ0);
  }
  if (N1) {
    grpNeighborsCath1 = pads[1]->buildFirstNeighbors();
    K1 = laplacian2D(*pads[1], grpNeighborsCath1, chamberId, localMax1, kMax1,
                     smoothQ1);
  }

  // Seed allocation
  double localXMax[K0 + K1];
  double localYMax[K0 + K1];
  double localQMax[K0 + K1];
  //
  // Need an array to transform global index to the grp indexes
  PadIdx_t mapIToGrpIdx[N0];
  vectorSetInt(mapIToGrpIdx, -1, N0);
  PadIdx_t mapGrpIdxToI[N0];
  for (int i = 0; i < N0; i++) {
    // ??? printf("mapGrpIdxToI[%d]=%d\n", i, mapGrpIdxToI[i]);
    // VPads mapIToGrpIdx[ mapGrpIdxToI[i]] = i;
    mapIToGrpIdx[i] = i;
    mapGrpIdxToI[i] = i;
  }
  PadIdx_t mapJToGrpIdx[N1];
  vectorSetInt(mapJToGrpIdx, -1, N1);
  PadIdx_t mapGrpIdxToJ[N0];
  for (int j = 0; j < N1; j++) {
    // ??? printf("mapGrpIdxToJJ[%d]=%d\n", j, mapGrpIdxToJ[j]);
    // Vpads mapJToGrpIdx[ mapGrpIdxToJ[j]] = j;
    mapJToGrpIdx[j] = j;
    mapGrpIdxToJ[j] = j;
  }
  const double* x0;
  const double* y0;
  const double* dx0;
  const double* dy0;
  const double* q0;

  const double* x1;
  const double* y1;
  const double* dx1;
  const double* dy1;
  const double* q1;
  if (N0) {
    x0 = pads[0]->getX();
    y0 = pads[0]->getY();
    dx0 = pads[0]->getDX();
    dy0 = pads[0]->getDY();
    q0 = pads[0]->getCharges();
  }
  if (N1) {
    x1 = pads[1]->getX();
    y1 = pads[1]->getY();
    dx1 = pads[1]->getDX();
    dy1 = pads[1]->getDY();
    q1 = pads[1]->getCharges();
  }
  const double* xProj = projectedPads->getX();
  const double* yProj = projectedPads->getY();
  const double* dxProj = projectedPads->getDX();
  const double* dyProj = projectedPads->getDX();

  // Debug
  // vectorPrintInt( "mapIToGrpIdx", mapIToGrpIdx, N0);
  // vectorPrintInt( "mapJToGrpIdx", mapJToGrpIdx, N1);
  if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
    vectorPrint("findLocalMax q0", q0, N0);
    vectorPrint("findLocalMax q1", q1, N1);
    vectorPrintInt("findLocalMax localMax0", localMax0, K0);
    vectorPrintInt("findLocalMax localMax1", localMax1, K1);
  }

  //
  // Make the combinatorics between the 2 cathodes
  // - Take the maxOf( N0,N1) for the external loop
  //
  if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
    printf("  Local max per cathode K0=%d, K1=%d\n", K0, K1);
  }
  bool K0GreaterThanK1 = (K0 >= K1);
  bool K0EqualToK1 = (K0 == K1);
  // Choose the highest last local max.
  bool highestLastLocalMax0;
  if (K0 == 0) {
    highestLastLocalMax0 = false;
  } else if (K1 == 0) {
    highestLastLocalMax0 = true;
  } else {
    // highestLastLocalMax0 = (smoothQ0[localMax0[std::max(K0-1, 0)]] >=
    // smoothQ1[localMax1[std::max(K1-1,0)]]);
    highestLastLocalMax0 = (q0[localMax0[std::max(K0 - 1, 0)]] >=
                            q1[localMax1[std::max(K1 - 1, 0)]]);
  }
  // Permute cathodes if necessary
  int NU, NV;
  int KU, KV;
  PadIdx_t *localMaxU, *localMaxV;
  const double *qU, *qV;
  PadIdx_t* interUV;
  bool permuteIJ;
  const double *xu, *yu, *dxu, *dyu;
  const double *xv, *yv, *dxv, *dyv;
  const PadIdx_t *mapGrpIdxToU, *mapGrpIdxToV;
  PadIdx_t *mapUToGrpIdx, *mapVToGrpIdx;

  // Do permutation between cath0/cath1 or not
  if (K0GreaterThanK1 || (K0EqualToK1 && highestLastLocalMax0)) {
    NU = N0;
    NV = N1;
    KU = K0;
    KV = K1;
    xu = x0;
    yu = y0;
    dxu = dx0;
    dyu = dy0;
    xv = x1;
    yv = y1;
    dxv = dx1;
    dyv = dy1;
    localMaxU = localMax0;
    localMaxV = localMax1;
    // qU = smoothQ0; qV = smoothQ1;
    qU = q0;
    qV = q1;
    interUV = IInterJ;
    mapGrpIdxToU = mapGrpIdxToI;
    mapGrpIdxToV = mapGrpIdxToJ;
    mapUToGrpIdx = mapIToGrpIdx;
    mapVToGrpIdx = mapJToGrpIdx;
    permuteIJ = false;
  } else {
    NU = N1;
    NV = N0;
    KU = K1;
    KV = K0;
    xu = x1;
    yu = y1;
    dxu = dx1;
    dyu = dy1;
    xv = x0;
    yv = y0;
    dxv = dx0;
    dyv = dy0;
    localMaxU = localMax1;
    localMaxV = localMax0;
    // qU = smoothQ1; qV = smoothQ0;
    qU = q1;
    qV = q0;
    interUV = JInterI;
    mapGrpIdxToU = mapGrpIdxToJ;
    mapGrpIdxToV = mapGrpIdxToI;
    mapUToGrpIdx = mapJToGrpIdx;
    mapVToGrpIdx = mapIToGrpIdx;
    permuteIJ = true;
  }
  // Keep the memory of the localMaxV already assigned
  Mask_t qvAvailable[KV];
  vectorSetShort(qvAvailable, 1, KV);
  // Compact intersection matrix
  PadIdx_t* UInterV;
  //
  // Cathodes combinatorics
  if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
    printf("  Local max combinatorics: KU=%d KV=%d\n", KU, KV);
    // printXYdXY("Projection", xyDxyProj, NProj, NProj, 0, 0);
    // printf("  mapIJToK=%p, N0=%d N1=%d\n", mapIJToK, N0, N1);
    for (int i = 0; i < N0; i++) {
      // VPads int ii = mapGrpIdxToI[i];
      int ii = i;
      for (int j = 0; j < N1; j++) {
        // VPads int jj = mapGrpIdxToJ[j];
        int jj = j;
        // if ( (mapIJToK[ii*nbrCath1+jj] != -1))
        printf("   %d inter %d, grp : %d inter %d yes=%d\n", ii, jj, i, j,
               mapIJToK[ii * N1 + jj]);
      }
    }
  }
  for (int u = 0; u < KU; u++) {
    //
    PadIdx_t uPadIdx = localMaxU[u];
    double maxValue = 0.0;
    // Cathode-V pad index of the max (localMaxV)
    PadIdx_t maxPadVIdx = -1;
    // Index in the maxCathv
    PadIdx_t maxCathVIdx = -1;
    // Choose the best localMaxV
    // i.e. the maximum value among
    // the unselected localMaxV
    //
    // uPadIdx in index in the Grp
    // need to get the cluster index
    // to checck the intersection
    // VPads int ug = mapGrpIdxToU[uPadIdx];
    int ug = uPadIdx;
    if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
      printf("  Cathode u=%d localMaxU[u]=%d, x,y= %6.3f,  %6.3f, q=%6.3f\n", u,
             localMaxU[u], xu[localMaxU[u]], yu[localMaxU[u]],
             qU[localMaxU[u]]);
    }
    bool interuv;
    for (int v = 0; v < KV; v++) {
      PadIdx_t vPadIdx = localMaxV[v];
      // VPads int vg = mapGrpIdxToV[vPadIdx];
      int vg = vPadIdx;
      if (permuteIJ) {
        // printf("uPadIdx=%d,vPadIdx=%d, mapIJToK[vPadIdx*N0+uPadIdx]=%d
        // permute\n",uPadIdx,vPadIdx, mapIJToK[vPadIdx*N0+uPadIdx]);
        interuv = (mapIJToK[vg * N1 + ug] != -1);
      } else {
        // printf("uPadIdx=%d,vPadIdx=%d,
        // mapIJToK[uPadIdx*N1+vPadIdx]=%d\n",uPadIdx,vPadIdx,
        // mapIJToK[uPadIdx*N1+vPadIdx]);
        interuv = (mapIJToK[ug * N1 + vg] != -1);
      }
      if (interuv) {
        double val = qV[vPadIdx] * qvAvailable[v];
        if (val > maxValue) {
          maxValue = val;
          maxCathVIdx = v;
          maxPadVIdx = vPadIdx;
        }
      }
    }
    // A this step, we've got (or not) an
    // intercepting pad v with u. This v is
    // the maximum of all possible values
    // ??? printf("??? maxPadVIdx=%d, maxVal=%f\n", maxPadVIdx, maxValue);
    if (maxPadVIdx != -1) {
      // Found an intersevtion and a candidate
      // add in the list of seeds
      PadIdx_t kProj;
      int vg = mapGrpIdxToV[maxPadVIdx];
      if (permuteIJ) {
        kProj = mapIJToK[vg * N1 + ug];
      } else {
        kProj = mapIJToK[ug * N1 + vg];
      }
      // mapIJToK and projection UNUSED ????
      localXMax[k] = xProj[kProj];
      localYMax[k] = yProj[kProj];
      // localQMax[k] = 0.5 * (qU[uPadIdx] + qV[maxPadVIdx]);
      localQMax[k] = qU[uPadIdx];
      // Cannot be selected again as a seed
      qvAvailable[maxCathVIdx] = 0;
      if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
        printf(
          "    found intersection of u with v: u,v=(%d,%d) , x=%f, y=%f, "
          "w=%f\n",
          u, maxCathVIdx, localXMax[k], localYMax[k], localQMax[k]);
        // printf("Projection u=%d, v=%d, uPadIdx=%d, ,maxPadVIdx=%d, kProj=%d,
        // xProj[kProj]=%f, yProj[kProj]=%f\n", u, maxCathVIdx,
        //        uPadIdx, maxPadVIdx, kProj, xProj[kProj], yProj[kProj] );
        // kProj = mapIJToK[maxPadVIdx*N0 + uPadIdx];
        // printf(" permut kProj=%d xProj[kProj], yProj[kProj] = %f %f\n",
        // kProj, xProj[kProj], yProj[kProj] );
      }
      k++;
    } else {
      // No intersection u with localMaxV set
      // Approximate the seed position
      //
      // Search v pads intersepting u
      PadIdx_t* uInterV;
      PadIdx_t uPad = 0;
      if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
        printf(
          "  No intersection between u=%d and v-set of , approximate the "
          "location\n",
          u);
      }
      // Go to the mapGrpIdxToU[uPadIdx] (???? mapUToGrpIdx[uPadIdx])
      uInterV = interUV;
      if (NV != 0) {
        for (uInterV = interUV; uPad < ug; uInterV++) {
          if (*uInterV == -1) {
            uPad++;
          }
        }
      }
      // if (uInterV) printf("??? uPad=%d, uPadIdx=%d *uInterV=%d\n", uPad,
      // uPadIdx, *uInterV); If intercepting pads or no V-Pad
      if ((NV != 0) && (uInterV[0] != -1)) {
        double vMin = 1.e+06;
        double vMax = -1.e+06;
        // Take the most precise direction
        if (dxu[u] < dyu[u]) {
          // x direction most precise
          // Find the y range intercepting pad u
          for (; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
              printf("  Global upad=%d intersect global vpad=%d grpIdx=%d\n",
                     uPad, *uInterV, idx);
            }
            if (idx != -1) {
              vMin = fmin(vMin, yv[idx] - dyv[idx]);
              vMax = fmax(vMax, yv[idx] + dyv[idx]);
            }
          }
          localXMax[k] = xu[uPadIdx];
          localYMax[k] = 0.5 * (vMin + vMax);
          localQMax[k] = qU[uPadIdx];
          if (localYMax[k] == 0 &&
              (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info)) {
            printf("WARNING localYMax[k] == 0, meaning no intersection");
          }
        } else {
          // y direction most precise
          // Find the x range intercepting pad u
          for (; *uInterV != -1; uInterV++) {
            PadIdx_t idx = mapVToGrpIdx[*uInterV];
            if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
              printf(" Global upad=%d intersect global vpad=%d  grpIdx=%d \n",
                     uPad, *uInterV, idx);
            }
            if (idx != -1) {
              if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
                printf(
                  "xv[idx], yv[idx], dxv[idx], dyv[idx]: %6.3f %6.3f "
                  "%6.3f %6.3f\n",
                  xv[idx], yv[idx], dxv[idx], dyv[idx]);
              }
              vMin = fmin(vMin, xv[idx] - dxv[idx]);
              vMax = fmax(vMax, xv[idx] + dxv[idx]);
            }
          }
          localXMax[k] = 0.5 * (vMin + vMax);
          localYMax[k] = yu[uPadIdx];
          localQMax[k] = qU[uPadIdx];
          // printf(" uPadIdx = %d/%d\n", uPadIdx, KU);
          if (localXMax[k] == 0 &&
              (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info)) {
            printf("WARNING localXMax[k] == 0, meaning no intersection");
          }
        }
        if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::no) {
          printf(
            "  solution found with all intersection of u=%d with all v, x "
            "more precise %d, position=(%f,%f), qU=%f\n",
            u, (dxu[u] < dyu[u]), localXMax[k], localYMax[k],
            localQMax[k]);
        }
        k++;
      } else {
        // No interception in the v-list
        // or no V pads
        // Takes the u values
        // printf("No intersection of the v-set with u=%d, take the u location",
        // u);

        localXMax[k] = xu[uPadIdx];
        localYMax[k] = yu[uPadIdx];
        localQMax[k] = qU[uPadIdx];
        if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::no) {
          printf(
            "  No intersection with u, u added in local Max: k=%d u=%d, "
            "position=(%f,%f), qU=%f\n",
            k, u, localXMax[k], localYMax[k], localQMax[k]);
        }
        k++;
      }
    }
  }
  // Proccess unselected localMaxV
  for (int v = 0; v < KV; v++) {
    if (qvAvailable[v]) {
      int l = localMaxV[v];
      localXMax[k] = xv[l];
      localYMax[k] = yv[l];
      localQMax[k] = qV[l];
      if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
        printf(
          "  Remaining VMax, v added in local Max:  v=%d, "
          "position=(%f,%f), qU=%f\n",
          v, localXMax[k], localYMax[k], localQMax[k]);
      }
      k++;
    }
  }
  // k seeds
  double* varX = getVarX(thetaOut, kMax);
  double* varY = getVarY(thetaOut, kMax);
  double* muX = getMuX(thetaOut, kMax);
  double* muY = getMuY(thetaOut, kMax);
  double* w = getW(thetaOut, kMax);
  //
  double wRatio = 0;
  for (int k_ = 0; k_ < k; k_++) {
    wRatio += localQMax[k_];
  }
  wRatio = 1.0 / wRatio;
  if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
    printf("Local max found k=%d kmax=%d\n", k, kMax);
  }
  for (int k_ = 0; k_ < k; k_++) {
    muX[k_] = localXMax[k_];
    muY[k_] = localYMax[k_];
    w[k_] = localQMax[k_] * wRatio;
    if (ClusterConfig::laplacianLocalMaxLog > ClusterConfig::info) {
      printf("  w=%6.3f, mux=%7.3f, muy=%7.3f\n", w[k_], muX[k_], muY[k_]);
    }
  }
  return k;
}

} // namespace mch
} // namespace o2
