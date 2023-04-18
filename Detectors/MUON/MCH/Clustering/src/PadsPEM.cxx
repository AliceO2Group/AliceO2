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

/// \file PadPEM.cxx
/// \brief Pads representation and transformation
///
/// \author Gilles Grasseau, Subatech

#include <cstring>
#include <stdexcept>
#include <vector>

#include "MCHClustering/PadsPEM.h"
#include "mathieson.h"
#include "mathUtil.h"

#define VERBOSE 1
#define CHECK 1

namespace o2
{
namespace mch
{

extern ClusterConfig clusterConfig;

void Pads::padBoundsToCenter(const Pads& pads)
{
  if (mode == PadMode::xyInfSupMode) {
    double* xInf = pads.x;
    double* yInf = pads.y;
    double* xSup = pads.dx;
    double* ySup = pads.dy;
    for (int i = 0; i < nPads; i++) {
      dx[i] = 0.5 * (xSup[i] - xInf[i]);
      dy[i] = 0.5 * (ySup[i] - yInf[i]);
      x[i] = xInf[i] + dx[i];
      y[i] = yInf[i] + dy[i];
    }
    mode = PadMode::xydxdyMode;
  }
}

void Pads::padCenterToBounds(const Pads& pads)
{
  if (mode == PadMode::xyInfSupMode) {
    double* xInf = x;
    double* yInf = y;
    double* xSup = dx;
    double* ySup = dy;
    for (int i = 0; i < nPads; i++) {
      xInf[i] = pads.x[i] - pads.dx[i];
      xSup[i] = pads.x[i] + pads.dx[i];
      yInf[i] = pads.y[i] - pads.dy[i];
      ySup[i] = pads.y[i] + pads.dy[i];
    }
    mode = PadMode::xydxdyMode;
  }
}

void Pads::padCenterToBounds()
{
  if (mode == PadMode::xydxdyMode) {
    double* xInf = x;
    double* yInf = y;
    double* xSup = dx;
    double* ySup = dy;
    double u;
    for (int i = 0; i < nPads; i++) {
      u = x[i];
      xInf[i] = u - dx[i];
      xSup[i] = u + dx[i];
      u = y[i];
      yInf[i] = u - dy[i];
      ySup[i] = u + dy[i];
    }
    mode = PadMode::xyInfSupMode;
  }
}

void Pads::padBoundsToCenter()
{
  if (mode == PadMode::xyInfSupMode) {
    double* xInf = x;
    double* yInf = y;
    double* xSup = dx;
    double* ySup = dy;
    double du;
    for (int i = 0; i < nPads; i++) {
      dx[i] = 0.5 * (xSup[i] - xInf[i]);
      dy[i] = 0.5 * (ySup[i] - yInf[i]);
      x[i] = xInf[i] + dx[i];
      y[i] = yInf[i] + dy[i];
    }
    mode = PadMode::xydxdyMode;
  }
}
PadIdx_t* Pads::buildFirstNeighbors(double* X, double* Y, double* DX,
                                    double* DY, int N)
{
  const double eps = epsilonGeometry;
  PadIdx_t* neighbors = new PadIdx_t[MaxNeighbors * N];
  for (PadIdx_t i = 0; i < N; i++) {
    PadIdx_t* i_neigh = getNeighborListOf(neighbors, i);
    // Search neighbors of i
    for (PadIdx_t j = 0; j < N; j++) {

      int xMask0 = (std::fabs(X[i] - X[j]) < (DX[i] + DX[j]) + eps);
      int yMask0 = (std::fabs(Y[i] - Y[j]) < (DY[i] + eps));
      int xMask1 = (std::fabs(X[i] - X[j]) < (DX[i] + eps));
      int yMask1 = (std::fabs(Y[i] - Y[j]) < (DY[i] + DY[j] + eps));
      if ((xMask0 && yMask0) || (xMask1 && yMask1)) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 &&
        // yMask0), (xMask1 && yMask1));
      }
    }
    *i_neigh = -1;
    if (CHECK &&
        (std::fabs(i_neigh - getNeighborListOf(neighbors, i)) > MaxNeighbors)) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n", i,
             i_neigh - getNeighborListOf(neighbors, i), MaxNeighbors);
      throw std::out_of_range("Not enough allocation");
    }
  }
  return neighbors;
}
// Build the K-neighbors list
PadIdx_t* Pads::buildKFirstsNeighbors(int kernelSize)
{
  // kernelSize must be in the interval [0:2]
  const double eps = epsilonGeometry;
  const double* X = x;
  const double* Y = y;
  const double* DX = dx;
  const double* DY = dy;
  int N = nPads;
  if ((kernelSize < 0) || (kernelSize > 2)) {
    // set to default values
    printf("Warning in getNeighbors : kerneSize overwritten by the default\n");
    kernelSize = 1;
  }
  PadIdx_t* neighbors_ = new PadIdx_t[MaxNeighbors * N];
  double factor = (2 * kernelSize - 1);
  for (PadIdx_t i = 0; i < N; i++) {
    PadIdx_t* i_neigh = getNeighborListOf(neighbors_, i);
    double xTerm = factor * DX[i] + eps;
    double yTerm = factor * DY[i] + eps;
    // Search neighbors of i
    for (PadIdx_t j = 0; j < N; j++) {
      int xMask0 =
        (fabs(X[i] - X[j]) < (xTerm + DX[j]));
      int yMask0 = 0;
      if (xMask0) {
        yMask0 = (fabs(Y[i] - Y[j]) < (yTerm + DY[j]));
      }
      if (xMask0 && yMask0) {
        *i_neigh = j;
        i_neigh++;
        // Check
        // printf( "pad %d neighbor %d xMask=%d yMask=%d\n", i, j, (xMask0 &&
        // yMask0), (xMask1 && yMask1));
      }
    }
    // Set the End of list
    *i_neigh = -1;
    //
    if (CHECK &&
        (fabs(i_neigh - getNeighborListOf(neighbors_, i)) > MaxNeighbors)) {
      printf("Pad %d : nbr of neighbours %ld greater than the limit %d \n", i,
             i_neigh - getNeighborListOf(neighbors_, i), MaxNeighbors);
      throw std::overflow_error("Not enough allocation");
    }
  }

  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    Pads::printNeighbors(neighbors_, N);
  }

  return neighbors_;
}

Pads* Pads::addBoundaryPads()
{
  double eps = epsilonGeometry;
  //
  std::vector<double> bX;
  std::vector<double> bY;
  std::vector<double> bdX;
  std::vector<double> bdY;
  int N = nPads;
  // Build neigbors if required
  PadIdx_t* neigh = getFirstNeighbors();
  for (int i = 0; i < N; i++) {
    bool east = true, west = true, north = true, south = true;
    for (const PadIdx_t* neigh_ptr = getTheFirtsNeighborOf(neigh, i);
         *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t v = *neigh_ptr;
      // If neighbours then no boundary pads to add
      double xDelta = (x[v] - x[i]);
      if (fabs(xDelta) > eps) {
        if (xDelta > 0) {
          east = false;
        } else {
          west = false;
        }
      }
      double yDelta = (y[v] - y[i]);
      if (fabs(yDelta) > eps) {
        if (yDelta > 0) {
          north = false;
        } else {
          south = false;
        }
      }
    }
    // Add new pads
    if (east) {
      bX.push_back(x[i] + 2 * dx[i]);
      bY.push_back(y[i]);
      bdX.push_back(dx[i]);
      bdY.push_back(dy[i]);
    }
    if (west) {
      bX.push_back(x[i] - 2 * dx[i]);
      bY.push_back(y[i]);
      bdX.push_back(dx[i]);
      bdY.push_back(dy[i]);
    }
    if (north) {
      bX.push_back(x[i]);
      bY.push_back(y[i] + 2 * dy[i]);
      bdX.push_back(dx[i]);
      bdY.push_back(dy[i]);
    }
    if (south) {
      bX.push_back(x[i]);
      bY.push_back(y[i] - 2 * dy[i]);
      bdX.push_back(dx[i]);
      bdY.push_back(dy[i]);
    }
  }
  // Suppress new pads which overlaps
  int nPadToAdd = bX.size();
  double error = epsilonGeometry;
  int K = bX.size();
  Mask_t toKeep[K];
  vectorSetShort(toKeep, 1, K);
  double xInf[K], xSup[K], yInf[K], ySup[K];
  double maxInf[K], minSup[K];
  double xOverlap[K], yOverlap[K];
  double overlap;
  // Compute x/y inf/sup
  for (int k = 0; k < K; k++) {
    xInf[k] = bX[k] - bdX[k];
    xSup[k] = bX[k] + bdX[k];
    yInf[k] = bY[k] - bdY[k];
    ySup[k] = bY[k] + bdY[k];
  }
  // printf("[addBoundary] n=%d boundary pads added\n", K);

  for (int k = 0; k < (K - 1); k++) {
    if (toKeep[k]) {
      // X overlap
      vectorMaxScalar(&xInf[k + 1], xInf[k], K - k - 1, &maxInf[k + 1]);
      vectorMinScalar(&xSup[k + 1], xSup[k], K - k - 1, &minSup[k + 1]);
      vectorAddVector(&minSup[k + 1], -1.0, &maxInf[k + 1], K - k - 1, &xOverlap[k + 1]);
      // Y overlap
      vectorMaxScalar(&yInf[k + 1], yInf[k], K - k - 1, &maxInf[k + 1]);
      vectorMinScalar(&ySup[k + 1], ySup[k], K - k - 1, &minSup[k + 1]);
      vectorAddVector(&minSup[k + 1], -1.0, &maxInf[k + 1], K - k - 1, &yOverlap[k + 1]);

      for (int l = k + 1; l < K; l++) {
        // printf("             xOverlap[l]=%f, yOverlap[l]=%f\n", xOverlap[l], yOverlap[l]);
        overlap = (xOverlap[l] < error) ? 0.0 : 1.0;
        overlap = (yOverlap[l] < error) ? 0.0 * overlap : overlap * 1;
        if (toKeep[l] && (overlap > 0.0)) {
          toKeep[l] = 0;
          nPadToAdd--;
          // printf("[addBoundary] overlapping k=%d l=%d \n", k, l);
          // printf("              pad k x=%f, dx=%f, y=%f, dy=%f\n", bX[k], bdX[k], bY[k], bdY[k]);
          // printf("              pad l x=%f, dx=%f, y=%f, dy=%f\n", bX[l], bdX[l], bY[l], bdY[l]);
          //
          // Update boundary Pads
          double infxy_ = bX[k] - bdX[k];
          double supxy_ = bX[k] + bdX[k];
          infxy_ = std::fmax(infxy_, xInf[l]);
          supxy_ = std::fmin(supxy_, xSup[l]);
          double dxy_ = 0.5 * (supxy_ - infxy_);
          // pad center : xInf + 0.5 dx
          bX[k] = infxy_ + dxy_;
          bdX[k] = dxy_;
          //
          // The same for Y
          infxy_ = bY[k] - bdY[k];
          supxy_ = bY[k] + bdY[k];
          infxy_ = std::fmax(infxy_, yInf[l]);
          supxy_ = std::fmin(supxy_, ySup[l]);
          dxy_ = 0.5 * (supxy_ - infxy_);
          bY[k] = infxy_ + dxy_;
          bdY[k] = dxy_;
          // printf("              new pad k x=%f, dx=%f, y=%f, dy=%f\n", bX[k], bdX[k], bY[k], bdY[k]);
        }
      }
    } // if (toKeep[k])
  }
  if (clusterConfig.processingLog >= clusterConfig.info) {
    printf("[addBoundary] n=%d final boundary pads added, %d removed overlapping pads\n", nPadToAdd, K - nPadToAdd);
  }

  int nTotalPads = N + nPadToAdd;
  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    printf("nTotalPads=%d, nPads=%d,  nPadToAdd=%d\n", nTotalPads, N,
           nPadToAdd);
  }
  Pads* padsWithBoundaries = new Pads(nTotalPads, chamberId);
  Pads* newPads = padsWithBoundaries;
  for (int i = 0; i < N; i++) {
    newPads->x[i] = x[i];
    newPads->y[i] = y[i];
    newPads->dx[i] = dx[i];
    newPads->dy[i] = dy[i];
    newPads->q[i] = q[i];
    newPads->saturate[i] = saturate[i];
  }
  newPads->nObsPads = N;
  for (int i = N, k = 0; i < nTotalPads; k++) {
    if (toKeep[k]) {
      newPads->x[i] = bX[k];
      newPads->y[i] = bY[k];
      newPads->dx[i] = bdX[k];
      newPads->dy[i] = bdY[k];
      newPads->q[i] = 0.0;
      newPads->saturate[i] = 0;
      i++;
    }
  }
  newPads->totalCharge = totalCharge;
  //
  // printPads( "[addBoundary] pads", *newPads );
  return padsWithBoundaries;
}

Pads::Pads(int N, int chId, PadMode mode_)
{
  nPads = N;
  nObsPads = N;
  mode = mode_;
  chamberId = chId;
  allocate();
}

/* Old Version: not used
// Merge the 2 pad sets
// Remark : pads0, pads1 correspond respectively
//  to the cath-plane 0, 1
Pads::Pads( const Pads *pads0, const Pads *pads1) {
  int n0 = Pads::getNbrOfPads(pads0);
  int n1 = Pads::getNbrOfPads(pads1);
  nPads = n0 + n1;
  if ( n0 != 0 ) {
    chamberId = pads0->chamberId;
    mode = pads0->mode;
  }
  else {
    chamberId = pads1->chamberId;
    mode = pads1->mode;
  }
  allocate();
  totalCharge = 0.0;
  // X, Y, dX, dY, q
  if( n0 ) {
    vectorCopy(pads0->x, n0, x);
    vectorCopy(pads0->y, n0, y);
    vectorCopy(pads0->dx, n0, dx);
    vectorCopy(pads0->dy, n0, dy);
    vectorCopy(pads0->q, n0, q);
   // saturate pads
    vectorCopyShort( pads0->saturate, n0, saturate);
    totalCharge += pads0->totalCharge;
 }
 if (n1) {
    vectorCopy(pads1->x, n1, &x[n0]);
    vectorCopy(pads1->y, n1, &y[n0]);
    vectorCopy(pads1->dx, n1, &dx[n0]);
    vectorCopy(pads1->dy, n1, &dy[n0]);
    vectorCopy(pads1->q, n1, &q[n0]);
  // saturate pads
    vectorCopyShort( pads1->saturate, n1, &saturate[n0]);
    totalCharge += pads1->totalCharge;

  }
  // Cathode plane
  vectorSetShort( cath, 0, n0 );
  vectorSetShort( &cath[n0], 1, n1 );
}
*/

// Over allocation
Pads::Pads(const Pads* pads, int size)
{
  nPads = pads->nPads;
  nObsPads = pads->nObsPads;
  mode = pads->mode;
  chamberId = pads->chamberId;
  totalCharge = pads->totalCharge;
  int size_ = (size < nPads) ? nPads : size;
  allocate(size_);
  memcpy(x, pads->x, sizeof(double) * nPads);
  memcpy(y, pads->y, sizeof(double) * nPads);
  memcpy(dx, pads->dx, sizeof(double) * nPads);
  memcpy(dy, pads->dy, sizeof(double) * nPads);
  memcpy(q, pads->q, sizeof(double) * nPads);
  if (pads->saturate != nullptr) {
    memcpy(saturate, pads->saturate, sizeof(Mask_t) * nPads);
  }
  if (pads->cath != nullptr) {
    memcpy(cath, pads->cath, sizeof(Mask_t) * nPads);
  }
}

Pads::Pads(const Pads& pads, PadMode mode_)
{
  nPads = pads.nPads;
  nObsPads = pads.nObsPads;
  mode = mode_;
  chamberId = pads.chamberId;
  totalCharge = pads.totalCharge;
  allocate();
  if (mode == pads.mode) {
    memcpy(x, pads.x, sizeof(double) * nPads);
    memcpy(y, pads.y, sizeof(double) * nPads);
    memcpy(dx, pads.dx, sizeof(double) * nPads);
    memcpy(dy, pads.dy, sizeof(double) * nPads);
    memcpy(q, pads.q, sizeof(double) * nPads);
  } else if (mode == PadMode::xydxdyMode) {
    //  xyInfSupMode ->  xydxdyMode
    padBoundsToCenter(pads);
    memcpy(q, pads.q, sizeof(double) * nPads);
  } else {
    // xydxdyMode -> xyInfSupMode
    padCenterToBounds(pads);
    memcpy(q, pads.q, sizeof(double) * nPads);
  }
  if (pads.saturate) {
    memcpy(saturate, pads.saturate, sizeof(Mask_t) * nPads);
  }
  if (pads.cath) {
    memcpy(cath, pads.cath, sizeof(Mask_t) * nPads);
  }
}

Pads::Pads(const Pads& pads, const Mask_t* mask)
{
  nPads = vectorSumShort(mask, pads.nPads);
  mode = PadMode::xydxdyMode;
  chamberId = pads.chamberId;
  allocate();

  vectorGather(pads.x, mask, pads.nPads, x);
  vectorGather(pads.y, mask, pads.nPads, y);
  vectorGather(pads.dx, mask, pads.nPads, dx);
  vectorGather(pads.dy, mask, pads.nPads, dy);
  vectorGather(pads.q, mask, pads.nPads, q);
  if (pads.saturate) {
    vectorGatherShort(pads.saturate, mask, pads.nPads, saturate);
  }
  if (pads.cath) {
    vectorGatherShort(pads.cath, mask, pads.nPads, cath);
  }
  totalCharge = vectorSum(q, nPads);
  nObsPads = nPads;
}

/* Old version: Unused
Pads::Pads( const double *x_, const double *y_, const double *dx_, const double
*dy_, const double *q_, const Mask_t *saturate_, int chId, int nPads_) { mode =
xydxdyMode; nPads = nPads_; chamberId = chId; allocate();
  // Copy pads
  memcpy ( x, x_, sizeof(double)*nPads );
  memcpy ( y, y_, sizeof(double)*nPads);
  memcpy ( dx, dx_, sizeof(double)*nPads );
  memcpy ( dy, dy_, sizeof(double)*nPads );
  memcpy ( q, q_, sizeof(double)*nPads );
  if( saturate_ != nullptr ) {
    memcpy ( saturate, saturate_, sizeof(Mask_t)*nPads );
  }
}
*/

// Take the ownership of coordinates (x, y, dx, dy)
Pads::Pads(double* x_, double* y_, double* dx_, double* dy_, int chId,
           int nPads_)
{
  mode = PadMode::xydxdyMode;
  nPads = nPads_;
  nObsPads = nPads;
  chamberId = chId;
  x = x_;
  y = y_;
  dx = dx_;
  dy = dy_;
  q = new double[nPads];
  // Set null Charge
  vectorSetZero(q, nPads);
  // others
  saturate = nullptr;
  cath = nullptr;
  neighbors = nullptr;
  totalCharge = 0;
}

Pads::Pads(const double* x_, const double* y_, const double* dx_,
           const double* dy_, const double* q_, const short* cathode,
           const Mask_t* saturate_, short selectedCath, int chId,
           PadIdx_t* mapCathPadIdxToPadIdx, int nAllPads)
{
  mode = PadMode::xydxdyMode;
  int nCathode1 = vectorSumShort(cathode, nAllPads);
  nPads = nCathode1;
  if (selectedCath == 0) {
    nPads = nAllPads - nCathode1;
  }
  nObsPads = nPads;
  chamberId = chId;
  allocate();
  double qSum = 0;
  // Copy pads
  int k = 0;
  for (int i = 0; i < nAllPads; i++) {
    if (cathode[i] == selectedCath) {
      x[k] = x_[i];
      y[k] = y_[i];
      dx[k] = dx_[i];
      dy[k] = dy_[i];
      q[k] = q_[i];
      qSum += q_[i];
      saturate[k] = saturate_[i];
      mapCathPadIdxToPadIdx[k] = i;
      k++;
    }
  }
  totalCharge = qSum;
}

Pads::Pads(const double* x_, const double* y_, const double* dx_,
           const double* dy_, const double* q_, const short* cathode,
           const Mask_t* saturate_, int chId, int nAllPads)
{
  mode = PadMode::xydxdyMode;
  // int nCathode1 = vectorSumShort(cathode, nAllPads);
  nPads = nAllPads;
  nObsPads = nPads;
  /*
  if (selectedCath == 0) {
    nPads = nAllPads - nCathode1;
  }
  */
  chamberId = chId;
  allocate();
  double qSum = 0;
  // Copy
  vectorCopy(x_, nPads, x);
  vectorCopy(y_, nPads, y);
  vectorCopy(dx_, nPads, dx);
  vectorCopy(dy_, nPads, dy);
  vectorCopy(q_, nPads, q);
  vectorCopyShort(cathode, nPads, cath);
  vectorCopyShort(saturate_, nPads, saturate);
  totalCharge = vectorSum(q, nPads);
}

// Concatenate pads
Pads::Pads(const Pads* pads0, const Pads* pads1, PadMode mode_)
{

  // Take Care: pads0 and pads2 must be in xydxdyMode
  bool padsMode = (pads0 == nullptr) ? true : (pads0->mode == PadMode::xydxdyMode);
  padsMode = (pads1 == nullptr) ? padsMode : (pads1->mode == PadMode::xydxdyMode);
  if (!padsMode) {
    throw std::out_of_range("Pads:: bad mode (xydxdyMode required) for pad merging");
  }

  int N0 = (pads0 == nullptr) ? 0 : pads0->nPads;
  int N1 = (pads1 == nullptr) ? 0 : pads1->nPads;
  int nObs0 = (pads0 == nullptr) ? 0 : pads0->nObsPads;
  int nObs1 = (pads1 == nullptr) ? 0 : pads1->nObsPads;
  nPads = N0 + N1;
  chamberId = (N0) ? pads0->chamberId : pads1->chamberId;
  allocate();
  // Copy observable pads0
  int destIdx = 0;
  copyPads(pads0, 0, destIdx, nObs0, 0);
  destIdx += nObs0;
  // Copy observable pads1
  copyPads(pads1, 0, destIdx, nObs1, 1);
  destIdx += nObs1;

  // Boundary pads0
  int n = N0 - nObs0;
  copyPads(pads0, nObs0, destIdx, n, 0);
  destIdx += n;
  n = N1 - nObs1;
  copyPads(pads1, nObs1, destIdx, n, 1);
  destIdx += n;

  /*
  if (N1) {
    memcpy(x, pads1->x, sizeof(double) * N1);
    memcpy(y, pads1->y, sizeof(double) * N1);
    memcpy(dx, pads1->dx, sizeof(double) * N1);
    memcpy(dy, pads1->dy, sizeof(double) * N1);
    memcpy(q, pads1->q, sizeof(double) * N1);
    memcpy(saturate, pads1->saturate, sizeof(Mask_t) * N1);
    vectorSetShort(cath, 0, N1);
  }
  if (N2) {
    // Copy pads2
    memcpy(&x[N1], pads2->x, sizeof(double) * N2);
    memcpy(&y[N1], pads2->y, sizeof(double) * N2);
    memcpy(&dx[N1], pads2->dx, sizeof(double) * N2);
    memcpy(&dy[N1], pads2->dy, sizeof(double) * N2);
    memcpy(&q[N1], pads2->q, sizeof(double) * N2);
    memcpy(&saturate[N1], pads2->saturate, sizeof(Mask_t) * N2);
    vectorSetShort(&cath[N1], 1, N2);
  }
  */
  // ??? printPads(" Before InfSup", *this);
  if (mode_ == PadMode::xyInfSupMode) {
    padCenterToBounds();
  }
  totalCharge = vectorSum(q, nPads);
  nObsPads = nObs0 + nObs1;
  // ??? printPads(" after InfSup", *this);
}
/*
void Pads::print(const char *title)
{
  printf("%s\n", title);
  printf("print pads nPads=%4d nObsPads=%4d mode=%1d\n", nPads, nObsPads, mode);
  printf("idx      x       y       dx        dy cath  sat  charge \n");
  for (int i=0; i < nPads; i++) {
    printf("%2d %7.3f %7.3f %7.3f %7.3f    %1d     %1d %7.3f \n", i, x[i], y[i], dx[i], dy[i], cath[i], saturate[i], q[i]);
  }
}
*/

Pads* Pads::selectPads(int* index, int K)
{
  Pads* sPads = new Pads(K, chamberId, mode);
  int k0 = 0;
  for (int k = 0; k < K; k++) {
    int idx = index[k];
    sPads->x[k0] = x[idx];
    sPads->y[k0] = y[idx];
    sPads->dx[k0] = dx[idx];
    sPads->dy[k0] = dy[idx];
    sPads->q[k0] = q[idx];
    sPads->saturate[k0] = saturate[idx];
    k0++;
  }
  sPads->nPads = K;
  sPads->nObsPads = K;
  return sPads;
}

// ??? removePad can be suppressed ????
void Pads::removePad(int index)
{
  if (nObsPads != nPads) {
    throw std::out_of_range("Pads::removePad: bad usage");
  }
  if ((index < 0) || (index >= nPads)) {
    return;
  }
  int nItems = nPads - index;
  if (index == nPads - 1) {
    nPads = nPads - 1;
    return;
  }
  vectorCopy(&x[index + 1], nItems, &x[index]);
  vectorCopy(&y[index + 1], nItems, &y[index]);
  vectorCopy(&dx[index + 1], nItems, &dx[index]);
  vectorCopy(&dy[index + 1], nItems, &dy[index]);
  //
  vectorCopy(&q[index + 1], nItems, &q[index]);
  vectorCopyShort(&saturate[index + 1], nItems, &saturate[index]);
  //
  nPads = nPads - 1;
  nObsPads = nObsPads - 1;
}

void Pads::allocate()
{
  // Note: Must be deallocated/releases if required
  x = nullptr;
  y = nullptr;
  dx = nullptr;
  dy = nullptr;
  saturate = nullptr;
  q = nullptr;
  neighbors = nullptr;
  int N = nPads;
  x = new double[N];
  y = new double[N];
  dx = new double[N];
  dy = new double[N];
  saturate = new Mask_t[N];
  cath = new Mask_t[N];
  q = new double[N];
}

// Over-allocation of pads
void Pads::allocate(int size)
{
  // Note: Must be deallocated/releases if required
  x = nullptr;
  y = nullptr;
  dx = nullptr;
  dy = nullptr;
  saturate = nullptr;
  q = nullptr;
  neighbors = nullptr;
  // N nbr of pads used
  int N = nPads;
  // size allocation
  x = new double[size];
  y = new double[size];
  dx = new double[size];
  dy = new double[size];
  saturate = new Mask_t[size];
  cath = new Mask_t[size];
  q = new double[size];
}

void Pads::copyPads(const Pads* srcPads, int srcIdx, int dstIdx, int N, Mask_t cathValue)
{
  if (N) {
    memcpy(&x[dstIdx], &srcPads->x[srcIdx], sizeof(double) * N);
    memcpy(&y[dstIdx], &srcPads->y[srcIdx], sizeof(double) * N);
    memcpy(&dx[dstIdx], &srcPads->dx[srcIdx], sizeof(double) * N);
    memcpy(&dy[dstIdx], &srcPads->dy[srcIdx], sizeof(double) * N);
    memcpy(&q[dstIdx], &srcPads->q[srcIdx], sizeof(double) * N);
    memcpy(&saturate[dstIdx], &srcPads->saturate[srcIdx], sizeof(Mask_t) * N);
    vectorSetShort(&cath[dstIdx], cathValue, N);
  }
}
double Pads::updateTotalCharge()
{
  totalCharge = vectorSum(q, nPads);
  return totalCharge;
}
//
double Pads::getMeanTotalCharge()
{
  double meanCharge;
  if (cath != nullptr) {
    int nCath1 = vectorSumShort(cath, nPads);
    int nCath0 = nPads - nCath1;
    int nCath = (nCath0 > 0) + (nCath1 > 0);
    meanCharge = totalCharge / nCath;
  } else {
    meanCharge = totalCharge;
  }
  return meanCharge;
}

void Pads::setCharges(double c)
{
  vectorSet(q, c, nPads);
  totalCharge = c * nPads;
}

void Pads::setCharges(double* q_, int n)
{
  vectorCopy(q_, n, q);
  totalCharge = vectorSum(q_, n);
}

void Pads::setCathodes(Mask_t cath_) { vectorSetShort(cath, cath_, nPads); }

void Pads::setSaturate(Mask_t val) { vectorSetShort(saturate, val, nPads); }

void Pads::setToZero()
{
  for (int i = 0; i < nPads; i++) {
    x[i] = 0.0;
    y[i] = 0.0;
    dx[i] = 0.0;
    dy[i] = 0.0;
    q[i] = 0.0;
  }
}

int Pads::removePads(double qCut)
{
  if (nObsPads != nPads) {
    throw std::out_of_range("Pads::removePad: bad usage");
  }
  double qSum = 0.0;
  int k = 0;
  for (int i = 0; i < nPads; i++) {
    // printf("q %f\n", q[i]);
    if (q[i] >= qCut) {
      qSum += q[i];
      q[k] = q[i];
      x[k] = x[i];
      y[k] = y[i];
      dx[k] = dx[i];
      dy[k] = dy[i];
      k++;
    }
  }
  totalCharge = qSum;
  nPads = k;
  nObsPads = k;
  return k;
}

void Pads::normalizeCharges()
{
  for (int i = 0; i < nPads; i++) {
    q[i] = q[i] / totalCharge;
  }
}

// Build the neighbor list
PadIdx_t* Pads::getFirstNeighbors()
{
  int N = nPads;
  if (neighbors == nullptr) {
    neighbors = buildFirstNeighbors(x, y, dx, dy, N);
  }
  return neighbors;
}

int Pads::addIsolatedPadInGroups(Mask_t* cathToGrp, Mask_t* grpToGrp,
                                 int nGroups)
{
  int nNewGroups = 0;
  if (nPads == 0) {
    return nGroups;
  }

  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    printf("[addIsolatedPadInGroups]  nGroups=%d\n", nGroups);
    vectorPrintShort("  cathToGrp input", cathToGrp, nPads);
  }
  PadIdx_t* neigh = getFirstNeighbors();

  for (int p = 0; p < nPads; p++) {
    if (cathToGrp[p] == 0) {
      // Neighbors
      //
      int q = -1;
      for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, p); *neigh_ptr != -1;
           neigh_ptr++) {
        q = *neigh_ptr;
        // printf("  Neigh of %d: %d\n", p, q);
        if (cathToGrp[q] != 0) {
          if (cathToGrp[p] == 0) {
            // Propagation
            cathToGrp[p] = cathToGrp[q];
            // printf("    Neigh=%d: Propagate the grp=%d of the neighbor to
            // p=%d\n", q, cathToGrp[q], p);
          } else if (cathToGrp[p] != cathToGrp[q]) {
            // newCathToGrp[p] changed
            // Fuse Grp
            Mask_t minGrp = cathToGrp[p];
            Mask_t maxGrp = cathToGrp[q];
            if (cathToGrp[p] > cathToGrp[q]) {
              minGrp = cathToGrp[q];
              maxGrp = cathToGrp[p];
            }
            grpToGrp[maxGrp] = minGrp;
            // printf("    Neigh=%d: Fuse the grp=%d of the neighbor with
            // p-Group=%d\n", q, cathToGrp[q], cathToGrp[p]); Update
            cathToGrp[p] = minGrp;
          }
        }
      }
      if (cathToGrp[p] == 0) {
        // New Group
        nGroups++;
        nNewGroups++;
        cathToGrp[p] = nGroups;
        // printf("    Grp-isolated pad p=%d, new grp=%d \n", p, nGroups);
      }
    }
  }

  // Finish the Fusion
  for (int g = 0; g < (nGroups + 1); g++) {
    Mask_t gBar = g;
    while (gBar != grpToGrp[gBar]) {
      gBar = grpToGrp[gBar];
    }
    // Terminal Grp :  gBar = grpToGrp[gBar]
    grpToGrp[g] = gBar;
  }

  if (clusterConfig.padMappingLog >= clusterConfig.debug) {
    printf("  grpToGrp\n");
    for (int g = 0; g < (nGroups + 1); g++) {
      printf("  %d -> %d\n", g, grpToGrp[g]);
    }
  }
  // Apply group to Pads
  for (int p = 0; p < nPads; p++) {
    cathToGrp[p] = grpToGrp[cathToGrp[p]];
  }
  // Save in grpToGrp
  vectorCopyShort(grpToGrp, (nGroups + 1), grpToGrp);
  //
  return nNewGroups;
}

void Pads::release()
{
  if (x != nullptr) {
    delete[] x;
    x = nullptr;
  }
  if (y != nullptr) {
    delete[] y;
    y = nullptr;
  }
  if (dx != nullptr) {
    delete[] dx;
    dx = nullptr;
  }
  if (dy != nullptr) {
    delete[] dy;
    dy = nullptr;
  }

  if (q != nullptr) {
    delete[] q;
    q = nullptr;
  }
  if (cath != nullptr) {
    delete[] cath;
    cath = nullptr;
  }
  if (saturate != nullptr) {
    delete[] saturate;
    saturate = nullptr;
  }
  deleteInt(neighbors);
  nPads = 0;
  nObsPads = 0;
}

// Refine on/around localMax
void Pads::refineLocalMaxAndUpdateCij(const Pads& pads,
                                      std::vector<PadIdx_t>& pixToRefineIdx, double Cij[])
{

  // Take care : here all pads data describe the pixels
  // Number of Pixels
  int K = nPads;
  // Number of Pads
  int N = pads.getNbrOfPads();

  const double* xInf = pads.getXInf();
  const double* yInf = pads.getYInf();
  const double* xSup = pads.getXSup();
  const double* ySup = pads.getYSup();
  int chId = pads.getChamberId();

  double cut = -1;
  int count = N;
  //
  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    vectorPrint("Pads::refinePads", q, N);
    printf("Pads::refinePads count(new nPads)=%d\n", count);
  }

  double* xWestIntegrals = new double[N];
  double* xEastIntegrals = new double[N];
  double* yNorthIntegrals = new double[N];
  double* ySouthIntegrals = new double[N];
  int axe;
  double totalChargeInc = 0.0;
  int k = K;
  for (int i = 0; i < pixToRefineIdx.size(); i++) {
    int pixelMaxIdx = pixToRefineIdx[i];
    // printf("Refine pixel i=%d, q[pixelMaxIdx]=%f saturate[pixelMaxIdx]=%d\n", i, q[pixelMaxIdx], saturate[pixelMaxIdx]);
    //
    // saturate is used to tag pixels already refined
    if (saturate[pixelMaxIdx] == 0) {

      saturate[pixelMaxIdx] = 1;
      double xOld = x[pixelMaxIdx];
      double yOld = y[pixelMaxIdx];
      double dxOld = dx[pixelMaxIdx];
      double dyOld = dy[pixelMaxIdx];
      double qOld = q[pixelMaxIdx];
      totalChargeInc += (3 * qOld);
      // NW
      // Done in place (same pixel index)
      x[pixelMaxIdx] = xOld - 0.5 * dxOld;
      y[pixelMaxIdx] = yOld + 0.5 * dyOld;
      dx[pixelMaxIdx] = 0.5 * dxOld;
      dy[pixelMaxIdx] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[pixelMaxIdx] = qOld;
      // Update Cij
      axe = 0;
      compute1DPadIntegrals(xInf, xSup, N, x[pixelMaxIdx], axe, chId, xWestIntegrals);
      axe = 1;
      compute1DPadIntegrals(yInf, ySup, N, y[pixelMaxIdx], axe, chId, yNorthIntegrals);
      // 2D Integral
      vectorMultVector(xWestIntegrals, yNorthIntegrals, N, &Cij[N * pixelMaxIdx]);
      // k++;

      // NE
      x[k] = xOld + 0.5 * dxOld;
      y[k] = yOld + 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      // Update Cij
      axe = 0;
      compute1DPadIntegrals(xInf, xSup, N, x[k], axe, chId, xEastIntegrals);
      vectorMultVector(xEastIntegrals, yNorthIntegrals, N, &Cij[N * k]);
      k++;

      // SW
      x[k] = xOld - 0.5 * dxOld;
      y[k] = yOld - 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      // Update Cij
      axe = 1;
      compute1DPadIntegrals(yInf, ySup, N, y[k], axe, chId, ySouthIntegrals);
      vectorMultVector(xWestIntegrals, ySouthIntegrals, N, &Cij[N * k]);
      k++;

      // SE
      x[k] = xOld + 0.5 * dxOld;
      y[k] = yOld - 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      // Update Cij
      vectorMultVector(xEastIntegrals, ySouthIntegrals, N, &Cij[N * k]);
      k++;
      nPads += 3;
    }
  }
  totalCharge += totalChargeInc;
  nObsPads = nPads;
  delete[] xWestIntegrals;
  delete[] xEastIntegrals;
  delete[] yNorthIntegrals;
  delete[] ySouthIntegrals;
}

// refinement on locam mwxima
// use for pixels
// Old version (without Cij update)
// To keep ???
void Pads::refineLocalMax(Pads& localMax, std::vector<PadIdx_t>& localMaxIdx)
{
  // ??? LocalMax not used except for the cutoff

  // Take care : here all pads data describe the pixels
  int N = nPads;
  int nLocalMax = localMax.getNbrOfPads();
  /* qCut : not used
  // Count pad such as q > 4 * pixCutOf
  int count=0;
  double cut = 0.2;
  for (int i=0; i < N; i++) {
    if ( q[i] > cut ) {
      count++;
    }
  }
  */
  // printf("nPixels=%d nLocalMax=%d localMaxIdx.size=%lu\n", N, nLocalMax, localMaxIdx.size());
  double cut = -1;
  int count = N;
  //
  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    vectorPrint("Pads::refinePads", q, N);
    printf("Pads::refinePads count(new nPads)=%d\n", count);
  }

  double totalChargeInc = 0.0;
  int k = N;
  for (int i = 0; i < localMaxIdx.size(); i++) {
    int pixelMaxIdx = localMaxIdx[i];
    // saturate is used to tag pixels already refined
    printf("Refinement i=%d, localMax.q[i]=%f saturate[pixelMaxIdx]=%d\n", i, localMax.q[i], saturate[pixelMaxIdx]);
    if ((localMax.q[i] > cut) && (saturate[pixelMaxIdx] == 0)) {
      saturate[pixelMaxIdx] = 1;
      double xOld = x[pixelMaxIdx];
      double yOld = y[pixelMaxIdx];
      double dxOld = dx[pixelMaxIdx];
      double dyOld = dy[pixelMaxIdx];
      double qOld = q[pixelMaxIdx];
      printf("refine on pixel %d\n", pixelMaxIdx);
      totalChargeInc += (3 * qOld);
      // NW
      // Done in place (same pixel index)
      x[pixelMaxIdx] = xOld - 0.5 * dxOld;
      y[pixelMaxIdx] = yOld + 0.5 * dyOld;
      dx[pixelMaxIdx] = 0.5 * dxOld;
      dy[pixelMaxIdx] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[pixelMaxIdx] = qOld;
      // k++;

      // NE
      x[k] = xOld + 0.5 * dxOld;
      y[k] = yOld + 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      k++;

      // SW
      x[k] = xOld - 0.5 * dxOld;
      y[k] = yOld - 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      k++;

      // SE
      x[k] = xOld + 0.5 * dxOld;
      y[k] = yOld - 0.5 * dyOld;
      dx[k] = 0.5 * dxOld;
      dy[k] = 0.5 * dyOld;
      // rPads->q[k] = 0.25 * qOld;
      q[k] = qOld;
      saturate[k] = 1;
      k++;
      nPads += 3;
    }
  }
  totalCharge += totalChargeInc;
  // nPads = N+3*nLocalMax;
  nObsPads = nPads;
  // return rPads;
}

Pads* Pads::refineAll()
{
  int N = nPads;
  /* qCut : not used
  // Count pad such as q > 4 * pixCutOf
  int count=0;
  double cut = 0.2;
  for (int i=0; i < N; i++) {
    if ( q[i] > cut ) {
      count++;
    }
  }
  */
  double cut = -1;
  int count = N;
  //
  if (clusterConfig.padMappingLog >= clusterConfig.detail) {
    vectorPrint("Pads::refinePads", q, N);
    printf("Pads::refinePads count(new nPads)=%d\n", count);
  }
  Pads* rPads = new Pads(count * 4, chamberId);
  int k = 0;
  for (int i = 0; i < N; i++) {
    if (q[i] > cut) {
      // NW
      rPads->x[k] = x[i] - 0.5 * dx[i];
      rPads->y[k] = y[i] + 0.5 * dy[i];
      rPads->dx[k] = 0.5 * dx[i];
      rPads->dy[k] = 0.5 * dy[i];
      // rPads->q[k] = 0.25 * q[i];
      rPads->q[k] = q[i];
      k++;

      // NE
      rPads->x[k] = x[i] + 0.5 * dx[i];
      rPads->y[k] = y[i] + 0.5 * dy[i];
      rPads->dx[k] = 0.5 * dx[i];
      rPads->dy[k] = 0.5 * dy[i];
      // rPads->q[k] = 0.25 * q[i];
      rPads->q[k] = q[i];
      k++;

      // SW
      rPads->x[k] = x[i] - 0.5 * dx[i];
      rPads->y[k] = y[i] - 0.5 * dy[i];
      rPads->dx[k] = 0.5 * dx[i];
      rPads->dy[k] = 0.5 * dy[i];
      // rPads->q[k] = 0.25 * q[i];
      rPads->q[k] = q[i];
      k++;

      // SE
      rPads->x[k] = x[i] + 0.5 * dx[i];
      rPads->y[k] = y[i] - 0.5 * dy[i];
      rPads->dx[k] = 0.5 * dx[i];
      rPads->dy[k] = 0.5 * dy[i];
      // rPads->q[k] = 0.25 * q[i];
      rPads->q[k] = q[i];
      k++;
    }
  }
  rPads->totalCharge = 4 * totalCharge;
  return rPads;
}

Pads* Pads::extractLocalMaxOnCoarsePads(std::vector<PadIdx_t>& localMaxIdx)
{
  if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
    printf("  - Pads::extractLocalMax on Coarse Pads(extractLocalMax nPads=%d)\n", nPads);
  }
  double qMax = vectorMax(q, nPads);
  //
  // TO DO ??? Compute the neighbors once
  // between to refinements
  if (neighbors != nullptr) {
    delete[] neighbors;
  }
  // 4(5) neighbors
  neighbors = getFirstNeighbors();
  PadIdx_t* neigh = neighbors;
  // printNeighbors( neigh, nPads);
  //
  // Part I - Morphologic Laplacian operator
  //
  double morphLaplacian[nPads];
  double laplacian[nPads];
  double weight[nPads];
  vectorSet(morphLaplacian, -1.0, nPads);
  // Invalid the neighbors of a local max
  Mask_t alreadyDone[nPads];
  vectorSetZeroShort(alreadyDone, nPads);
  std::vector<PadIdx_t> newPixelIdx;
  bool less;
  for (int i = 0; i < nPads; i++) {
    if (alreadyDone[i] == 0) {
      int nLess = 0;
      int count = 0;
      laplacian[i] = 0.0;
      weight[i] = 0.0;
      for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
           neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        // Morphologic Laplacian
        // nLess += (q[v] < q[i]);
        less = (q[v] <= q[i]);
        count++;
        if (less) {
          nLess++;
          // Laplacian
          double cst;
          cst = (i == v) ? 1.0 : -0.25;
          laplacian[i] += cst * q[v];
          weight[i] += q[v];
        }
      }
      // Invalid ?? morphLaplacian[i] = double(nLess) / (count - 1);
      morphLaplacian[i] = double(nLess) / count;
      //
      if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
        printf(
          "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
          "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
          i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
          weight[i]);
      }
      if (morphLaplacian[i] >= 1.0) {
        //  Local max charge must be higher than 1.5 % of the max and
        //  the curvature must be greater than 50% of the peak
        // Inv ??? if ((q[i] > 0.015 * qMax) || (fabs(laplacian[i]) > (0.5 * q[i]))) {
        if (q[i] > 0.015 * qMax) {
          newPixelIdx.push_back(i);
          // if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
          if (0) {
            printf(
              "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
              "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
              i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
              weight[i]);
            printf("  Selected %d\n", i);
          }
        }
        // Invalid the neihbors
        // they can't be a maximun
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
             neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          alreadyDone[v] += 1;
          /*
          if (q[v] > 0.5 * q[i] ) {
            // Tag to be refined
            newPixelIdx.push_back(v);

          }
          */
        }
      }
    }
  }
  //
  // Part II - Extract the local max
  //
  // Extract the new selected pixels
  int nNewPixels = newPixelIdx.size();
  // int indexInThePixel[nNewPixels];
  Pads* newPixels = new Pads(nNewPixels, chamberId);
  for (int i = 0; i < nNewPixels; i++) {
    newPixels->x[i] = x[newPixelIdx[i]];
    newPixels->y[i] = y[newPixelIdx[i]];
    newPixels->dx[i] = dx[newPixelIdx[i]];
    newPixels->dy[i] = dy[newPixelIdx[i]];
    newPixels->q[i] = q[newPixelIdx[i]];
  }
  Pads* localMax = nullptr;
  // Suppress local max. whose charge is less of 1%
  // of the max charge of local Max
  double cutRatio = 0.01;
  double qCut = cutRatio * vectorMax(newPixels->q, newPixels->nPads);

  localMax = new Pads(nNewPixels, chamberId);
  localMax->setToZero();

  int k0 = 0;
  printf(" q Cut-Off %f\n", qCut);
  for (int k = 0; k < nNewPixels; k++) {
    if (newPixels->q[k] > qCut) {
      localMax->q[k0] = newPixels->q[k];
      localMax->x[k0] = newPixels->x[k];
      localMax->y[k0] = newPixels->y[k];
      localMax->dx[k0] = newPixels->dx[k];
      localMax->dy[k0] = newPixels->dy[k];
      printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
             localMax->q[k0], localMax->x[k0], localMax->y[k0]);
      k0++;
    }
  }
  localMax->nPads = k0;
  localMax->nObsPads = k0;
  /// ???? delete[] neigh;
  //
  // Part IV - Refine the charge and coordinates of the local max.
  //
  // Avoid taking the same charge for 2 different localMax
  // neigh = newPixels->buildFirstNeighbors();
  // printNeighbors( neigh, newPixels->getNbrOfPads());
  if (0) {
    Mask_t mask[nNewPixels];
    vectorSetShort(mask, 1, nNewPixels);
    int kSelected = 0;
    // ???
    qCut = 0.0;

    for (int k = 0; k < nNewPixels; k++) {
      if (mask[k] == 1) {
        // Compute the charge barycenter
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, k);
             *neigh_ptr != -1; neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          localMax->q[k] += newPixels->q[v] * mask[v];
          localMax->x[k] += newPixels->x[v] * newPixels->q[v] * mask[v];
          localMax->y[k] += newPixels->y[v] * newPixels->q[v] * mask[v];
          mask[v] = 0;
        }
        // Select (or not) the local Max
        if (localMax->q[k] > qCut) {
          localMax->q[kSelected] = localMax->q[k];
          localMax->x[kSelected] = localMax->x[k] / localMax->q[k];
          localMax->y[kSelected] = localMax->y[k] / localMax->q[k];
          localMax->dx[kSelected] = newPixels->dx[k];
          localMax->dy[kSelected] = newPixels->dy[k];
          if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
            printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
                   localMax->q[k], localMax->x[k], localMax->y[k]);
          }
          localMaxIdx.push_back(newPixelIdx[k]);
          kSelected++;
        }
      }
    }

    for (int k = 0; k < nNewPixels; k++) {
      localMax->q[k] = newPixels->q[k];
      localMax->x[k] = newPixels->x[k];
      localMax->y[k] = newPixels->y[k];
      localMax->dx[k] = newPixels->dx[k];
      localMax->dy[k] = newPixels->dy[k];
      printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
             localMax->q[k], localMax->x[k], localMax->y[k]);
    }
    kSelected = nNewPixels;
    localMax->nPads = kSelected;
    localMax->nObsPads = kSelected;
  }

  delete[] neighbors;
  neighbors = nullptr;

  delete newPixels;

  return localMax;
}

// Assess or not if xyCheck is a remanent local Max (can be removed)
bool Pads::assessRemanent(double xyCheck, double* xy, double precision, int N)
{
  //
  double xyDiff[N];
  Mask_t mask[N];
  vectorAddScalar(xy, -xyCheck, N, xyDiff);
  // vectorPrint("  [assessRemanent] xy", xy, N);
  // vectorPrint("  [assessRemanent] xyDiff", xyDiff, N);
  vectorAbs(xyDiff, N, xyDiff);
  vectorBuildMaskLess(xyDiff, precision, N, mask);
  int nRemanents = vectorSumShort(mask, N);
  // One xyDiff is zero => nRemanents >= 1
  bool remanent = (nRemanents > 1) ? true : false;
  // printf("  [assessRemanent] xyCheck=%f precision=%f remanent=%d\n", xyCheck, precision, nRemanents);
  return remanent;
}

Pads* Pads::extractLocalMaxOnCoarsePads_Remanent(std::vector<PadIdx_t>& localMaxIdx, double dxMinPadSize, double dyMinPadSize)
{
  if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
    printf("  - Pads::extractLocalMax on Coarse Pads(extractLocalMax nPads=%d)\n", nPads);
  }
  double qMax = vectorMax(q, nPads);
  //
  // TO DO ??? Compute the neighbors once
  // between to refinements
  if (neighbors != nullptr) {
    delete[] neighbors;
  }
  // 4(5) neighbors
  neighbors = getFirstNeighbors();
  PadIdx_t* neigh = neighbors;
  // printNeighbors( neigh, nPads);
  //
  // Part I - Morphologic Laplacian operator
  //
  double morphLaplacian[nPads];
  double laplacian[nPads];
  double weight[nPads];
  vectorSet(morphLaplacian, -1.0, nPads);
  // Invalid the neighbors of a local max
  Mask_t alreadyDone[nPads];
  vectorSetZeroShort(alreadyDone, nPads);
  std::vector<PadIdx_t> newPixelIdx;
  bool less;
  for (int i = 0; i < nPads; i++) {
    if (alreadyDone[i] == 0) {
      int nLess = 0;
      int count = 0;
      laplacian[i] = 0.0;
      weight[i] = 0.0;
      for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
           neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        // Morphologic Laplacian
        // nLess += (q[v] < q[i]);
        less = (q[v] <= q[i]);
        count++;
        if (less) {
          nLess++;
          // Laplacian
          double cst;
          cst = (i == v) ? 1.0 : -0.25;
          laplacian[i] += cst * q[v];
          weight[i] += q[v];
        }
      }
      // Invalid ?? morphLaplacian[i] = double(nLess) / (count - 1);
      morphLaplacian[i] = double(nLess) / count;
      //
      if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
        printf(
          "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
          "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
          i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
          weight[i]);
      }
      if (morphLaplacian[i] >= 1.0) {
        //  Local max charge must be higher than 1.5 % of the max and
        //  the curvature must be greater than 50% of the peak
        // Inv ??? if ((q[i] > 0.015 * qMax) || (fabs(laplacian[i]) > (0.5 * q[i]))) {
        if (q[i] > 0.015 * qMax) {
          newPixelIdx.push_back(i);
          // if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
          if (0) {
            printf(
              "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
              "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
              i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
              weight[i]);
            printf("  Selected %d\n", i);
          }
        }
        // Invalid the neihbors
        // they can't be a maximun
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
             neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          alreadyDone[v] += 1;
          /*
          if (q[v] > 0.5 * q[i] ) {
            // Tag to be refined
            newPixelIdx.push_back(v);

          }
          */
        }
      }
    }
  }
  //
  // Part II - Extract the local max
  //
  // Extract the new selected pixels
  int nNewPixels = newPixelIdx.size();
  // int indexInThePixel[nNewPixels];
  Pads* newPixels = new Pads(nNewPixels, chamberId);
  for (int i = 0; i < nNewPixels; i++) {
    newPixels->x[i] = x[newPixelIdx[i]];
    newPixels->y[i] = y[newPixelIdx[i]];
    newPixels->dx[i] = dx[newPixelIdx[i]];
    newPixels->dy[i] = dy[newPixelIdx[i]];
    newPixels->q[i] = q[newPixelIdx[i]];
  }
  Pads* localMax = nullptr;
  // Suppress local max. whose charge is less of 1%
  // of the max charge of local Max
  double cutRatio = 0.01;
  double qCut = cutRatio * vectorMax(newPixels->q, newPixels->nPads);

  // Add pads / pixel to be refined.
  // They are neigbous of 2 or more local max
  /*
  for (int i = 0; i < nPads; i++) {
    if (alreadyDone[i] > 1) {
      newPixelIdx.push_back(i);
      printf("Other pad/pixel to be refined: i=%d x,y=(%7.2f,%7.2f) q=%8.1f \n", i, x[i], y[i], q[i]);
    }
  }
  */

  //
  // Part III - suppress the remanent local max
  //
  if (clusterConfig.processingLog >= clusterConfig.info) {
    printf("  [extractLocalMaxOnCoarsePads] Start suppressing remanent localMax: nbr Local Max [nNewPixels]=%d\n", nNewPixels);
  }
  int k0;
  if (nNewPixels > 3) {
    // ??? TODO:  suppress the refinment to optimize
    localMax = new Pads(nNewPixels, chamberId);
    localMax->setToZero();
    // Sort local max by charge value
    // ??? visibly not used ???
    int index[nNewPixels];
    for (int k = 0; k < nNewPixels; k++) {
      index[k] = k;
    }
    std::sort(index, &index[nNewPixels], [=](int a, int b) {
      return (newPixels->q[a] > newPixels->q[b]);
    });
    // k0 describe the list of true local max (local max - remanent local max)
    // k0 number of true local max
    k0 = 0;
    for (int k = 0; k < nNewPixels; k++) {
      if (index[k] > -1) {
        // Store the true local max
        index[k0] = index[k];
        int idx0 = index[k0];

        // Remove horizontal/vertical remanent local max
        double x0 = newPixels->x[idx0];
        double y0 = newPixels->y[idx0];
        if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
          printf("    Check remanent from loc.max k=%d, (x,y,q)= %f %f %f\n", k, x0, y0, newPixels->q[idx0]);
        }
        for (int l = k + 1; l < nNewPixels; l++) {
          if (index[l] > -1) {
            if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
              printf("      Case l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
            }

            bool sameX = (std::abs(newPixels->x[index[l]] - x0) < dxMinPadSize);
            bool sameY = (std::abs(newPixels->y[index[l]] - y0) < dyMinPadSize);
            if (sameX) {
              // Check in Y axe
              // Check other remanent loc max in y direction)
              // If founded : true remanent loc Max
              // if not a real remanent loc max (must be kept)
              bool realRemanent = assessRemanent(newPixels->y[index[l]], newPixels->y, dyMinPadSize, nNewPixels);
              if (realRemanent) {
                // Remanent local max: remove it
                // The local max absorb the charge of the remanent loc max                newPixels->q[idx0] += newPixels->q[index[l]];
                // Remove the remanent
                if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
                  printf("      XY-Remanent: remove l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
                }
                index[l] = -1;
              }
            }
            if (sameY) {
              // Check in Y axe
              // Check other remanent loc max in y direction)
              // If founded : true remanent loc Max
              // if not a real remanent loc max (must be kept)
              bool realRemanent = assessRemanent(newPixels->x[index[l]], newPixels->x, dyMinPadSize, nNewPixels);
              if (realRemanent) {
                // Remanent local max: remove it
                // The local max absorb the charge of the remanent loc max
                newPixels->q[idx0] += newPixels->q[index[l]];
                // Remove the remanent
                if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
                  printf("      YX-Remanent: remove l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
                }
                index[l] = -1;
              }
              if ((sameX == 0) && (sameX == 0) && (clusterConfig.EMLocalMaxLog >= clusterConfig.info)) {
                printf("      Keep l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
              }
            }
          }
        }
        k0++;
      }
    }
    if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
      for (int l = 0; l < k0; l++) {
        printf("    l=%d index[l]=%d (x, y, q)= %f %f %f\n", l, index[l], newPixels->getX()[index[l]], newPixels->getY()[index[l]], newPixels->getCharges()[index[l]]);
      }
    }
    // Clean the local Max - Remove definitely remanent local max
    delete localMax;
    localMax = newPixels->selectPads(index, k0);
    nNewPixels = k0;
  } else {
    if (localMax != nullptr) {
      delete localMax;
    }
    localMax = new Pads(*newPixels, PadMode::xydxdyMode);
    k0 = nNewPixels;
  }

  if (0) {
    localMax = new Pads(nNewPixels, chamberId);
    localMax->setToZero();
  }
  /// ???? delete[] neigh;
  //
  // Part IV - Refine the charge and coordinates of the local max.
  //
  // Avoid taking the same charge for 2 different localMax
  // neigh = newPixels->buildFirstNeighbors();
  // printNeighbors( neigh, newPixels->getNbrOfPads());
  if (0) {
    Mask_t mask[nNewPixels];
    vectorSetShort(mask, 1, nNewPixels);
    int kSelected = 0;
    // ???
    qCut = 0.0;

    for (int k = 0; k < nNewPixels; k++) {
      if (mask[k] == 1) {
        // Compute the charge barycenter
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, k);
             *neigh_ptr != -1; neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          localMax->q[k] += newPixels->q[v] * mask[v];
          localMax->x[k] += newPixels->x[v] * newPixels->q[v] * mask[v];
          localMax->y[k] += newPixels->y[v] * newPixels->q[v] * mask[v];
          mask[v] = 0;
        }
        // Select (or not) the local Max
        if (localMax->q[k] > qCut) {
          localMax->q[kSelected] = localMax->q[k];
          localMax->x[kSelected] = localMax->x[k] / localMax->q[k];
          localMax->y[kSelected] = localMax->y[k] / localMax->q[k];
          localMax->dx[kSelected] = newPixels->dx[k];
          localMax->dy[kSelected] = newPixels->dy[k];
          if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
            printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
                   localMax->q[k], localMax->x[k], localMax->y[k]);
          }
          localMaxIdx.push_back(newPixelIdx[k]);
          kSelected++;
        }
      }
    }

    for (int k = 0; k < nNewPixels; k++) {
      localMax->q[k] = newPixels->q[k];
      localMax->x[k] = newPixels->x[k];
      localMax->y[k] = newPixels->y[k];
      localMax->dx[k] = newPixels->dx[k];
      localMax->dy[k] = newPixels->dy[k];
      printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
             localMax->q[k], localMax->x[k], localMax->y[k]);
    }
    kSelected = nNewPixels;
    localMax->nPads = kSelected;
    localMax->nObsPads = kSelected;
  }

  delete[] neighbors;
  neighbors = nullptr;

  delete newPixels;

  return localMax;
}

Pads* Pads::extractLocalMax(std::vector<PadIdx_t>& localMaxIdx, double dxMinPadSize, double dyMinPadSize)
{
  if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
    printf("  - Pads::extractLocalMax (extractLocalMax nPads=%d)\n", nPads);
  }
  double qMax = vectorMax(q, nPads);
  //
  // TO DO ??? Compute the neighbors once
  // between to refinements
  if (neighbors != nullptr) {
    delete[] neighbors;
  }
  // Kernel size of 1
  neighbors = buildKFirstsNeighbors(1);
  PadIdx_t* neigh = neighbors;
  // printNeighbors( neigh, nPads);
  //
  // Result of the Laplacian-like operator
  double morphLaplacian[nPads];
  double laplacian[nPads];
  double weight[nPads];
  vectorSet(morphLaplacian, -1.0, nPads);
  Mask_t alreadyDone[nPads];
  vectorSetZeroShort(alreadyDone, nPads);
  std::vector<PadIdx_t> newPixelIdx;
  bool less;
  for (int i = 0; i < nPads; i++) {
    if (alreadyDone[i] == 0) {
      int nLess = 0;
      int count = 0;
      laplacian[i] = 0.0;
      weight[i] = 0.0;
      for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
           neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        // Morphologic Laplacian
        // nLess += (q[v] < q[i]);
        less = (q[v] <= q[i]);
        count++;
        if (less) {
          nLess++;
          // Laplacian
          double cst;
          cst = (i == v) ? 1.0 : -0.125;
          laplacian[i] += cst * q[v];
          weight[i] += q[v];
        }
      }
      // Invalid ?? morphLaplacian[i] = double(nLess) / (count - 1);
      morphLaplacian[i] = double(nLess) / count;
      //
      if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
        printf(
          "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
          "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
          i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
          weight[i]);
      }
      if (morphLaplacian[i] >= 1.0) {
        //  Local max charge must be higher than 1.5 % of the max and
        //  the curvature must be greater than 50% of the peak
        // Inv ??? if ((q[i] > 0.015 * qMax) || (fabs(laplacian[i]) > (0.5 * q[i]))) {
        if (q[i] > 0.015 * qMax) {
          newPixelIdx.push_back(i);
          // if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
          if (0) {
            printf(
              "    Laplacian i=%d, x=%6.3f, y=%6.3f, dx=%6.3f,dy=%6.3f, q=%6.3f, "
              "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight=%6.3f",
              i, x[i], y[i], dx[i], dy[i], q[i], count, morphLaplacian[i], laplacian[i],
              weight[i]);
            printf("  Selected %d\n", i);
          }
        }
        // Invalid the neihbors
        // they can't be a maximun
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
             neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          alreadyDone[v] += 1;
          /*
          if (q[v] > 0.5 * q[i] ) {
            // Tag to be refined
            newPixelIdx.push_back(v);

          }
          */
        }
      }
    }
  }
  //
  // Extract the new selected pixels
  int nNewPixels = newPixelIdx.size();
  // int indexInThePixel[nNewPixels];
  Pads* newPixels = new Pads(nNewPixels, chamberId);
  for (int i = 0; i < nNewPixels; i++) {
    newPixels->x[i] = x[newPixelIdx[i]];
    newPixels->y[i] = y[newPixelIdx[i]];
    newPixels->dx[i] = dx[newPixelIdx[i]];
    newPixels->dy[i] = dy[newPixelIdx[i]];
    newPixels->q[i] = q[newPixelIdx[i]];
    localMaxIdx.push_back(newPixelIdx[i]);
  }
  Pads* localMax = nullptr;
  // Suppress local max. whose charge is less of 1%
  // of the max charge of local Max
  double cutRatio = 0.01;
  double qCut = cutRatio * vectorMax(newPixels->q, newPixels->nPads);

  // Add pads / pixel to be refined.
  // They are neigbous of 2 or more local max
  /*
  for (int i = 0; i < nPads; i++) {
    if (alreadyDone[i] > 1) {
       newPixelIdx.push_back(i);
      printf("Other pad/pixel to be refined: i=%d x,y=(%7.2f,%7.2f) q=%8.1f \n", i, x[i], y[i], q[i]);
    }
  }
   */

  //
  // Part III - suppress the remanent local max
  //
  if (clusterConfig.processingLog >= clusterConfig.info) {
    printf("  [extractLocalMax] (medium pads) Starting suppressing remanent Loc. Max nNewPixels=%d\n", nNewPixels);
  }
  int k0;
  std::vector<int> newPixelIdx2;

  // if ( (nNewPixels > 3) && ( (dxMinPadSize > 3.5) || (dyMinPadSize > 3.5) )) {
  if ((nNewPixels > 3) && ((dxMinPadSize > 2.4) || (dyMinPadSize > 2.4))) {
    // ??? TODO:  suppress the refinment to optimize
    localMax = new Pads(nNewPixels, chamberId);
    localMax->setToZero();
    // Sort local max by charge value
    // ??? visibly not used ???
    int index[nNewPixels];
    for (int k = 0; k < nNewPixels; k++) {
      index[k] = k;
    }
    std::sort(index, &index[nNewPixels], [=](int a, int b) {
      return (newPixels->q[a] > newPixels->q[b]);
    });
    // k0 describe the list of true local max (local max - remanent local max)
    // k0 number of true local max
    k0 = 0;
    // vectorPrintInt("Index",  index, nNewPixels);
    // Pads::printPads("local max", *newPixels);

    for (int k = 0; k < nNewPixels; k++) {
      if (index[k] > -1) {
        // Store the true local max
        index[k0] = index[k];
        int idx0 = index[k0];

        // Remove horizontal/vertical remanent local max
        double x0 = newPixels->x[idx0];
        double y0 = newPixels->y[idx0];
        double dx0 = newPixels->dx[idx0];
        double dy0 = newPixels->dy[idx0];
        if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
          printf("  Remanent from loc max k=%d, (x,y,q)= %f %f %f (dx, dy)= (%f, %f)\n", k, x0, y0, newPixels->q[idx0], dx0, dy0);
        }
        for (int l = k + 1; l < nNewPixels; l++) {
          if (index[l] > -1) {
            double dx_ = 0.5 * (dx0 + newPixels->dx[index[l]]);
            double dy_ = 0.5 * (dy0 + newPixels->dy[index[l]]);
            bool sameX = (std::abs(newPixels->x[index[l]] - x0) < dx_);
            bool sameY = (std::abs(newPixels->y[index[l]] - y0) < dy_);
            // printf("  Remanent: precision l=%d, (dx,dy)= %f %f \n", l, dx_, dy_ );
            if (sameX) {
              // Check in Y axe
              // Check other remanent loc max in y direction)
              // If founded : true remanent loc Max
              // if not a real remanent loc max (must be kept)
              bool realRemanent = assessRemanent(newPixels->y[index[l]], newPixels->y, dy_, nNewPixels);
              if (realRemanent) {
                // Remanent local max: remove it
                // The local max absorb the charge of the remanent loc max                newPixels->q[idx0] += newPixels->q[index[l]];
                // Remove the remanent
                if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
                  printf("    XY-Remanent: remove l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
                }
                index[l] = -1;
              }
            }
            if (sameY) {
              // Check in Y axe
              // Check other remanent loc max in y direction)
              // If founded : true remanent loc Max
              // if not a real remanent loc max (must be kept)
              bool realRemanent = assessRemanent(newPixels->x[index[l]], newPixels->x, dx_, nNewPixels);
              if (realRemanent) {
                // Remanent local max: remove it
                // The local max absorb the charge of the remanent loc max
                newPixels->q[idx0] += newPixels->q[index[l]];
                // Remove the remanent
                if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
                  printf("    YX-Remanent: remove l=%d, (x,y,q)= %f %f %f\n", l, newPixels->x[index[l]], newPixels->y[index[l]], newPixels->q[index[l]]);
                }
                index[l] = -1;
              }
            }
          }
        }
        k0++;
      }
    }
    if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
      printf("  Local. max status before to suppress remanents\n");
      for (int l = 0; l < k0; l++) {
        printf("   l=%d index[l]=%d (x, y, q)= %f %f %f\n", l, index[l], newPixels->getX()[index[l]], newPixels->getY()[index[l]], newPixels->getCharges()[index[l]]);
      }
    }

    // Clean the local Max - Remove definitely remanent local max
    if (localMax != nullptr) {
      delete localMax;
    }
    localMax = newPixels->selectPads(index, k0);
    // Update  newPixelIdx
    if (1) {
      for (int l = 0; l < k0; l++) {
        int idx = index[l];
        newPixelIdx2.push_back(newPixelIdx[idx]);
        // Debug
        // printf("  newPixelIdx2 l=%d index[l]=%d (x, y, q)= %f %f %f\n", l, index[l], x[newPixelIdx[l]], y[newPixelIdx[l]], q[newPixelIdx[l]]);
      }
    }
    nNewPixels = k0;
  } else {
    // Copy newPixels -> localMax
    localMax = new Pads(*newPixels, PadMode::xydxdyMode);
    k0 = nNewPixels;
    newPixelIdx2 = newPixelIdx;
  }

  /*
  //
  // Refine the charge and coordinates of the local max.
  //
  // ??? TODO:  suppress te refinment to optimize
  localMax = new Pads(nNewPixels, chamberId);
  localMax->setToZero();
  // Sort local max by charge value
  int index[nNewPixels];
  for (int k = 0; k < nNewPixels; k++) {
    index[k] = k;
  }
  // ??? visibly not used ???
  std::sort(index, &index[nNewPixels], [=](int a, int b) {
    return (newPixels->q[a] > newPixels->q[b]);
  });
  */

  /// ???? delete[] neigh;
  // Avoid taking the same charge for 2 different localMax
  // Add the local max in list (to be refined)

  // Unused
  // Mask_t mask[nNewPixels];
  // vectorSetShort(mask, 1, nNewPixels);
  int kSelected = 0;
  for (int l = 0; l < nNewPixels; l++) {
    PadIdx_t pixelIdx = newPixelIdx2[l];
    // Unused
    // if (mask[k] == 1) {
    // Compute the charge barycenter
    localMax->q[l] = 0.0;
    localMax->x[l] = 0.0;
    localMax->y[l] = 0.0;
    int nNeigh = 0;
    for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, pixelIdx);
         *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t v = *neigh_ptr;
      localMax->q[l] += q[v];        // * mask[v];
      localMax->x[l] += x[v] * q[v]; // * mask[v];
      localMax->y[l] += y[v] * q[v]; // * mask[v];
      // Unused
      // mask[v] = 0;
      nNeigh++;
    }
    // Select (or not) the local Max
    if (localMax->q[l] > qCut) {

      localMax->x[kSelected] = localMax->x[l] / localMax->q[l];
      localMax->y[kSelected] = localMax->y[l] / localMax->q[l];
      localMax->q[kSelected] = localMax->q[l] / nNeigh;
      localMax->dx[kSelected] = dx[pixelIdx];
      localMax->dy[kSelected] = dy[pixelIdx];
      if (clusterConfig.EMLocalMaxLog >= clusterConfig.info) {
        printf("  [extractLocalMax] final seed selected (x,y) = (%8.3f, %8.3f), q=%8.2f\n",
               localMax->x[l], localMax->y[l], localMax->q[l]);
      }
      // localMaxIdx.push_back( pixelIdx );
      kSelected++;
    }
    // Unused }
  }
  localMax->nPads = kSelected;
  localMax->nObsPads = kSelected;

  // Add high charge neighbors to be refined
  if (0) {
    for (int k = 0; k < nNewPixels; k++) {
      // Compute the charge barycenter
      PadIdx_t idxMax = newPixelIdx[k];
      for (PadIdx_t* neigh_ptr = getNeighborListOf(neighbors, idxMax);
           *neigh_ptr != -1; neigh_ptr++) {
        PadIdx_t v = *neigh_ptr;
        if ((q[v] > 0.5 * q[idxMax]) && (q[v] > clusterConfig.minChargeOfClusterPerCathode)) {
          // Tag to be refined
          localMaxIdx.push_back(v);
          printf("??? neigbors of idMax=%d: %d to be refined (charge %f/%f)\n", idxMax, v, q[v], q[idxMax]);
          // Inv printf("x,y : %f %f \n", x[v], y[v]);
        }
      }
    }
  }

  delete[] neighbors;
  neighbors = nullptr;
  delete newPixels;

  return localMax;
}

Pads* Pads::clipOnLocalMax(bool extractLocalMax)
{
  // Option extractLocalMax
  //   - true: extraxt local maxima
  //   - false: filter pixels arround the maxima
  if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
    printf("  - ClipOnLocalMax (extractLocalMax Flag=%d, nPads=%d)\n",
           extractLocalMax, nPads);
  }
  double eps = epsilonGeometry;
  // relativeNoise unused (set to 0)
  double relativeNoise = 0.00;
  double qMax = vectorMax(q, nPads);
  double cutoff;
  // Compute the neighbors once
  if ((neighbors == nullptr) && extractLocalMax) {
    // Kernel size of 1
    neighbors = buildKFirstsNeighbors(1);
  } else if (neighbors == nullptr) {
    neighbors = buildKFirstsNeighbors(2);
  }
  PadIdx_t* neigh = neighbors;
  //
  // Result of the Laplacian-like operator
  double morphLaplacian[nPads];
  double laplacian[nPads];
  double weight[nPads];
  vectorSet(morphLaplacian, -1.0, nPads);
  Mask_t alreadySelect[nPads];
  vectorSetZeroShort(alreadySelect, nPads);
  std::vector<PadIdx_t> newPixelIdx;
  for (int i = 0; i < nPads; i++) {
    int nLess = 0;
    int count = 0;
    laplacian[i] = 0.0;
    weight[i] = 0.0;
    cutoff = relativeNoise * q[i];
    for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
         neigh_ptr++) {
      PadIdx_t v = *neigh_ptr;
      // Morphologic Laplacian
      nLess += (q[v] < (q[i] - cutoff));
      count++;
      // Laplacian
      double cst;
      cst = (i == v) ? 1.0 : -0.125;
      laplacian[i] += cst * q[v];
      weight[i] += q[v];
    }
    morphLaplacian[i] = double(nLess) / (count - 1);
    //
    if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
      printf(
        "  Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, count=%d, "
        "morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight[i]=%6.3f\n",
        i, x[i], y[i], q[i], count, morphLaplacian[i], laplacian[i],
        weight[i]);
    }
    if (morphLaplacian[i] >= 1.0) {
      if (extractLocalMax) {
        //  Local max charge must be higher than 1.5 % of the max and
        //  the curvature must be greater than 50% of the peak
        if ((q[i] > 0.015 * qMax) || (fabs(laplacian[i]) > (0.5 * q[i]))) {
          newPixelIdx.push_back(i);
          if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
            printf(
              "  Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, "
              "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight[i]=%6.3f",
              i, x[i], y[i], q[i], count, morphLaplacian[i], laplacian[i],
              weight[i]);
            printf("  Selected %d\n", i);
          }
        }
      } else {
        // Select as new pixels in the vinicity of the local max
        if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
          printf("  Selected neighbors of i=%d: ", i);
        }
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i);
             *neigh_ptr != -1; neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          if (alreadySelect[v] == 0) {
            alreadySelect[v] = 1;
            newPixelIdx.push_back(v);
            if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
              printf("%d, ", v);
            }
          }
        }
        if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
          printf("\n");
        }
      }
    }
  }
  // Extract the new selected pixels
  int nNewPixels = newPixelIdx.size();
  Pads* newPixels = new Pads(nNewPixels, chamberId);
  for (int i = 0; i < nNewPixels; i++) {
    newPixels->x[i] = x[newPixelIdx[i]];
    newPixels->y[i] = y[newPixelIdx[i]];
    newPixels->dx[i] = dx[newPixelIdx[i]];
    newPixels->dy[i] = dy[newPixelIdx[i]];
    newPixels->q[i] = q[newPixelIdx[i]];
  }
  Pads* localMax = nullptr;
  if (extractLocalMax) {
    // Suppress local max. whose charge is less of 1% of the max charge of local
    // Max
    double cutRatio = 0.01;
    double qCut = cutRatio * vectorMax(newPixels->q, newPixels->nPads);
    //
    // Refine the charge and coordinates of the local max.
    //
    // ??? TODO:  suppress te refinment to optimize
    localMax = new Pads(nNewPixels, chamberId);
    localMax->setToZero();
    // Sort local max by charge value
    int index[nNewPixels];
    for (int k = 0; k < nNewPixels; k++) {
      index[k] = k;
    }
    std::sort(index, &index[nNewPixels], [=](int a, int b) {
      return (newPixels->q[a] > newPixels->q[b]);
    });
    /// ???? delete[] neigh;
    neigh = newPixels->buildKFirstsNeighbors(1);
    // Avoid taking the same charge for 2 different localMax
    Mask_t mask[nNewPixels];
    vectorSetShort(mask, 1, nNewPixels);
    int kSelected = 0;
    for (int k = 0; k < nNewPixels; k++) {
      if (mask[k] == 1) {
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, k);
             *neigh_ptr != -1; neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          localMax->q[k] += newPixels->q[v] * mask[v];
          localMax->x[k] += newPixels->x[v] * newPixels->q[v] * mask[v];
          localMax->y[k] += newPixels->y[v] * newPixels->q[v] * mask[v];
          mask[v] = 0;
        }
        if (localMax->q[k] > qCut) {
          localMax->q[kSelected] = localMax->q[k];
          localMax->x[kSelected] = localMax->x[k] / localMax->q[k];
          localMax->y[kSelected] = localMax->y[k] / localMax->q[k];
          localMax->dx[kSelected] = newPixels->dx[k];
          localMax->dy[kSelected] = newPixels->dy[k];
          if (clusterConfig.EMLocalMaxLog >= clusterConfig.detail) {
            printf("  add a seed q=%9.4f, (x,y) = (%9.4f, %9.4f)\n",
                   localMax->q[k], localMax->x[k], localMax->q[k]);
          }
          kSelected++;
        }
      }
    }
    localMax->nPads = kSelected;
    localMax->nObsPads = kSelected;
  }
  delete[] neigh;
  if (extractLocalMax) {
    delete newPixels;
    return localMax;
  } else {
    return newPixels;
  }
}

void Pads::printNeighbors(const PadIdx_t* neigh, int N)
{
  printf("Neighbors %d\n", N);
  for (int i = 0; i < N; i++) {
    printf("  neigh of i=%2d: ", i);
    for (const PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i);
         *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t j = *neigh_ptr;
      printf("%d, ", j);
    }
    printf("\n");
  }
}

void Pads::printPads(const char* title, const Pads& pads)
{
  printf("%s\n", title);
  printf("print pads nPads=%4d nObsPads=%4d mode=%1d\n", pads.nPads, pads.nObsPads, (int)pads.mode);
  if (pads.mode == PadMode::xydxdyMode) {
    printf("    i       x       y      dx      dy         q\n");
    for (int i = 0; i < pads.nPads; i++) {
      printf("  %3d %7.3f %7.3f %7.3f %7.3f %9.2f\n", i,
             pads.x[i], pads.dx[i], pads.y[i], pads.dy[i], pads.q[i]);
    }
  } else {
    printf("    i    xInf    xSup    yInf    ySup         q\n");
    for (int i = 0; i < pads.nPads; i++) {
      printf("  %3d %7.3f %7.3f %7.3f %7.3f %9.2f\n",
             i, pads.x[i], pads.dx[i], pads.y[i], pads.dy[i], pads.q[i]);
    }
  }
  // Invalid
  // } else {
  //  printf("%s can't print nullptr\n", title);
  // }
}

Pads::~Pads() { release(); }

} // namespace mch
} // namespace o2
