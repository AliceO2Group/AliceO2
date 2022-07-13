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
#include "mathUtil.h"

#define VERBOSE 1
#define CHECK 1

namespace o2
{
namespace mch
{

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

  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
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
  // Build neigbours if required
  PadIdx_t* neigh = buildFirstNeighbors();
  for (int i = 0; i < N; i++) {
    bool east = true, west = true, north = true, south = true;
    for (const PadIdx_t* neigh_ptr = getTheFirtsNeighborOf(neigh, i);
         *neigh_ptr != -1; neigh_ptr++) {
      PadIdx_t v = *neigh_ptr;
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
  int nPadToAdd = bX.size();
  int nTotalPads = N + nPadToAdd;
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
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
  for (int i = N, k = 0; i < nTotalPads; i++, k++) {
    newPads->x[i] = bX[k];
    newPads->y[i] = bY[k];
    newPads->dx[i] = bdX[k];
    newPads->dy[i] = bdY[k];
    newPads->q[i] = 0.0;
    newPads->saturate[i] = 0;
  }
  newPads->totalCharge = totalCharge;
  //
  return padsWithBoundaries;
}

Pads::Pads(int N, int chId, int mode_)
{
  nPads = N;
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

Pads::Pads(const Pads& pads, int mode_)
{
  nPads = pads.nPads;
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
  } else if (mode == xydxdyMode) {
    //  xyInfSupMode ->  xydxdyMode
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
    memcpy(q, pads.q, sizeof(double) * nPads);
  } else {
    // xydxdyMode -> xyInfSupMode
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
    memcpy(q, pads.q, sizeof(double) * nPads);
  }
  memcpy(saturate, pads.saturate, sizeof(Mask_t) * nPads);
}

Pads::Pads(const Pads& pads, const Mask_t* mask)
{
  nPads = vectorSumShort(mask, pads.nPads);
  mode = xydxdyMode;
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
  totalCharge = vectorSum(q, nPads);
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
  mode = xydxdyMode;
  nPads = nPads_;
  chamberId = chId;
  x = x_;
  y = y_;
  dx = dx_;
  dy = dy_;
  q = new double[nPads];
  // Set null Charge
  vectorSetZero(q, nPads);
  neighbors = nullptr;
  totalCharge = 0;
}

Pads::Pads(const double* x_, const double* y_, const double* dx_,
           const double* dy_, const double* q_, const short* cathode,
           const Mask_t* saturate_, short selectedCath, int chId,
           PadIdx_t* mapCathPadIdxToPadIdx, int nAllPads)
{
  mode = xydxdyMode;
  int nCathode1 = vectorSumShort(cathode, nAllPads);
  nPads = nCathode1;
  if (selectedCath == 0) {
    nPads = nAllPads - nCathode1;
  }
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
  mode = xydxdyMode;
  // int nCathode1 = vectorSumShort(cathode, nAllPads);
  nPads = nAllPads;
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
Pads::Pads(const Pads* pads1, const Pads* pads2, int mode_)
{
  // Take Care: pads1 and pads2 must be in xydxdyMode
  int N1 = (pads1 == nullptr) ? 0 : pads1->nPads;
  int N2 = (pads2 == nullptr) ? 0 : pads2->nPads;
  nPads = N1 + N2;
  chamberId = (N1) ? pads1->chamberId : pads2->chamberId;
  mode = mode_;
  allocate();
  if (mode == xydxdyMode) {
    // Copy pads1
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
  } else {
    double* xInf = x;
    double* yInf = y;
    double* xSup = dx;
    double* ySup = dy;
    for (int i = 0; i < N1; i++) {
      xInf[i] = pads1->x[i] - pads1->dx[i];
      xSup[i] = pads1->x[i] + pads1->dx[i];
      yInf[i] = pads1->y[i] - pads1->dy[i];
      ySup[i] = pads1->y[i] + pads1->dy[i];
      q[i] = pads1->q[i];
      saturate[i] = pads1->saturate[i];
      cath[i] = 0;
    }
    for (int i = 0; i < N2; i++) {
      xInf[i + N1] = pads2->x[i] - pads2->dx[i];
      xSup[i + N1] = pads2->x[i] + pads2->dx[i];
      yInf[i + N1] = pads2->y[i] - pads2->dy[i];
      ySup[i + N1] = pads2->y[i] + pads2->dy[i];
      q[i + N1] = pads2->q[i];
      saturate[i + N1] = pads2->saturate[i];
      cath[i + N1] = 1;
    }
  }
  totalCharge = vectorSum(q, nPads);
}

void Pads::removePad(int index)
{
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
  double qSum = 0.0;
  int k = 0;
  for (int i = 0; i < nPads; i++) {
    if (q[i] > qCut) {
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
  return k;
}

void Pads::normalizeCharges()
{
  for (int i = 0; i < nPads; i++) {
    q[i] = q[i] / totalCharge;
  }
}

// Build the neighbor list
PadIdx_t* Pads::buildFirstNeighbors()
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

  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
    printf("[addIsolatedPadInGroups]  nGroups=%d\n", nGroups);
    vectorPrintShort("  cathToGrp input", cathToGrp, nPads);
  }
  PadIdx_t* neigh = buildFirstNeighbors();

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

  if (ClusterConfig::padMappingLog >= ClusterConfig::debug) {
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
}

Pads* Pads::refinePads()
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
  if (ClusterConfig::padMappingLog >= ClusterConfig::detail) {
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

Pads* Pads::extractLocalMax()
{
  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
    printf("  - Pads::extractLocalMax (extractLocalMax nPads=%d)\n",
           nPads);
  }
  double qMax = vectorMax(q, nPads);
  //
  // Compute the neighbors once
  if (neighbors == nullptr) {
    // Kernel size of 1
    neighbors = buildKFirstsNeighbors(1);
  }
  PadIdx_t* neigh = neighbors;
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
      if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
        printf(
          "    Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, count=%d, "
          "morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight[i]=%6.3f\n",
          i, x[i], y[i], q[i], count, morphLaplacian[i], laplacian[i],
          weight[i]);
      }
      if (morphLaplacian[i] >= 1.0) {
        //  Local max charge must be higher than 1.5 % of the max and
        //  the curvature must be greater than 50% of the peak
        // Inv ??? if ((q[i] > 0.015 * qMax) || (fabs(laplacian[i]) > (0.5 * q[i]))) {
        if (q[i] > 0.015 * qMax) {
          newPixelIdx.push_back(i);
          if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
            printf(
              "    Laplacian i=%d, x[i]=%6.3f, y[i]=%6.3f, z[i]=%6.3f, "
              "count=%d, morphLapl[i]=%6.3f, lapl[i]=%6.3f, weight[i]=%6.3f",
              i, x[i], y[i], q[i], count, morphLaplacian[i], laplacian[i],
              weight[i]);
            printf("  Selected %d\n", i);
          }
        }
        // Invalid the neihbors
        // they can't be a maximun
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i); *neigh_ptr != -1;
             neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          alreadyDone[v] = 1;
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
  // Suppress local max. whose charge is less of 1%
  // of the max charge of local Max
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
      // Compute the barycenter
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
        if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::info) {
          printf("    seed selected q=%8.2f, (x,y) = (%8.3f, %8.3f)\n",
                 localMax->q[k], localMax->x[k], localMax->q[k]);
        }
        kSelected++;
      }
    }
  }
  localMax->nPads = kSelected;

  delete[] neigh;
  delete newPixels;

  return localMax;
}

Pads* Pads::clipOnLocalMax(bool extractLocalMax)
{
  // Option extractLocalMax
  //   - true: extraxt local maxima
  //   - false: filter pixels arround the maxima
  if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
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
    if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
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
          if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
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
        if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
          printf("  Selected neighbors of i=%d: ", i);
        }
        for (PadIdx_t* neigh_ptr = getNeighborListOf(neigh, i);
             *neigh_ptr != -1; neigh_ptr++) {
          PadIdx_t v = *neigh_ptr;
          if (alreadySelect[v] == 0) {
            alreadySelect[v] = 1;
            newPixelIdx.push_back(v);
            if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
              printf("%d, ", v);
            }
          }
        }
        if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
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
          if (ClusterConfig::EMLocalMaxLog >= ClusterConfig::detail) {
            printf("  add a seed q=%9.4f, (x,y) = (%9.4f, %9.4f)\n",
                   localMax->q[k], localMax->x[k], localMax->q[k]);
          }
          kSelected++;
        }
      }
    }
    localMax->nPads = kSelected;
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
  if (pads.mode == xydxdyMode) {
    for (int i = 0; i < pads.nPads; i++) {
      printf("  pads i=%3d: x=%3.5f, dx=%3.5f, y=%3.5f, dy=%3.5f\n", i,
             pads.x[i], pads.dx[i], pads.y[i], pads.dy[i]);
    }
  } else {
    for (int i = 0; i < pads.nPads; i++) {
      printf("  pads i=%3d: xInf=%3.5f, xSup=%3.5f, yInf=%3.5f, ySup=%3.5f\n",
             i, pads.x[i], pads.dx[i], pads.y[i], pads.dy[i]);
    }
  }
}

Pads::~Pads() { release(); }

} // namespace mch
} // namespace o2
