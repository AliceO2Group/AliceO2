// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPC3DCylindricalInterpolatorIrregular.cxx
/// \brief Irregular grid interpolator for cylindrical coordinate with r,phi,z different coordinates
///        RBF-based interpolation
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Jan 5, 2016

#include "TMath.h"
#include "TVector.h"
#include "TVectorD.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TDecompSVD.h"
#include "AliTPCPoissonSolver.h"
#include "AliTPC3DCylindricalInterpolatorIrregular.h"
#include <cstdlib>

/// \cond CLASSIMP3
ClassImp(AliTPC3DCylindricalInterpolatorIrregular);
/// \endcond

/// constructor
///
/// \param nRRow
/// \param nZColumn
/// \param nPhiSlice
/// \param rStep
/// \param zStep
/// \param phiStep
/// \param type
AliTPC3DCylindricalInterpolatorIrregular::AliTPC3DCylindricalInterpolatorIrregular(
  Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice, Int_t rStep, Int_t zStep, Int_t phiStep, Int_t type)
{
  fOrder = 1;
  fIsAllocatingLookUp = kFALSE;
  fMinZIndex = 0;
  fNR = nRRow;
  fNZ = nZColumn;
  fNPhi = nPhiSlice;
  fNGridPoints = nRRow * nZColumn * nPhiSlice;

  fRBFWeightLookUp = new Int_t[nRRow * nZColumn * nPhiSlice];

  Int_t nd = rStep * zStep * phiStep;
  fStepR = rStep;
  fStepZ = zStep;
  fStepPhi = phiStep;
  fNRBFpoints = nRRow * nZColumn * nPhiSlice * nd;

  fType = type;
  fRBFWeight = new Double_t[nRRow * nZColumn * nPhiSlice * nd];
  for (Int_t i = 0; i < nRRow * nZColumn * nPhiSlice; i++) {
    fRBFWeightLookUp[i] = 0;
  }

  SetKernelType(kRBFInverseMultiQuadratic);
}

/// constructor
///
AliTPC3DCylindricalInterpolatorIrregular::AliTPC3DCylindricalInterpolatorIrregular()
{
  fOrder = 1;
  fIsAllocatingLookUp = kFALSE;

  fMinZIndex = 0;
}

/// destructor
///
AliTPC3DCylindricalInterpolatorIrregular::~AliTPC3DCylindricalInterpolatorIrregular()
{

  delete fValue;
  delete fRList;
  delete fPhiList;
  delete fZList;

  if (fKDTreeIrregularPoints) {
    delete[] fKDTreeIrregularPoints;
    delete fKDTreeIrregularRoot;
  }
  delete[] fRBFWeightLookUp;
  delete[] fRBFWeight;
}

/// irregular grid interpolation with IDW (inverse distance weight)
///
/// \param r
/// \param z
/// \param phi
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \return
Double_t
  AliTPC3DCylindricalInterpolatorIrregular::Interpolate3DTableCylIDW(
    Double_t r, Double_t z, Double_t phi, Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep,
    Int_t zStep)
{
  Double_t r0, z0, phi0, d;
  Double_t MIN_DIST = 1e-3;
  Double_t val = 0.0;
  Int_t startPhi = phiIndex - phiStep / 2;
  Int_t indexPhi;
  Int_t startR = rIndex - rStep / 2;
  Int_t startZ = zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }
  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  Int_t index;
  Double_t sum_w = 0.0;
  Double_t sum_d = 0.0;
  Double_t shortest_d = 10000.0;
  Int_t new_rIndex = 0;
  Int_t new_zIndex = 0;
  Int_t new_phiIndex = 0;

  for (Int_t iPhi = startPhi; iPhi < startPhi + phiStep; iPhi++) {
    indexPhi = iPhi % fNPhi;
    for (Int_t index_r = startR; index_r < startR + rStep; index_r++) {
      for (Int_t index_z = startZ; index_z < startZ + zStep; index_z++) {
        // check for the closest poInt_t
        index = indexPhi * (fNZ * fNR) + index_r * fNZ + index_z;

        r0 = fRList[index];
        z0 = fZList[index];
        phi0 = fPhiList[index];

        d = Distance(r0, phi0, z0, r, phi, z);
        if (d < shortest_d) {
          shortest_d = d;
          new_rIndex = index_r;
          new_phiIndex = indexPhi;
          new_zIndex = index_z;
        }
      }
    }
  }

  phiStep = 3;
  rStep = 3;
  startPhi = new_phiIndex - phiStep / 2;
  startR = new_rIndex - rStep / 2;
  startZ = new_zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }
  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  for (Int_t iPhi = startPhi; iPhi < startPhi + phiStep; iPhi++) {
    indexPhi = iPhi % fNPhi;
    for (Int_t index_r = startR; index_r < startR + rStep; index_r++) {
      for (Int_t index_z = startZ; index_z < startZ + zStep; index_z++) {
        // check for the closest poInt_t
        index = indexPhi * (fNZ * fNR) + index_r * fNZ + index_z;

        r0 = fRList[indexPhi * (fNR * fNZ) + index_r * fNZ + index_z];
        z0 = fZList[indexPhi * (fNR * fNZ) + index_r * fNZ + index_z];
        phi0 = fPhiList[indexPhi * (fNR * fNZ) + index_r * fNZ + index_z];
        d = Distance(r0, phi0, z0, r, phi, z);
        if (d < MIN_DIST) {
          return fValue[index];
        }
        d = 1.0 / d;
        sum_w += (fValue[index] * d * d * d * d);
        sum_d += d * d * d * d;
      }
    }
  }
  return (sum_w / sum_d);
}

/// distance in Cyl coordinate
///
/// \param r0
/// \param phi0
/// \param z0
/// \param r
/// \param phi
/// \param z
/// \return
Double_t
  AliTPC3DCylindricalInterpolatorIrregular::Distance(Double_t r0, Double_t phi0, Double_t z0, Double_t r, Double_t phi,
                                                     Double_t z)
{
  if (phi < 0) {
    phi = TMath::TwoPi() + phi;
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi();
  }

  if (phi0 < 0) {
    phi0 = TMath::TwoPi() + phi0;
  }
  if (phi0 > TMath::TwoPi()) {
    phi0 = phi0 - TMath::TwoPi();
  }

  Double_t dPhi = phi - phi0;
  if (dPhi > TMath::Pi()) {
    dPhi = TMath::TwoPi() - dPhi;
  }
  if (dPhi < -TMath::Pi()) {
    dPhi = TMath::TwoPi() + dPhi;
  }

  Double_t ret = r * r + r0 * r0 - 2 * r0 * r * TMath::Cos(dPhi) + (z - z0) * (z - z0);

  return TMath::Sqrt(ret);
}

/// main operation
/// interpolation by RBF
///
/// \param r
/// \param z
/// \param phi
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radiusRBF0
/// \return
Double_t
  AliTPC3DCylindricalInterpolatorIrregular::Interpolate3DTableCylRBF(
    Double_t r, Double_t z, Double_t phi, Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep,
    Int_t zStep, Double_t radiusRBF0)
{
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNR - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZ - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhi;
  Double_t r0, z0, phi0, d;
  Double_t MIN_DIST = 1e-3;
  Double_t val = 0.0;
  Int_t startPhi = phiIndex - phiStep / 2;
  Int_t indexPhi;
  Int_t startR = rIndex - rStep / 2;
  Int_t startZ = zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }
  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  Int_t index;
  Double_t sum_w = 0.0;
  Double_t sum_d = 0.0;
  Double_t shortest_d = 10000.0;
  Int_t new_rIndex = 0;
  Int_t new_zIndex = 0;
  Int_t new_phiIndex = 0;

  for (Int_t iPhi = startPhi; iPhi < startPhi + phiStep; iPhi++) {
    indexPhi = iPhi % fNPhi;
    for (Int_t index_r = startR; index_r < startR + rStep; index_r++) {
      for (Int_t index_z = startZ; index_z < startZ + zStep; index_z++) {
        // check for the closest poInt_t
        index = indexPhi * (fNZ * fNR) + index_r * fNZ + index_z;

        r0 = fRList[index];
        z0 = fZList[index];
        phi0 = fPhiList[index];

        d = Distance(r0, phi0, z0, r, phi, z);
        if (d < shortest_d) {
          shortest_d = d;
          new_rIndex = index_r;
          new_phiIndex = indexPhi;
          new_zIndex = index_z;
        }
      }
    }
  }

  index = new_phiIndex * (fNZ * fNR) + new_rIndex * fNZ + new_zIndex;
  phiStep = fStepPhi;
  rStep = fStepR;
  zStep = fStepZ;
  startPhi = new_phiIndex - phiStep / 2;

  startR = new_rIndex - rStep / 2;
  startZ = new_zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }

  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  Double_t* w;

  //Int_t nd = (phiStep-1) + (rStep-1) + (zStep-1) + 1;
  Int_t nd = phiStep * rStep * zStep;

  w = new Double_t[nd];

  Float_t minTemp, minTemp2;

  radiusRBF0 = GetRadius0RBF(new_rIndex, new_phiIndex, new_zIndex);

  if (fType == 1) {

    for (Int_t i = 0; i < nd; i++) {
      w[i] = 0.0;
    }
    GetRBFWeight(new_rIndex, new_zIndex, new_phiIndex, rStep, phiStep, zStep, radiusRBF0, 0, w);
    val = InterpRBF(r, phi, z, startR, startPhi, startZ, rStep, phiStep, zStep, radiusRBF0, 0, w);
  } else {
    GetRBFWeightHalf(new_rIndex, new_zIndex, new_phiIndex, rStep, phiStep, zStep, radiusRBF0, 0, w);
    val = InterpRBFHalf(r, phi, z, startR, startPhi, startZ, rStep, phiStep, zStep, radiusRBF0, 0, w);
  }
  delete[] w;
  return val;
}

/// Search nearest point at grid
/// \param n
/// \param xArray
/// \param offset
/// \param x
/// \param low
void AliTPC3DCylindricalInterpolatorIrregular::Search(Int_t n, Double_t* xArray, Int_t offset, Double_t x, Int_t& low)
{
  /// Search an ordered table by starting at the most recently used poInt_t

  Long_t middle, high;
  Int_t ascend = 0, increment = 1;

  if (xArray[(n - 1) * offset] >= xArray[0 * offset]) {
    ascend = 1; // Ascending ordered table if true
  }
  if (low < 0 || low > n - 1) {
    low = -1;
    high = n;
  } else { // Ordered Search phase
    if ((Int_t)(x >= xArray[low * offset]) == ascend) {
      if (low == n - 1) {
        return;
      }
      high = low + 1;
      while ((Int_t)(x >= xArray[high * offset]) == ascend) {
        low = high;
        increment *= 2;
        high = low + increment;
        if (high > n - 1) {
          high = n;
          break;
        }
      }
    } else {
      if (low == 0) {
        low = -1;
        return;
      }
      high = low - 1;
      while ((Int_t)(x < xArray[low * offset]) == ascend) {
        high = low;
        increment *= 2;
        if (increment >= high) {
          low = -1;
          break;
        } else {
          low = high - increment;
        }
      }
    }
  }

  while ((high - low) != 1) { // Binary Search Phase
    middle = (high + low) / 2;
    if ((Int_t)(x >= xArray[middle * offset]) == ascend) {
      low = middle;
    } else {
      high = middle;
    }
  }

  if (x > xArray[n - 1]) {
    low = n;
  }
  if (x < xArray[0]) {
    low = -1;
  }
}

/// get value, interpolation with RBF
///
/// \param r
/// \param phi
/// \param z
/// \param rIndex
/// \param phiIndex
/// \param zIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \return
Double_t AliTPC3DCylindricalInterpolatorIrregular::GetValue(
  Double_t r, Double_t phi, Double_t z, Int_t rIndex, Int_t phiIndex, Int_t zIndex, Int_t rStep, Int_t phiStep,
  Int_t zStep)
{

  fMinZIndex = 0;
  return Interpolate3DTableCylRBF(r, z, phi, rIndex, zIndex, phiIndex, rStep, phiStep, zStep, 0.0);
}

/// get value
///
/// \param r
/// \param phi
/// \param z
/// \param rIndex
/// \param phiIndex
/// \param zIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param minZColumnIndex
/// \return
Double_t AliTPC3DCylindricalInterpolatorIrregular::GetValue(
  Double_t r, Double_t phi, Double_t z, Int_t rIndex, Int_t phiIndex, Int_t zIndex, Int_t rStep, Int_t phiStep,
  Int_t zStep, Int_t minZColumnIndex)
{
  fMinZIndex = minZColumnIndex;
  return Interpolate3DTableCylRBF(r, z, phi, rIndex, zIndex, phiIndex, rStep, phiStep, zStep, 0.0);
}

// GetValue using searching at KDTree
Double_t AliTPC3DCylindricalInterpolatorIrregular::GetValue(
  Double_t r, Double_t phi, Double_t z)
{

  KDTreeNode n;
  n.pR = &r;
  n.pPhi = &phi;
  n.pZ = &z;
  KDTreeNode* nearest;
  Double_t dist;
  dist = 100000000.0;
  Int_t startIndex = 0; // Z
  Int_t dim = 3;        // dimenstion
  KDTreeNearest(fKDTreeIrregularRoot, &n, startIndex, dim, &nearest, &dist);
  return Interpolate3DTableCylRBF(r, z, phi, nearest);
}
/// Set value and distorted point for irregular grid interpolation
///
/// \param matrixRicesValue
/// \param matrixRicesRPoint
/// \param matrixRicesPhiPoint
/// \param matrixRicesZPoint
void AliTPC3DCylindricalInterpolatorIrregular::SetValue(
  TMatrixD** matrixRicesValue, TMatrixD** matrixRicesRPoint, TMatrixD** matrixRicesPhiPoint,
  TMatrixD** matrixRicesZPoint)
{
  Int_t indexInner;
  Int_t index;

  if (!fIsAllocatingLookUp) {
    fValue = new Double_t[fNPhi * fNR * fNZ];
    fRList = new Double_t[fNPhi * fNR * fNZ];
    fPhiList = new Double_t[fNPhi * fNR * fNZ];
    fZList = new Double_t[fNPhi * fNR * fNZ];
    fIsAllocatingLookUp = kTRUE;
  }

  for (Int_t m = 0; m < fNPhi; m++) {
    indexInner = m * fNR * fNZ;
    TMatrixD* mat = matrixRicesValue[m];
    TMatrixD* matrixR = matrixRicesRPoint[m];
    TMatrixD* matrixPhi = matrixRicesPhiPoint[m];
    TMatrixD* matrixZ = matrixRicesZPoint[m];

    for (Int_t i = 0; i < fNR; i++) {
      index = indexInner + i * fNZ;
      for (Int_t j = 0; j < fNZ; j++) {
        fValue[index + j] = (*mat)(i, j);

        fRList[index + j] = (*matrixR)(i, j);
        fPhiList[index + j] = (*matrixPhi)(i, j);
        fZList[index + j] = (*matrixZ)(i, j);
      }
    }
  }
  // KD Tree  is used for look-up a point to irregular grid to find
  // closest neughboor point
  InitKDTree();
  InitRBFWeight();
}

/// init RBF Weights assume value already been set
///
void AliTPC3DCylindricalInterpolatorIrregular::InitRBFWeight()
{

  Int_t indexInner;
  Int_t rIndex;
  Int_t index;
  Int_t startR;
  Int_t nd;

  const Double_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNR - 1);
  const Double_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZ - 1);
  const Double_t gridSizePhi = TMath::TwoPi() / fNPhi;

  Float_t r0;
  Double_t radiusRBF0, minTemp, minTemp2;

  nd = fStepR * fStepPhi * fStepZ;
  for (Int_t m = 0; m < fNPhi; m++) {
    indexInner = m * fNR * fNZ;
    for (Int_t i = 0; i < fNR; i++) {
      rIndex = indexInner + i * fNZ;

      startR = i - fStepR / 2;

      if (startR < 0) {
        startR = 0;
      }
      if (startR + fStepR >= fNR) {
        startR = fNR - fStepR;
      }

      for (Int_t j = 0; j < fNZ; j++) {
        index = rIndex + j;

        radiusRBF0 = GetRadius0RBF(i, j, m);

        RBFWeight(
          i,
          j,
          m,
          fStepR,
          fStepPhi,
          fStepZ,
          radiusRBF0,
          fKernelType,
          &fRBFWeight[index * nd]);
        fRBFWeightLookUp[index] = 1;
      }
    }
  }
}

/// Set value and distorted Point
///
/// \param matrixRicesValue
/// \param matrixRicesRPoint
/// \param matrixRicesPhiPoint
/// \param matrixRicesZPoint
/// \param jy
void AliTPC3DCylindricalInterpolatorIrregular::SetValue(
  TMatrixD** matrixRicesValue, TMatrixD** matrixRicesRPoint, TMatrixD** matrixRicesPhiPoint,
  TMatrixD** matrixRicesZPoint,
  Int_t jy)
{
  Int_t indexInner;
  Int_t index;

  if (!fIsAllocatingLookUp) {
    fValue = new Double_t[fNPhi * fNR * fNZ];
    fRList = new Double_t[fNPhi * fNR * fNZ];
    fPhiList = new Double_t[fNPhi * fNR * fNZ];
    fZList = new Double_t[fNPhi * fNR * fNZ];

    fIsAllocatingLookUp = kTRUE;
  }

  for (Int_t m = 0; m < fNPhi; m++) {
    indexInner = m * fNR * fNZ;
    TMatrixD* mat = matrixRicesValue[m];
    TMatrixD* matrixR = matrixRicesRPoint[m];
    TMatrixD* matrixPhi = matrixRicesPhiPoint[m];
    TMatrixD* matrixZ = matrixRicesZPoint[m];

    for (Int_t i = 0; i < fNR; i++) {
      index = indexInner + i * fNZ;
      fValue[index + jy] = (*mat)(i, jy);
      fRList[index + jy] = (*matrixR)(i, jy);
      fPhiList[index + jy] = (*matrixPhi)(i, jy);
      fZList[index + jy] = (*matrixZ)(i, jy);
    }
  }
}

/// calculate
/// RBFWeight for all points in the interpolation
///
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param w
void AliTPC3DCylindricalInterpolatorIrregular::RBFWeight(
  Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep, Int_t zStep, Double_t radius0,
  Int_t kernelType, Double_t* w)
{

  Double_t* a;
  Int_t i;
  Int_t j;
  Int_t k;
  Int_t ii;
  Int_t jj;
  Int_t kk;

  Int_t index0, index1;
  Int_t indexCyl0, indexCyl1;
  Double_t* r;
  Double_t* v;

  Double_t phi0;
  Double_t z0;
  Double_t r0;

  Double_t phi1;
  Double_t z1;
  Double_t r1;

  Int_t nd = rStep * phiStep * zStep;

  a = new Double_t[nd * nd];
  r = new Double_t[nd];
  v = new Double_t[nd];

  Int_t startPhi = phiIndex - phiStep / 2;
  Int_t indexPhi;
  Int_t indexPhi1;

  Int_t startR = rIndex - rStep / 2;
  Int_t startZ = zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }

  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  index0 = 0;

  for (i = startPhi; i < startPhi + phiStep; i++) {
    indexPhi = i % fNPhi;

    for (j = startR; j < startR + rStep; j++) {
      for (k = startZ; k < startZ + zStep; k++) {
        indexCyl0 = indexPhi * fNR * fNZ + j * fNZ + k;

        r0 = fRList[indexCyl0];
        z0 = fZList[indexCyl0];
        phi0 = fPhiList[indexCyl0];

        index1 = 0;
        for (ii = startPhi; ii < startPhi + phiStep; ii++) {
          indexPhi1 = ii % fNPhi;
          for (jj = startR; jj < startR + rStep; jj++) {
            for (kk = startZ; kk < startZ + zStep; kk++) {
              indexCyl1 = indexPhi1 * fNR * fNZ + jj * fNZ + kk;
              r1 = fRList[indexCyl1];
              z1 = fZList[indexCyl1];
              phi1 = fPhiList[indexCyl1];
              r[index1] = Distance(r0, phi0, z0, r1, phi1, z1);

              index1++;
            }
          }
        }

        Phi(nd, r, radius0, v);

        index1 = 0;
        for (ii = startPhi; ii < startPhi + phiStep; ii++) {
          indexPhi1 = ii % fNPhi;
          for (jj = startR; jj < startR + rStep; jj++) {
            for (kk = startZ; kk < startZ + zStep; kk++) {
              a[index0 * nd + index1] = v[index1];
              index1++;
            }
          }
        }
        w[index0] = fValue[indexCyl0];
        index0++;
      }
    }
  }

  TMatrixD mat_a;
  mat_a.Use(nd, nd, a);
  TVectorD vec_w;
  vec_w.Use(nd, w);
  TDecompSVD svd(mat_a);

  svd.Solve(vec_w);

  delete[] a;
  delete[] r;
  delete[] v;
}

/// rbf1
/// \param n
/// \param r
/// \param r0
/// \param v
void AliTPC3DCylindricalInterpolatorIrregular::rbf1(Int_t n, Double_t r[], Double_t r0, Double_t v[])
{
  Int_t i;

  for (i = 0; i < n; i++) {
    v[i] = sqrt(1 + (r[i] * r[i] + r0 * r0));
  }
  return;
}

/// rbf2
/// \param n
/// \param r
/// \param r0
/// \param v

void AliTPC3DCylindricalInterpolatorIrregular::rbf2(Int_t n, Double_t r[], Double_t r0, Double_t v[])
{
  Int_t i;

  for (i = 0; i < n; i++) {
    v[i] = 1.0 / sqrt(1 + (r[i] * r[i] + r0 * r0));
  }
  return;
}

/// rbf3
/// \param n
/// \param r
/// \param r0
/// \param v
void AliTPC3DCylindricalInterpolatorIrregular::rbf3(Int_t n, Double_t r[], Double_t r0, Double_t v[])
{
  Int_t i;

  for (i = 0; i < n; i++) {
    if (r[i] <= 0.0) {
      v[i] = 0.0;
    } else {
      v[i] = r[i] * r[i] * log(r[i] / r0);
    }
  }
  return;
}

/// rbf4
/// \param n
/// \param r
/// \param r0
/// \param v
void AliTPC3DCylindricalInterpolatorIrregular::rbf4(Int_t n, Double_t r[], Double_t r0, Double_t v[])
{
  Int_t i;

  for (i = 0; i < n; i++) {
    v[i] = TMath::Exp(-0.5 * r[i] * r[i] / (r0 * r0));
  }
  return;
}

// RBF based interpolation
// return interpolated value
///
/// \param r
/// \param phi
/// \param z
/// \param startR
/// \param startPhi
/// \param startZ
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param weight
/// \return
Double_t AliTPC3DCylindricalInterpolatorIrregular::InterpRBF(
  Double_t r, Double_t phi, Double_t z, Int_t startR, Int_t startPhi, Int_t startZ, Int_t rStep, Int_t phiStep,
  Int_t zStep, Double_t radius0, Int_t kernelType, Double_t* weight)
{
  Double_t interpVal = 0.0;
  Double_t r0, z0, phi0;
  Double_t* dList;
  Double_t* v;

  Int_t indexCyl0, index0, indexPhi;

  Int_t nd = rStep * phiStep * zStep;

  dList = new Double_t[nd];
  v = new Double_t[nd];

  index0 = 0;
  for (Int_t i = startPhi; i < startPhi + phiStep; i++) {
    indexPhi = i % fNPhi;

    for (Int_t j = startR; j < startR + rStep; j++) {
      for (Int_t k = startZ; k < startZ + zStep; k++) {

        indexCyl0 = indexPhi * fNR * fNZ + j * fNZ + k;

        r0 = fRList[indexCyl0];
        z0 = fZList[indexCyl0];
        phi0 = fPhiList[indexCyl0];

        dList[index0] = Distance(r, phi, z, r0, phi0, z0);
        index0++;
      }
    }
  }

  Phi(nd, dList, radius0, v);

  TVectorD vec_v;
  vec_v.Use(nd, v);

  TVectorD vec_w;
  vec_w.Use(nd, weight);

  interpVal = vec_v * vec_w;
  delete[] v;
  delete[] dList;
  return interpVal;
}

// calculate
// RBFWeight for all points in the interpolation
///
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param w
void AliTPC3DCylindricalInterpolatorIrregular::GetRBFWeight(
  Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep, Int_t zStep, Double_t radius0,
  Int_t kernelType, Double_t* w)
{

  Int_t index = phiIndex * fNR * fNZ + rIndex * fNZ + zIndex;
  if (fRBFWeightLookUp[index] == 0) {
    RBFWeight(rIndex, zIndex, phiIndex, rStep, phiStep, zStep, radius0, kernelType, w);

    fRBFWeightLookUp[index] = 1;
    Int_t nd = rStep * zStep * phiStep;

    for (Int_t i = 0; i < nd; i++) {
      fRBFWeight[index * nd + i] = w[i];
    }
  } else {

    Int_t ndw = rStep * zStep * phiStep;

    Int_t nd = fStepR * fStepZ * fStepPhi;
    Int_t indexWeight = phiIndex * fNR * fNZ * nd + rIndex * fNZ * nd + zIndex * nd;

    for (Int_t i = 0; i < nd; i++) {
      w[i] = fRBFWeight[indexWeight + i];
    }
  }
}

// calculate
// RBFWeight for all points in the interpolation
///
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param w
void AliTPC3DCylindricalInterpolatorIrregular::GetRBFWeightHalf(
  Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep, Int_t zStep, Double_t radius0,
  Int_t kernelType, Double_t* w)
{

  Int_t index = phiIndex * fNR * fNZ + rIndex * fNZ + zIndex;

  if (fRBFWeightLookUp[index] == 0) {
    RBFWeightHalf(rIndex, zIndex, phiIndex, rStep, phiStep, zStep, radius0, kernelType, w);

    if ((rStep == fStepR) && (zStep == fStepZ) && (phiStep == fStepPhi) && (zIndex > fMinZIndex + fStepZ)) {
      fRBFWeightLookUp[index] = 1;
      // copy to lookup
      Int_t nd = rStep + zStep + phiStep - 2;

      for (Int_t i = 0; i < nd; i++) {
        fRBFWeight[index * nd + i] = w[i];
      }
    }
  } else {

    //Int_t ndw = rStep*zStep*phiStep;
    Int_t nd = rStep + zStep + phiStep - 2;
    Int_t indexWeight = phiIndex * fNR * fNZ * nd + rIndex * fNZ * nd + zIndex * nd;

    for (Int_t i = 0; i < nd; i++) {
      w[i] = fRBFWeight[indexWeight + i];
    }
  }
}

// calculate
// RBFWeight for all points in the interpolation
// half cubes (not included
///
/// \param rIndex
/// \param zIndex
/// \param phiIndex
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param w
void AliTPC3DCylindricalInterpolatorIrregular::RBFWeightHalf(
  Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t rStep, Int_t phiStep, Int_t zStep, Double_t radius0,
  Int_t kernelType, Double_t* w)
{
  Double_t* a;
  Int_t i;
  Int_t j;
  Int_t k;
  Int_t ii;
  Int_t jj;
  Int_t kk;

  Int_t index0, index1;
  Int_t indexCyl0, indexCyl1;
  Double_t* r;
  Double_t* v;

  Double_t phi0;
  Double_t z0;
  Double_t r0;

  Double_t phi1;
  Double_t z1;
  Double_t r1;

  Int_t nd = (rStep - 1) + (phiStep - 1) + (zStep - 1) + 1;

  a = new Double_t[nd * nd];
  r = new Double_t[nd];
  v = new Double_t[nd];

  Int_t startPhi = phiIndex - phiStep / 2;
  Int_t indexPhi;
  Int_t indexPhi1;

  Int_t startR = rIndex - rStep / 2;
  Int_t startZ = zIndex - zStep / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }

  if (startR < 0) {
    startR = 0;
  }
  if (startR + rStep >= fNR) {
    startR = fNR - rStep;
  }

  if (startZ < fMinZIndex) {
    startZ = fMinZIndex;
  }
  if (startZ + zStep >= fNZ) {
    startZ = fNZ - zStep;
  }

  index0 = 0;

  for (i = startPhi; i < startPhi + phiStep; i++) {
    indexPhi = i % fNPhi;

    for (j = startR; j < startR + rStep; j++) {
      for (k = startZ; k < startZ + zStep; k++) {

        if (
          (i == (startPhi + phiStep / 2) && j == (startR + rStep / 2)) ||
          (i == (startPhi + phiStep / 2) && k == (startZ + zStep / 2)) ||
          (j == (startR + rStep / 2) && k == (startZ + zStep / 2))) {
          indexCyl0 = indexPhi * fNR * fNZ + j * fNZ + k;

          r0 = fRList[indexCyl0];
          z0 = fZList[indexCyl0];
          phi0 = fPhiList[indexCyl0];

          index1 = 0;
          for (ii = startPhi; ii < startPhi + phiStep; ii++) {
            indexPhi1 = ii % fNPhi;
            for (jj = startR; jj < startR + rStep; jj++) {
              for (kk = startZ; kk < startZ + zStep; kk++) {
                if (
                  (ii == (startPhi + phiStep / 2) && jj == (startR + rStep / 2)) ||
                  (ii == (startPhi + phiStep / 2) && kk == (startZ + zStep / 2)) ||
                  (jj == (startR + rStep / 2) && kk == (startZ + zStep / 2))) {

                  indexCyl1 = indexPhi1 * fNR * fNZ + jj * fNZ + kk;
                  r1 = fRList[indexCyl1];
                  z1 = fZList[indexCyl1];
                  phi1 = fPhiList[indexCyl1];

                  r[index1] = Distance(r0, phi0, z0, r1, phi1, z1);
                  index1++;
                }
              }
            }
          }

          Phi(nd, r, radius0, v);

          index1 = 0;
          for (ii = startPhi; ii < startPhi + phiStep; ii++) {
            indexPhi1 = ii % fNPhi;
            for (jj = startR; jj < startR + rStep; jj++) {
              for (kk = startZ; kk < startZ + zStep; kk++) {
                if (
                  (ii == (startPhi + phiStep / 2) && jj == (startR + rStep / 2)) ||
                  (ii == (startPhi + phiStep / 2) && kk == (startZ + zStep / 2)) ||
                  (jj == (startR + rStep / 2) && kk == (startZ + zStep / 2))) {
                  a[index0 * nd + index1] = v[index1];
                  index1++;
                }
              }
            }
          }

          w[index0] = fValue[indexCyl0];
          index0++;
        }
      }
    }
  }

  TMatrixD mat_a;
  mat_a.Use(nd, nd, a);
  TVectorD vec_w;

  vec_w.Use(nd, w);
  TDecompSVD svd(mat_a);

  svd.Solve(vec_w);

  delete[] a;
  delete[] r;
  delete[] v;
}

// RBF based interpolation
// return interpolated value
// half points
///
/// \param r
/// \param phi
/// \param z
/// \param startR
/// \param startPhi
/// \param startZ
/// \param rStep
/// \param phiStep
/// \param zStep
/// \param radius0
/// \param kernelType
/// \param weight
/// \return
Double_t AliTPC3DCylindricalInterpolatorIrregular::InterpRBFHalf(
  Double_t r, Double_t phi, Double_t z, Int_t startR, Int_t startPhi, Int_t startZ, Int_t rStep, Int_t phiStep,
  Int_t zStep, Double_t radius0, Int_t kernelType, Double_t* weight)
{
  Double_t interpVal = 0.0;
  Double_t r0, z0, phi0;
  Double_t* dList;
  Double_t* v;

  Int_t indexCyl0, index0, indexPhi;

  //	Int_t nd = rStep * phiStep * zStep;
  Int_t nd = (rStep - 1) + (phiStep - 1) + (zStep - 1) + 1;

  dList = new Double_t[nd];
  v = new Double_t[nd];

  index0 = 0;
  for (Int_t i = startPhi; i < startPhi + phiStep; i++) {
    indexPhi = i % fNPhi;

    for (Int_t j = startR; j < startR + rStep; j++) {
      for (Int_t k = startZ; k < startZ + zStep; k++) {
        if (
          (i == (startPhi + phiStep / 2) && j == (startR + rStep / 2)) ||
          (i == (startPhi + phiStep / 2) && k == (startZ + zStep / 2)) ||
          (j == (startR + rStep / 2) && k == (startZ + zStep / 2))) {

          indexCyl0 = indexPhi * fNR * fNZ + j * fNZ + k;

          r0 = fRList[indexCyl0];
          z0 = fZList[indexCyl0];
          phi0 = fPhiList[indexCyl0];

          dList[index0] = Distance(r, phi, z, r0, phi0, z0);
          index0++;
        }
      }
    }
  }

  Phi(nd, dList, radius0, v);

  TVectorD vec_v;
  vec_v.Use(nd, v);

  TVectorD vec_w;
  vec_w.Use(nd, weight);

  interpVal = vec_v * vec_w;
  delete[] v;
  delete[] dList;
  return interpVal;
}

// set Radius0
///
/// \param n
/// \param r
/// \param r0
/// \param v
void AliTPC3DCylindricalInterpolatorIrregular::Phi(Int_t n, Double_t r[], Double_t r0, Double_t v[])
{

  switch (fKernelType) {
    case kRBFMultiQuadratic:
      rbf1(n, r, r0, v);
      break;
    case kRBFInverseMultiQuadratic:
      rbf2(n, r, r0, v);
      break;
    case kRBFThinPlateSpline:
      rbf3(n, r, r0, v);
      break;
    case kRBFGaussian:
      rbf4(n, r, r0, v);
      break;

    default:
      rbf1(n, r, r0, v);
      break;
  }
}

//
Double_t
  AliTPC3DCylindricalInterpolatorIrregular::GetRadius0RBF(const Int_t rIndex, const Int_t phiIndex, const Int_t zIndex)
{
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNR - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZ - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhi;
  Int_t startPhi = phiIndex - fStepPhi / 2;

  Int_t startR = rIndex - fStepR / 2;
  Int_t startZ = zIndex - fStepZ / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }

  if (startR < 0) {
    startR = 0;
  }
  if (startR + fStepR >= fNR) {
    startR = fNR - fStepR;
  }

  if (startZ < 0) {
    startZ = 0;
  }
  if (startZ + fStepZ >= fNZ) {
    startZ = fNZ - fStepZ;
  }

  Double_t r0 = AliTPCPoissonSolver::fgkIFCRadius + (startR * gridSizeR);
  Double_t phi0 = startPhi * gridSizePhi;
  Double_t z0 = startZ * gridSizeZ;

  Double_t r1 = AliTPCPoissonSolver::fgkIFCRadius + (startR * gridSizeR);
  Double_t phi1 = (startPhi + 1) * gridSizePhi;
  Double_t z1 = (startZ + 1) * gridSizeZ;

  if (fKernelType == kRBFThinPlateSpline) {
    r0 = AliTPCPoissonSolver::fgkIFCRadius + ((startR - 1) * gridSizeR);
  } else {
    r0 = AliTPCPoissonSolver::fgkIFCRadius + (startR * gridSizeR);
  }

  return Distance(r0, 0.0, 0.0, r0 + gridSizeR, gridSizePhi, gridSizeR);
}

// make kdtree for irregular look-up
void AliTPC3DCylindricalInterpolatorIrregular::InitKDTree()
{
  Int_t count = fNR * fNZ * fNPhi;

  fKDTreeIrregularPoints = new KDTreeNode[count];

  for (Int_t i = 0; i < count; i++) {
    fKDTreeIrregularPoints[i].pR = &fRList[i];
    fKDTreeIrregularPoints[i].pZ = &fZList[i];
    fKDTreeIrregularPoints[i].pPhi = &fPhiList[i];
    fKDTreeIrregularPoints[i].index = i;
  }

  fKDTreeIrregularRoot = MakeKDTree(fKDTreeIrregularPoints, count, 0, 3);
}

// create KDTree
AliTPC3DCylindricalInterpolatorIrregular::KDTreeNode* AliTPC3DCylindricalInterpolatorIrregular::MakeKDTree(KDTreeNode* t, Int_t count, Int_t index, Int_t dim)
{
  KDTreeNode* n;

  if (!count) {
    return nullptr;
  }
  if ((n = FindMedian(t, t + count, index))) {
    index = (index + 1) % dim;
    n->left = MakeKDTree(t, (n - t), index, dim);
    n->right = MakeKDTree(n + 1, (t + count) - (n + 1), index, dim);
  }
  return n;
}

// find median
AliTPC3DCylindricalInterpolatorIrregular::KDTreeNode* AliTPC3DCylindricalInterpolatorIrregular::FindMedian(KDTreeNode* start, KDTreeNode* end, Int_t index)
{
  if (end <= start) {
    return nullptr;
  }
  if (end == start + 1) {
    return start;
  }

  KDTreeNode *p, *store, *md = start + (end - start) / 2;
  Double_t pivot;

  while (1) {
    if (index == 0) {
      pivot = *(md->pZ);
    } else if (index == 1) {
      pivot = *(md->pR);
    } else {
      pivot = *(md->pPhi);
    }

    Swap(md, end - 1);

    for (store = p = start; p < end; p++) {

      if (((index == 0) && (*(p->pZ) < pivot)) ||
          ((index == 1) && (*(p->pR) < pivot)) ||
          ((index == 2) && (*(p->pPhi) < pivot)))

      {
        if (p != store) {
          Swap(p, store);
        }
        store++;
      }
    }
    Swap(store, end - 1);

    if ((index == 0) && (*(store->pZ) == *(md->pZ))) {
      return md;
    }
    if ((index == 1) && (*(store->pR) == *(md->pR))) {
      return md;
    }
    if ((index == 2) && (*(store->pPhi) == *(md->pPhi))) {
      return md;
    }

    //	if (md->index == store->index) return md;

    if (store > md) {
      end = store;
    } else {
      start = store;
    }
  }
}

//swap
void AliTPC3DCylindricalInterpolatorIrregular::Swap(KDTreeNode* x, KDTreeNode* y)
{
  KDTreeNode* tmp = new KDTreeNode;
  tmp->pR = x->pR;
  tmp->pZ = x->pZ;
  tmp->pPhi = x->pPhi;
  tmp->index = x->index;

  x->pR = y->pR;
  x->pZ = y->pZ;
  x->pPhi = y->pPhi;
  x->index = y->index;

  y->pR = tmp->pR;
  y->pZ = tmp->pZ;
  y->pPhi = tmp->pPhi;
  y->index = tmp->index;

  delete tmp;
}

// look for nearest point
void AliTPC3DCylindricalInterpolatorIrregular::KDTreeNearest(KDTreeNode* root, KDTreeNode* nd, Int_t index, Int_t dim,
                                                             KDTreeNode** best, Double_t* best_dist)
{
  Double_t d, dx2, dx;

  if (!root) {
    return;
  }
  d = Distance(*(root->pR), *(root->pPhi), *(root->pZ), *(nd->pR), *(nd->pPhi), *(nd->pZ));
  if (index == 0) {
    dx = *(root->pZ) - *(nd->pZ);
    dx2 = Distance(*(nd->pR), *(nd->pPhi), *(root->pZ), *(nd->pR), *(nd->pPhi), *(nd->pZ));
  } else if (index == 1) {
    dx = *(root->pR) - *(nd->pR);
    dx2 = Distance(*(root->pR), *(nd->pPhi), *(nd->pZ), *(nd->pR), *(nd->pPhi), *(nd->pZ));
  } else {
    dx = *(root->pPhi) - *(nd->pPhi);
    dx2 = Distance(*(nd->pR), *(root->pPhi), *(nd->pZ), *(nd->pR), *(nd->pPhi), *(nd->pZ));
  }

  if (!*best || (d < *best_dist)) {
    *best_dist = d;
    *best = root;
  }

  if (!*best_dist) {
    return;
  }

  if (++index >= dim) {
    index = 0;
  }

  KDTreeNearest(dx > 0 ? root->left : root->right, nd, index, dim, best, best_dist);
  if (dx2 >= *best_dist) {
    return;
  }
  KDTreeNearest(dx > 0 ? root->right : root->left, nd, index, dim, best, best_dist);
}

// interpolate on the nearest neighbor of irregular grid
Double_t
  AliTPC3DCylindricalInterpolatorIrregular::Interpolate3DTableCylRBF(
    Double_t r, Double_t z, Double_t phi, KDTreeNode* nearestNode)
{
  Double_t val = 0.0;
  Int_t startPhi, startR, startZ;
  Int_t phiIndex, rIndex, zIndex;

  phiIndex = nearestNode->index / (fNR * fNZ);
  rIndex = (nearestNode->index - (phiIndex * (fNR * fNZ))) / fNZ;
  zIndex = nearestNode->index - (phiIndex * (fNR * fNZ) + rIndex * fNZ);

  startPhi = phiIndex - fStepPhi / 2;
  startR = rIndex - fStepR / 2;
  startZ = zIndex - fStepZ / 2;

  if (startPhi < 0) {
    startPhi = fNPhi + startPhi;
  }

  if (startR < 0) {
    startR = 0;
  }
  if (startR + fStepR >= fNR) {
    startR = fNR - fStepR;
  }

  if (startZ < 0) {
    startZ = 0;
  }
  if (startZ + fStepZ >= fNZ) {
    startZ = fNZ - fStepZ;
  }
  Int_t indexPhi;

  Int_t index;
  Double_t r0, z0, phi0;

  Int_t rStep = fStepR;
  Int_t zStep = fStepZ;
  Int_t phiStep = fStepPhi;

  Double_t* w;

  //Int_t nd = (phiStep-1) + (rStep-1) + (zStep-1) + 1;
  Int_t nd = fStepPhi * fStepR * fStepZ;

  w = new Double_t[nd];

  Float_t minTemp, minTemp2;

  Double_t radiusRBF0 = GetRadius0RBF(rIndex, phiIndex, zIndex);

  if (fType == 1) {

    for (Int_t i = 0; i < nd; i++) {
      w[i] = 0.0;
    }
    GetRBFWeight(rIndex, zIndex, phiIndex, rStep, phiStep, zStep, radiusRBF0, 0, w);
    val = InterpRBF(r, phi, z, startR, startPhi, startZ, rStep, phiStep, zStep, radiusRBF0, 0, w);
  } else {
    GetRBFWeightHalf(rIndex, zIndex, phiIndex, rStep, phiStep, zStep, radiusRBF0, 0, w);
    val = InterpRBFHalf(r, phi, z, startR, startPhi, startZ, rStep, phiStep, zStep, radiusRBF0, 0, w);
  }
  delete[] w;
  return val;
}
