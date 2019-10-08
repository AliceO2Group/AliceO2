// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPC3DCylindricalInterpolator.cxx
/// \brief Interpolator for cylindrical coordinate
///        this class provides: cubic spline, quadratic and linear interpolation
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Jan 5, 2016

#include "TMath.h"
#include "AliTPC3DCylindricalInterpolator.h"

/// \cond CLASSIMP
ClassImp(AliTPC3DCylindricalInterpolator);
/// \endcond

/// constructor
///
AliTPC3DCylindricalInterpolator::AliTPC3DCylindricalInterpolator()
{
  fOrder = 1;
  fIsAllocatingLookUp = kFALSE;
  fIsInitCubic = kFALSE;
}

/// destructor
///
AliTPC3DCylindricalInterpolator::~AliTPC3DCylindricalInterpolator()
{
  delete fValue;
  delete fRList;
  delete fPhiList;
  delete fZList;
  if (fIsInitCubic) {
    delete fSecondDerZ;
  }
}

/// Get interpolation value on a point in a cylindrical volume
///
/// \param r position r
/// \param phi position $\phi$
/// \param z position  z
///
/// \return interpolation value
Double_t AliTPC3DCylindricalInterpolator::GetValue(Double_t r, Double_t phi, Double_t z) const
{
  return InterpolateCylindrical(r, z, phi);
}

/// Get interpolation value on a point in a cylindrical volume
///
/// \param r Double_t position r
/// \param phi Double_t position $\phi$
/// \param z Double_t position  z
///
/// \return interpolation value
Double_t AliTPC3DCylindricalInterpolator::InterpolateCylindrical(Double_t r, Double_t z, Double_t phi) const
{
  Int_t iLow = 0, jLow = 0, m = 0;
  Int_t kLow = 0;
  Int_t index;

  // tri cubic points
  Double_t saveArray[fOrder + 1];
  Double_t savedArray[fOrder + 1];
  Double_t zListM1[fOrder + 1];
  Double_t valueM1[fOrder + 1];

  Bool_t neg = kFALSE;

  // check phi
  while (phi < 0.0) {
    phi = TMath::TwoPi() + phi;
  }
  while (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi();
  }

  // search lowest index related to r,z and phi
  Search(fNR, fRList, r, iLow);
  Search(fNZ, fZList, z, jLow);
  Search(fNPhi, fPhiList, phi, kLow);

  // order >= 3
  kLow -= (fOrder / 2);
  iLow -= (fOrder / 2);
  jLow -= (fOrder / 2);

  // check if out of range
  if (iLow < 0) {
    iLow = 0;
  }
  if (jLow < 0) {
    jLow = 0;
  }
  if (kLow < 0) {
    kLow = fNPhi + kLow;
  }
  // check if out of range
  if (iLow + fOrder >= fNR - 1) {
    iLow = fNR - 1 - fOrder;
  }
  if (jLow + fOrder >= fNZ - 1) {
    jLow = fNZ - 1 - fOrder;
  }

  // do for each
  for (Int_t k = 0; k < fOrder + 1; k++) {
    m = (kLow + k) % fNPhi;
    // interpolate
    for (Int_t i = iLow; i < iLow + fOrder + 1; i++) {
      if (fOrder < 3) {
        if (jLow >= 0) {
          index = m * (fNZ * fNR) + i * (fNZ) + jLow;
          saveArray[i - iLow] = Interpolate(&fZList[jLow], &fValue[index], z);
        } else {
          index = m * (fNZ * fNR) + i * (fNZ);
          zListM1[0] = fZList[0] - (fZList[1] - fZList[0]);
          zListM1[1] = fZList[0];
          zListM1[2] = fZList[1];
          valueM1[0] = fValue[index] - (fValue[index + 1] - fValue[index]);
          valueM1[1] = fValue[index];
          valueM1[2] = fValue[index + 1];
          saveArray[i - iLow] = Interpolate(&zListM1[0], &valueM1[0], z);
        }

      } else {
        index = m * (fNZ * fNR) + i * (fNZ);
        saveArray[i - iLow] = InterpolateCubicSpline(fZList, &fValue[index], &fSecondDerZ[index], fNZ, fNZ, fNZ,
                                                     z, 1);
      }
    }
    savedArray[k] = Interpolate(&fRList[iLow], saveArray, r);
  }
  return (InterpolatePhi(&fPhiList[0], kLow, fNPhi, savedArray, phi));
}

/// Get interpolation for 1 dimension non cyclic
///
/// \param xArray Double_t[] known position x
/// \param yArray Double_t[] known y = f(x)
/// \param x unknown position
///
/// \return interpolation value f(x)
Double_t AliTPC3DCylindricalInterpolator::Interpolate(Double_t xArray[], Double_t yArray[], Double_t x) const
{
  Double_t y;

  // if cubic spline
  if (fOrder > 2) {
    Double_t y2Array[fOrder + 1];
    InitCubicSpline(xArray, yArray, fOrder + 1, y2Array, 1);
    y = InterpolateCubicSpline(xArray, yArray, y2Array, fOrder + 1, fOrder + 1, fOrder + 1, x, 1);
  } else if (fOrder == 2) {
    // Quadratic Interpolation = 2
    y = (x - xArray[1]) * (x - xArray[2]) * yArray[0] / ((xArray[0] - xArray[1]) * (xArray[0] - xArray[2]));
    y += (x - xArray[2]) * (x - xArray[0]) * yArray[1] / ((xArray[1] - xArray[2]) * (xArray[1] - xArray[0]));
    y += (x - xArray[0]) * (x - xArray[1]) * yArray[2] / ((xArray[2] - xArray[0]) * (xArray[2] - xArray[1]));
  } else {
    // Linear Interpolation = 1
    y = yArray[0] + (yArray[1] - yArray[0]) * (x - xArray[0]) / (xArray[1] - xArray[0]);
  }
  return (y);
}

/// Get interpolation for 1 dimension cyclic
///
/// \param xArray Double_t[] known position x
/// \param yArray Double_t[] known y = f(x)
/// \param x unknown position
///
/// \return interpolation value f(x)
Double_t AliTPC3DCylindricalInterpolator::InterpolatePhi(
  Double_t xArray[], const Int_t iLow, const Int_t lenX, Double_t yArray[], Double_t x) const
{
  Int_t i0 = iLow;
  Double_t xi0 = xArray[iLow];
  Int_t i1 = (iLow + 1) % lenX;
  Double_t xi1 = xArray[i1];
  Int_t i2 = (iLow + 2) % lenX;
  Double_t xi2 = xArray[i2];

  if (fOrder <= 2) {
    if (xi1 < xi0) {
      xi1 = TMath::TwoPi() + xi1;
    }
    if (xi2 < xi1) {
      xi2 = TMath::TwoPi() + xi2;
    }
    if (x < xi0) {
      x = TMath::TwoPi() + x;
    }
  }

  Double_t y;
  if (fOrder > 2) {
    Double_t y2Array[fOrder + 1];
    Double_t xArrayTemp[fOrder + 1];
    Double_t dPhi = xArray[1] - xArray[0];
    // make list phi ascending order
    for (Int_t i = 0; i < fOrder + 1; i++) {
      xArrayTemp[i] = xArray[iLow] + (dPhi * i);
    }
    if (x < xArrayTemp[0]) {
      x = TMath::TwoPi() + x;
    }
    if (x < xArrayTemp[0] || x > xArrayTemp[fOrder]) {
      printf("x (%f) is outside of interpolation box (%f,%f)\n", x, xArrayTemp[0], xArrayTemp[fOrder]);
    }

    InitCubicSpline(xArrayTemp, yArray, fOrder + 1, y2Array, 1);
    y = InterpolateCubicSpline(xArrayTemp, yArray, y2Array, fOrder + 1, fOrder + 1, fOrder + 1, x, 1);
  } else if (fOrder == 2) { // Quadratic Interpolation = 2
    y = (x - xi1) * (x - xi2) * yArray[0] / ((xi0 - xi1) * (xi0 - xi2));
    y += (x - xi2) * (x - xi0) * yArray[1] / ((xi1 - xi2) * (xi1 - xi0));
    y += (x - xi0) * (x - xi1) * yArray[2] / ((xi2 - xi0) * (xi2 - xi1));
  } else { // Li2near Interpolation = 1
    y = yArray[0] + (yArray[1] - yArray[0]) * (x - xi0) / (xi1 - xi0);
  }
  return (y);
}

/// Solving cubic splines for system of splines
///
/// \param xArray Double_t[] known position x
/// \param yArray Double_t[] known y = f(x)
/// \param n Int_t length of splines
/// \param y2Array Double_t[] calculated $d^2Y$ spline (output)
/// \param skip memory offset for xArray
///
void AliTPC3DCylindricalInterpolator::InitCubicSpline(Double_t* xArray, Double_t* yArray, const Int_t n, Double_t* y2Array,
                                                      const Int_t skip) const
{
  Double_t u[n];
  Double_t sig, p, qn, un;

  y2Array[0] = 0.0;
  u[0] = 0.0; //natural condition

  for (Int_t i = 1; i <= n - 2; i++) {
    sig = (xArray[i] - xArray[i - 1]) / (xArray[i + 1] - xArray[i - 1]);
    p = sig * y2Array[(i - 1) * skip] + 2.0;
    y2Array[i * skip] = (sig - 1.0) / p;
    u[i] = (yArray[(i + 1) * skip] - yArray[i * skip]) / (xArray[i + 1] - xArray[i]) -
           (yArray[i * skip] - yArray[(i - 1) * skip]) / (xArray[i] - xArray[i - 1]);
    u[i] = (6.0 * u[i] / (xArray[i + 1] - xArray[i - 1]) - sig * u[i - 1]) / p;
  }

  qn = un = 0.0;

  y2Array[(n - 1) * skip] = (un - qn * u[n - 2]) / (qn * y2Array[(n - 2) * skip] + 1.0);
  for (Int_t k = n - 2; k >= 0; k--) {
    y2Array[k * skip] = y2Array[k * skip] * y2Array[(k + 1) * skip] + u[k];
  }
}

/// Solving cubic splines for system of splines
///
/// \param xArray Double_t[] known position x
/// \param yArray Double_t[] known y = f(x)
/// \param n Int_t length of splines
/// \param y2Array Double_t[] calculated $d^2Y$ spline (output)
/// \param skip memory offset for xArray
///
void AliTPC3DCylindricalInterpolator::InitCubicSpline(Double_t* xArray, Double_t* yArray, const Int_t n, Double_t* y2Array,
                                                      const Int_t skip, Double_t yp0, Double_t ypn1) const
{
  Double_t u[n];
  Double_t sig, p, qn, un;

  y2Array[0] = 0.0;
  u[0] = 0.0; //natural condition

  for (Int_t i = 1; i <= n - 2; i++) {
    sig = (xArray[i] - xArray[i - 1]) / (xArray[i + 1] - xArray[i - 1]);
    p = sig * y2Array[(i - 1) * skip] + 2.0;
    y2Array[i * skip] = (sig - 1.0) / p;
    u[i] = (yArray[(i + 1) * skip] - yArray[i * skip]) / (xArray[i + 1] - xArray[i]) -
           (yArray[i * skip] - yArray[(i - 1) * skip]) / (xArray[i] - xArray[i - 1]);
    u[i] = (6.0 * u[i] / (xArray[i + 1] - xArray[i - 1]) - sig * u[i - 1]) / p;
  }

  qn = un = 0.0;
  y2Array[(n - 1) * skip] = (un - qn * u[n - 2]) / (qn * y2Array[(n - 2) * skip] + 1.0);
  for (Int_t k = n - 2; k >= 0; k--) {
    y2Array[k * skip] = y2Array[k * skip] * y2Array[(k + 1) * skip] + u[k];
  }
}

/// Interpolate initialized cubic spline
///
/// \param xArray
/// \param yArray
/// \param y2Array
/// \param nxArray
/// \param nyArray
/// \param ny2Array
/// \param x
/// \param skip
/// \return
Double_t AliTPC3DCylindricalInterpolator::InterpolateCubicSpline(Double_t* xArray, Double_t* yArray, Double_t* y2Array,
                                                                 const Int_t nxArray, const Int_t nyArray,
                                                                 const Int_t ny2Array, Double_t x, Int_t skip) const
{
  Int_t klo, khi, k;
  Float_t h, b, a;
  klo = 0;
  khi = nxArray - 1;

  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if (xArray[k] > x) {
      khi = k;
    } else {
      klo = k;
    }
  }

  h = xArray[khi] - xArray[klo];

  if (TMath::Abs(h) < 1e-10) {
    return 0.0;
  }

  a = (xArray[khi] - x) / h;
  b = (x - xArray[klo]) / h;

  Double_t y = a * yArray[klo] + b * yArray[khi] +
               ((a * a * a - a) * y2Array[klo * skip] + (b * b * b - b) * y2Array[khi * skip]) * (h * h) / 6.0;

  return y;
}

/// init cubic spline for all
///
void AliTPC3DCylindricalInterpolator::InitCubicSpline()
{

  Double_t yp0, ypn1;
  if (fIsInitCubic != kTRUE) {
    fSecondDerZ = new Double_t[fNR * fNZ * fNPhi];

    // Init at Z direction
    for (Int_t m = 0; m < fNPhi; m++) {
      for (Int_t i = 0; i < fNR; i++) {
        yp0 = (-(11.0 / 6.0) * fValue[(m * (fNZ * fNR) + i * fNZ)] +
               (3.0 * fValue[(m * (fNZ * fNR) + i * fNZ) + 1]) -
               (1.5 * fValue[(m * (fNZ * fNR) + i * fNZ) + 2]) +
               ((1.0 / 3.0) * fValue[(m * (fNZ * fNR) + i * fNZ) + 4])) /
              (fZList[1] - fZList[0]);
        ypn1 = (-(11.0 / 6.0) * fValue[(m * (fNZ * fNR) + i * fNZ) + (fNZ - 1)] +
                (3.0 * fValue[(m * (fNZ * fNR) + i * fNZ) + (fNZ - 2)]) -
                (1.5 * fValue[(m * (fNZ * fNR) + i * fNZ) + (fNZ - 3)]) +
                ((1.0 / 3.0) * fValue[(m * (fNZ * fNR) + i * fNZ) + (fNZ - 4)])) /
               (fZList[0] - fZList[1]);
        InitCubicSpline(fZList, &fValue[m * (fNZ * fNR) + i * fNZ], fNZ,
                        &fSecondDerZ[m * (fNZ * fNR) + i * fNZ], 1);
      }
    }

    fIsInitCubic = kTRUE;
  }
}

/// Search the nearest grid index position to a Point
///
/// \param n
/// \param xArray
/// \param x
/// \param low
void AliTPC3DCylindricalInterpolator::Search(Int_t n, const Double_t xArray[], Double_t x, Int_t& low) const
{
  /// Search an ordered table by starting at the most recently used point

  Long_t middle, high;
  Int_t ascend = 0, increment = 1;

  if (xArray[n - 1] > xArray[0]) {
    ascend = 1; // Ascending ordered table if true
  }
  if (low < 0 || low > n - 1) {
    low = -1;
    high = n;
  } else { // Ordered Search phase
    if ((Int_t)(x > xArray[low]) == ascend) {
      if (low == n - 1) {
        return;
      }
      high = low + 1;
      while ((Int_t)(x > xArray[high]) == ascend) {
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
      while ((Int_t)(x < xArray[low]) == ascend) {
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
    if ((Int_t)(x > xArray[middle]) == ascend) {
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

/// Set the value as interpolation point
///
/// \param matricesVal TMatrixD** reference value for each point
void AliTPC3DCylindricalInterpolator::SetValue(TMatrixD** matricesVal)
{
  Int_t indexVal1D;
  Int_t index1D;
  if (!fIsAllocatingLookUp) {
    fValue = new Double_t[fNPhi * fNR * fNZ];
    fIsAllocatingLookUp = kTRUE;
  }
  for (Int_t m = 0; m < fNPhi; m++) {
    indexVal1D = m * fNR * fNZ;
    TMatrixD* mat = matricesVal[m];
    for (Int_t i = 0; i < fNR; i++) {
      index1D = indexVal1D + i * fNZ;
      for (Int_t j = 0; j < fNZ; j++) {
        fValue[index1D + j] = (*mat)(i, j);
      }
    }
  }
}

/// Set the value as interpolation point
///
/// \param matricesVal TMatrixD** reference value for each point
void AliTPC3DCylindricalInterpolator::SetValue(TMatrixD** matricesVal, Int_t iZ)
{
  Int_t indexVal1D;
  Int_t index1D;
  if (!fIsAllocatingLookUp) {
    fValue = new Double_t[fNPhi * fNR * fNZ];
    fIsAllocatingLookUp = kTRUE;
  }
  for (Int_t m = 0; m < fNPhi; m++) {
    indexVal1D = m * fNR * fNZ;
    TMatrixD* mat = matricesVal[m];
    for (Int_t i = 0; i < fNR; i++) {
      index1D = indexVal1D + i * fNZ;
      fValue[index1D + iZ] = (*mat)(i, iZ);
    }
  }
}

/// set the position of R
///
/// \param rList
void AliTPC3DCylindricalInterpolator::SetRList(Double_t* rList)
{
  fRList = new Double_t[fNR];
  for (Int_t i = 0; i < fNR; i++) {
    fRList[i] = rList[i];
  }
}

/// set the position of phi
///
/// \param phiList
void AliTPC3DCylindricalInterpolator::SetPhiList(Double_t* phiList)
{
  fPhiList = new Double_t[fNPhi];
  for (Int_t i = 0; i < fNPhi; i++) {
    fPhiList[i] = phiList[i];
  }
}

/// Setting z position
///
/// \param zList
void AliTPC3DCylindricalInterpolator::SetZList(Double_t* zList)
{
  fZList = new Double_t[fNZ];
  for (Int_t i = 0; i < fNZ; i++) {
    fZList[i] = zList[i];
  }
}

/// Setting values from 1D
///
/// \param valueList
void AliTPC3DCylindricalInterpolator::SetValue(Double_t* valueList) { fValue = valueList; }

/// Set number of total grid points
void AliTPC3DCylindricalInterpolator::SetNGridPoints()
{
  if (fNR == 0 || fNPhi == 0 || fNZ == 0) {
    Error("AliTPC3DCylindricalInterpolator::SetNGridPoints", "Error in calculating total number of grid points! Either nR, nPhi or nZ are zero!");
  }
  fNGridPoints = fNR * fNPhi * fNZ;
}