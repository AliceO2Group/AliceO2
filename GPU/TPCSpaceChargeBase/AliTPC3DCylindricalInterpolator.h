#ifndef AliTPC3DCylindricalInterpolator_H
#define AliTPC3DCylindricalInterpolator_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

/// \class AliTPC3DCylindricalInterpolator
/// \brief Interpolator for cylindrical coordinate
///        this class provides: cubic spline, quadratic and linear interpolation
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Jan 5, 2016


#include <TMatrixD.h>

class AliTPC3DCylindricalInterpolator {
public:
  AliTPC3DCylindricalInterpolator();
  virtual ~AliTPC3DCylindricalInterpolator();
  Double_t GetValue(Double_t r, Double_t phi, Double_t z);
  void InitCubicSpline();
  void SetOrder(Int_t order) { fOrder = order; }
  void SetNR(Int_t nR) { fNR = nR; }
  void SetNPhi(Int_t nPhi) { fNPhi = nPhi; }
  void SetNZ(Int_t nZ) { fNZ = nZ; }
  void SetRList(Double_t *rList);
  void SetPhiList(Double_t *phiList);
  void SetZList(Double_t *zList);
  void SetValue(Double_t *vList);
  void SetValue(TMatrixD **vList);

  Int_t GetNR() { return fNR; }
  Int_t GetNPhi() { return fNPhi; }
  Int_t GetNZ() { return fNZ; }
  Int_t GetOrder() { return fOrder; }

private:
  Int_t fOrder; ///< Order of interpolation, 1 - linear, 2 - quadratic, 3 >= - cubic,
  Int_t fNR; ///< Grid size in direction of R
  Int_t fNPhi; ///< Grid size in direction of Phi
  Int_t fNZ; ///< Grid size in direction of Z

  Double_t *fValue;  //[fNR*fNPhi*fNZ] Description 3D for storing known values interpolation should be in size fNR*fNPhi*fNZ
  Double_t *fRList; //[fNR] coordinate in R (cm) (should be increasing)
  Double_t *fPhiList; //[fNPhi] coordinate in phiList (rad) (should be increasing) 0 <= < 2 pi (cyclic)
  Double_t *fZList; //[fNZ] coordinate in z list (cm) (should be increasing)
  Double_t *fSecondDerZ; //[fNR*fNPhi*fNZ] store second derivative of cubic interpolation in z direction

  Bool_t fIsAllocatingLookUp; ///< is allocating memory
  Bool_t fIsInitCubic; ///< is cubic second derivative already been initialized

  Double_t InterpolatePhi(Double_t xArray[], const Int_t iLow, const Int_t lenX, Double_t yArray[], Double_t x);
  Double_t InterpolateCylindrical(Double_t r, Double_t z, Double_t phi);
  Double_t Interpolate(Double_t xArray[], Double_t yArray[], Double_t x);
  Double_t InterpolateCubicSpline(Double_t *xArray, Double_t *yArray, Double_t *y2Array, const Int_t nxArray,
                                  const Int_t nyArray, const Int_t ny2Array, Double_t x, const Int_t skip);
  void Search(Int_t n, const Double_t xArray[], Double_t x, Int_t &low);
  void InitCubicSpline(Double_t *xArray, Double_t *yArray, const Int_t n, Double_t *y2Array, const Int_t skip);
  void InitCubicSpline(Double_t *xArray, Double_t *yArray, const Int_t n, Double_t *y2Array, const Int_t skip,
                       Double_t yp0, Double_t ypn1);

/// \cond CLASSIMP
  ClassDef(AliTPC3DCylindricalInterpolator,1);
/// \endcond
};

#endif
