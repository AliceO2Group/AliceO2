// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCLookUpTable3DInterpolatorIrregularD.h
/// \brief Wrap up look-up table with irregular grid
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Mar 4, 2015

#ifndef AliTPCLookUpTable3DInterpolatorIrregularD_H
#define AliTPCLookUpTable3DInterpolatorIrregularD_H

#include "TMatrixD.h"
#include "AliTPC3DCylindricalInterpolatorIrregular.h"

class AliTPCLookUpTable3DInterpolatorIrregularD
{
 public:
  void SetNR(Int_t nRRow) { fNR = nRRow; }
  void SetNPhi(Int_t nPhiSlice) { fNPhi = nPhiSlice; }
  void SetNZ(Int_t nZColumn) { fNZ = nZColumn; }

  Int_t GetNR() { return fNR; }
  Int_t GetNPhi() { return fNPhi; }
  Int_t GetNZ() { return fNZ; }

  void SetRList(TMatrixD** matricesRPoint) { fMatricesRPoint = matricesRPoint; }
  void SetPhiList(TMatrixD** matricesPhiPoint) { fMatricesPhiPoint = matricesPhiPoint; }
  void SetZList(TMatrixD** matricesZPoint) { fMatricesZPoint = matricesZPoint; }

  void SetLookUpR(TMatrixD** matricesRValue) { fMatricesRValue = matricesRValue; }
  void SetLookUpPhi(TMatrixD** matricesPhiValue) { fMatricesPhiValue = matricesPhiValue; }
  void SetLookUpZ(TMatrixD** matricesZValue) { fMatricesZValue = matricesZValue; }

  AliTPCLookUpTable3DInterpolatorIrregularD();
  AliTPCLookUpTable3DInterpolatorIrregularD(Int_t nRRow, TMatrixD** matricesRValue, TMatrixD** r, Int_t nPhiSlice,
                                            TMatrixD** matricesPhiValue, TMatrixD** matricesPhiPoint, Int_t nZColumn,
                                            TMatrixD** matricesZValue, TMatrixD** matricesZPoint, Int_t order,
                                            Int_t stepR, Int_t stepZ, Int_t stepPhi, Int_t type);

  ~AliTPCLookUpTable3DInterpolatorIrregularD();

  void GetValue(Double_t r, Double_t phi, Double_t z, Double_t& rValue, Double_t& phiValue, Double_t& zValue, Int_t rIndex,
                Int_t phiIndex, Int_t zIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ);
  void GetValue(Double_t r, Double_t phi, Double_t z, Double_t& rValue, Double_t& phiValue, Double_t& zValue, Int_t rIndex,
                Int_t phiIndex, Int_t zIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ, Int_t minZColumnIndex);
  void GetValue(Double_t r, Double_t phi, Double_t z, Float_t& rValue, Float_t& phiValue, Float_t& zValue, Int_t rIndex, Int_t phiIndex,
                Int_t zIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ);
  void GetValue(Double_t r, Double_t phi, Double_t z, Double_t& rValue, Double_t& phiValue, Double_t& zValue);
  void SetOrder(Int_t order) { fOrder = order; }
  void CopyFromMatricesToInterpolator();
  void CopyFromMatricesToInterpolator(Int_t j);

  Int_t GetIrregularGridSize() { return fInterpolatorR->GetIrregularGridSize(); }
  void SetIrregularGridSize(Int_t size)
  {
    fInterpolatorR->SetIrregularGridSize(size);
    fInterpolatorPhi->SetIrregularGridSize(size);
    fInterpolatorZ->SetIrregularGridSize(size);
  }
  void SetKernelType(Int_t kernelType)
  {
    fInterpolatorR->SetKernelType(kernelType);
    fInterpolatorPhi->SetKernelType(kernelType);
    fInterpolatorZ->SetKernelType(kernelType);
  }
  Int_t GetKernelType() { return fInterpolatorR->GetKernelType(); }

 private:
  Int_t fOrder;             ///< Order of interpolation
  Int_t fIrregularGridSize; ///< Size of irregular interpolation neighborhood
  Int_t fNR;                ///< Number of grid in R
  Int_t fNPhi;              ///< Number of grid in Phi
  Int_t fNZ;                ///< Number of grid in Z

  TMatrixD** fMatricesRValue = nullptr;   //!<! Matrices to store r-component
  TMatrixD** fMatricesPhiValue = nullptr; //!<! Matrices to store phi-component
  TMatrixD** fMatricesZValue = nullptr;   //!<! Matrices to store z-component

  AliTPC3DCylindricalInterpolatorIrregular* fInterpolatorR = nullptr;   //-> Irregular interpolator for R-component
  AliTPC3DCylindricalInterpolatorIrregular* fInterpolatorPhi = nullptr; //-> Irregular interpolator for Phi-component
  AliTPC3DCylindricalInterpolatorIrregular* fInterpolatorZ = nullptr;   //-> Irregular interpolator for Z-component

  TMatrixD** fMatricesRPoint = nullptr;   //!<! Matrices to store distorted point (r component)
  TMatrixD** fMatricesPhiPoint = nullptr; //!<! Matrices to store distorted point (phi component)
  TMatrixD** fMatricesZPoint = nullptr;   //!<! Matrices to store distorted point (z component)

  Bool_t fIsAllocatingLookUp;

  /// \cond CLASSIMP
  ClassDefNV(AliTPCLookUpTable3DInterpolatorIrregularD, 1);
  /// \endcond
};

#endif
