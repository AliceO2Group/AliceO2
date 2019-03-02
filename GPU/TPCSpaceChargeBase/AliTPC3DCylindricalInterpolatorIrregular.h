#ifndef AliTPC3DCylindricalInterpolatorIrregular_H
#define AliTPC3DCylindricalInterpolatorIrregular_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

/// \class AliTPC3DInterpolatorIrregular
/// \brief Irregular grid interpolator for cylindrical coordinate with r,phi,z different coordinates
///        RBF-based interpolation
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Jan 5, 2016


#include "TMatrixD.h"


class AliTPC3DCylindricalInterpolatorIrregular {
public:
  AliTPC3DCylindricalInterpolatorIrregular(Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice, Int_t rStep, Int_t zStep,
                                           Int_t phiStep, Int_t intType);
  AliTPC3DCylindricalInterpolatorIrregular();
  virtual ~AliTPC3DCylindricalInterpolatorIrregular();

  Double_t
  GetValue(Double_t r, Double_t phi, Double_t z, Int_t rIndex, Int_t phiIndex, Int_t zIndex, Int_t stepR, Int_t stepPhi,
           Int_t stepZ);
  Double_t
  GetValue(Double_t r, Double_t phi, Double_t z, Int_t rIndex, Int_t phiIndex, Int_t zIndex, Int_t stepR, Int_t stepPhi,
           Int_t stepZ, Int_t minZColumnIndex);
  void SetOrder(Int_t order) { fOrder = order; }

  void InitRBFWeight();
  void SetIrregularGridSize(Int_t size) { fIrregularGridSize = size; }
  Int_t GetIrregularGridSize() { return fIrregularGridSize; }
  void SetKernelType(Int_t kernelType) { fKernelType = kernelType; }
  Int_t GetKernelType() { return fKernelType; }

  ///< Enumeration of Poisson Solver Strategy Type
  enum RBFKernelType {
    kRBFMultiQuadratic = 0,
    kRBFInverseMultiQuadratic = 1,
    kRBFThinPlateSpline = 2,
    kRBFGaussian = 3
  };

  void SetNR(Int_t nRRow) { fNR = nRRow; }
  void SetNPhi(Int_t nPhiSlice) { fNPhi = nPhiSlice; }
  void SetNZ(Int_t nZColumn) { fNZ = nZColumn; }

  Int_t GetNR() { return fNR; }
  Int_t GetNPhi() { return fNPhi; }
  Int_t GetNZ() { return fNZ; }

  void SetRList(Double_t *rList) { fRList = rList; }
  void SetPhiList(Double_t *phiList) { fPhiList = phiList; }
  void SetZList(Double_t *zList) { fZList = zList; }

  void SetValue(Double_t *value) { fValue = value; }
  void
  SetValue(TMatrixD **matricesValue, TMatrixD **matricesRPoint, TMatrixD **matricesPhiPoint, TMatrixD **matricesZPoint);
  void
  SetValue(TMatrixD **matricesValue, TMatrixD **matricesRPoint, TMatrixD **matricesPhiPoint, TMatrixD **matricesZPoint,
           Int_t jy);

private:
  Int_t fOrder;      ///< Order of interpolation, 1 - linear, 2 - quadratic, 3 - cubic
  Int_t fType;       ///< 0 INVERSE WEIGHT, 1 RBF FULL, 2 RBF Half
  Int_t fKernelType; ///< type kernel RBF 1--5
  Int_t fIrregularGridSize; ///< size when interpolating for irregular grid
  Int_t fNR;        ///< Grid size in direction of R
  Int_t fNPhi;      ///< Grid size in direction of Phi
  Int_t fNZ;        ///< Grid size in direction of Z
  Int_t fMinZIndex; ///<index z minimal as lower bound
  Int_t fStepR; ///< step in R direction for irregular grid
  Int_t fStepZ; ///< step in Z direction for irregular grid
  Int_t fStepPhi;  ///< step in Phi direction for irregular grid
  Int_t *fRBFWeightLookUp; ///[numofpoints] weighted look up

  Double_t fRadiusRBF0; ///< Radius RBF0
  Double_t *fValue;  ///< 3D for storing known values interpolation should be in size fNR*fNPhi*fNZ
  Double_t *fRList; ///< coordinate in R (cm) (should be increasing) in 3D
  Double_t *fPhiList; ///< coordinate in phiList (rad) (should be increasing) 0 <= < 2 pi (cyclic) in 3D
  Double_t *fZList; ///< coordinate in z list (cm) (should be increasing) in 3D
  Double_t *fRBFWeight; ///< weight for RBF
  Bool_t fIsAllocatingLookUp; ///< is allocating memory?

  Double_t Interpolate3DTableCylIDW(Double_t r, Double_t z, Double_t phi, Int_t rIndex, Int_t zIndex, Int_t phiIndex,
                                    Int_t stepR, Int_t stepZ, Int_t stepPhi);
  Double_t Interpolate3DTableCylRBF(Double_t r, Double_t z, Double_t phi, Int_t rIndex, Int_t zIndex, Int_t phiIndex,
                                    Int_t stepR, Int_t stepZ, Int_t stepPhi, Double_t radiusRBF0);

  void Search(Int_t n, const Double_t xArray[], Double_t x, Int_t &low);
  void Search(Int_t n, Double_t *xArray, Int_t offset, Double_t x, Int_t &low);
  Double_t Distance(Double_t r0, Double_t phi0, Double_t z0, Double_t r, Double_t phi, Double_t z);
  void RBFWeight(Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ, Double_t radius0,
                 Int_t kernelType, Double_t *weight);
  void
  GetRBFWeight(Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ, Double_t radius0,
               Int_t kernelType, Double_t *weight);
  void Phi(Int_t n, Double_t r[], Double_t r0, Double_t v[]);
  void rbf1(Int_t n, Double_t r[], Double_t r0, Double_t v[]);
  void rbf2(Int_t n, Double_t r[], Double_t r0, Double_t v[]);
  void rbf3(Int_t n, Double_t r[], Double_t r0, Double_t v[]);
  void rbf4(Int_t n, Double_t r[], Double_t r0, Double_t v[]);
  Double_t InterpRBF(Double_t r, Double_t phi, Double_t z, Int_t startR, Int_t startPhi, Int_t startZ, Int_t stepR,
                     Int_t stepPhi, Int_t stepZ, Double_t radius0, Int_t kernelType, Double_t *weight);
  void
  RBFWeightHalf(Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ, Double_t radius0,
                Int_t kernelType, Double_t *weight);
  Double_t InterpRBFHalf(Double_t r, Double_t phi, Double_t z, Int_t startR, Int_t startPhi, Int_t startZ, Int_t stepR,
                         Int_t stepPhi, Int_t stepZ, Double_t radius0, Int_t kernelType, Double_t *weight);
  void GetRBFWeightHalf(Int_t rIndex, Int_t zIndex, Int_t phiIndex, Int_t stepR, Int_t stepPhi, Int_t stepZ,
                        Double_t radius0, Int_t kernelType, Double_t *weight);
  Double_t GetRadius0RBF(const Int_t rIndex, const Int_t phiIndex, const Int_t zIndex);

/// \cond CLASSIMP
  ClassDef(AliTPC3DCylindricalInterpolatorIrregular,1);
/// \endcond
};

#endif
