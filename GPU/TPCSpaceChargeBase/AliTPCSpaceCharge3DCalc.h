// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCSpaceCharge3DCalc.h
/// \brief This class provides distortion and correction map calculation with integration following electron drift
/// TODO: validate distortion z by comparing with exisiting classes
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#ifndef ALI_TPC_SPACECHARGE3D_CALC_H
#define ALI_TPC_SPACECHARGE3D_CALC_H

#include "TF1.h"
#include "TH3F.h"
#include "TMatrixD.h"

#include "AliTPCPoissonSolver.h"
#include "AliTPCLookUpTable3DInterpolatorD.h"
#include "AliTPC3DCylindricalInterpolator.h"
#include "AliTPCLookUpTable3DInterpolatorIrregularD.h"
#include "AliTPC3DCylindricalInterpolatorIrregular.h"

class TFormula;

class AliTPCSpaceCharge3DCalc
{
 public:
  AliTPCSpaceCharge3DCalc();
  AliTPCSpaceCharge3DCalc(Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice);
  AliTPCSpaceCharge3DCalc(Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice,
                          Int_t interpolationOrder, Int_t irregularGridSize, Int_t rbfKernelType);
  virtual ~AliTPCSpaceCharge3DCalc();
  void InitSpaceCharge3DPoissonIntegralDz(Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                          Double_t stopConvergence);

  // outdated, to be removed after modifications in aliroot are pushed
  void InitSpaceCharge3DPoissonIntegralDz(
    Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stopConvergence,
    TMatrixD** matricesErA, TMatrixD** matricesEphiA, TMatrixD** matricesEzA,
    TMatrixD** matricesErC, TMatrixD** matricesEphiC, TMatrixD** matricesEzC,
    TMatrixD** matricesDistDrDzA, TMatrixD** matricesDistDPhiRDzA, TMatrixD** matricesDistDzA,
    TMatrixD** matricesCorrDrDzA, TMatrixD** matricesCorrDPhiRDzA, TMatrixD** matricesCorrDzA,
    TMatrixD** matricesDistDrDzC, TMatrixD** matricesDistDPhiRDzC, TMatrixD** matricesDistDzC,
    TMatrixD** matricesCorrDrDzC, TMatrixD** matricesCorrDPhiRDzC, TMatrixD** matricesCorrDzC,
    TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction);

  void InitSpaceCharge3DPoissonIntegralDz(
    Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stopConvergence,
    TMatrixD** matricesErA, TMatrixD** matricesEphiA, TMatrixD** matricesEzA,
    TMatrixD** matricesErC, TMatrixD** matricesEphiC, TMatrixD** matricesEzC,
    TMatrixD** matricesDistDrDzA, TMatrixD** matricesDistDPhiRDzA, TMatrixD** matricesDistDzA,
    TMatrixD** matricesCorrDrDzA, TMatrixD** matricesCorrDPhiRDzA, TMatrixD** matricesCorrDzA,
    TMatrixD** matricesDistDrDzC, TMatrixD** matricesDistDPhiRDzC, TMatrixD** matricesDistDzC,
    TMatrixD** matricesCorrDrDzC, TMatrixD** matricesCorrDPhiRDzC, TMatrixD** matricesCorrDzC,
    TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction, TFormula* ezF);

  void
    InitSpaceCharge3DPoisson(Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stopConvergence);
  void ForceInitSpaceCharge3DPoissonIntegralDz(Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                               Double_t stopConvergence);
  void GetDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetDistortionCylAC(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetCorrectionCyl(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetCorrectionCylAC(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetCorrectionCylACIrregular(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetCorrectionCylACIrregular(const Float_t x[], Short_t roc, Float_t dx[], const Int_t side);
  void GetDistortion(const Float_t x[], Short_t roc, Float_t dx[]);

  void GetCorrection(const Float_t x[], Short_t roc, Float_t dx[]);
  void GetCorrection(const Float_t x[], Short_t roc, Float_t dx[], const Int_t side);

  Double_t GetChargeCylAC(const Float_t x[], Short_t roc);
  Double_t GetPotentialCylAC(const Float_t x[], Short_t roc);

  Double_t GetInverseChargeCylAC(const Float_t x[], Short_t roc);

  void SetCorrectionType(Int_t correctionType) { fCorrectionType = correctionType; }

  enum {
    kNumSector = 18
  };

  enum CorrectionType {
    kRegularInterpolator = 0,   ///< use interpolation with regular interpolator for correction look up table
    kIrregularInterpolator = 1, ///< use irregular interpolator for correction look up table
  };

  enum IntegrationStrategy {
    kNaive = 0, ///< use interpolation with regular interpolator for correction look up table
    kOpt = 1,   ///< use irregular interpolator for correction look up table
  };
  void SetInputSpaceCharge(TH3* hisSpaceCharge3D, Double_t norm);
  void SetInputSpaceCharge(TH3* hisSpaceCharge3D) { SetInputSpaceCharge(hisSpaceCharge3D, 1); }
  void SetInputSpaceCharge(TH3* hisSpaceCharge3D, Double_t norm, Int_t side);
  void SetInputSpaceCharge(TH3* hisSpaceCharge3D, Int_t side) { SetInputSpaceCharge(hisSpaceCharge3D, 1, side); }

  void SetInputSpaceChargeA(TMatrixD** matricesLookUpCharge)
  {
    fInterpolatorChargeA->SetValue(matricesLookUpCharge);
    fInterpolatorChargeA->InitCubicSpline();
  }

  void SetInputSpaceChargeC(TMatrixD** matricesLookUpCharge)
  {
    fInterpolatorChargeC->SetValue(matricesLookUpCharge);
    fInterpolatorChargeC->InitCubicSpline();
  }

  void SetNRRows(Int_t nRRow) { fNRRows = nRRow; }

  void SetNPhiSlices(Int_t nPhiSlice) { fNPhiSlices = nPhiSlice; }

  void SetNZColumns(Int_t nZColumn) { fNZColumns = nZColumn; }

  Int_t GetNRRows() { return fNRRows; }

  Int_t GetNPhiSlices() { return fNPhiSlices; }

  Int_t GetNZColumns() { return fNZColumns; }

  void SetPoissonSolver(AliTPCPoissonSolver* poissonSolver)
  {
    if (fPoissonSolver != nullptr) {
      delete fPoissonSolver;
    }
    fPoissonSolver = poissonSolver;
  }

  AliTPCPoissonSolver* GetPoissonSolver() { return fPoissonSolver; }

  void SetInterpolationOrder(Int_t order);

  Int_t GetInterpolationOrder() { return fInterpolationOrder; }

  void SetOmegaTauT1T2(Float_t omegaTau, Float_t t1, Float_t t2)
  {
    const Double_t wt0 = t2 * omegaTau;
    fC0 = 1. / (1. + wt0 * wt0);
    const Double_t wt1 = t1 * omegaTau;
    fC1 = wt1 / (1. + wt1 * wt1);
  };

  void SetC0C1(Float_t c0, Float_t c1)
  {
    fC0 = c0;
    fC1 = c1;
  }

  Float_t GetC0() const { return fC0; }

  Float_t GetC1() const { return fC1; }

  void SetCorrectionFactor(Float_t correctionFactor) { fCorrectionFactor = correctionFactor; }

  Float_t GetCorrectionFactor() const { return fCorrectionFactor; }

  void InverseDistortionMaps(TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi,
                             TMatrixD** matricesEz, TMatrixD** matricesInvLocalIntErDz,
                             TMatrixD**, TMatrixD** matricesInvLocalEz,
                             TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz,
                             const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t nStep,
                             const Bool_t useCylAC, Int_t stepR, Int_t stepZ, Int_t stepPhi, Int_t interpType);

  void InverseDistortionMapsNoDrift(TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi,
                                    TMatrixD** matricesEz, TMatrixD** matricesInvLocalIntErDz,
                                    TMatrixD** matricesInvLocalIntEPhiDz, TMatrixD** matricesInvLocalEz,
                                    TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz,
                                    TMatrixD** matricesDistDz, const Int_t nRRow, const Int_t nZColumn,
                                    const Int_t phiSlice);

  void GetCorrectionCylNoDrift(const Float_t x[], const Short_t roc, Float_t dx[]);

  void GetDistortionCylNoDrift(const Float_t x[], Short_t roc, Float_t dx[]);

  void InverseGlobalToLocalDistortionNoDrift(TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz,
                                             TMatrixD** matricesDistDz, Double_t* rList, Double_t* zList,
                                             Double_t* phiList, const Int_t nRRow, const Int_t nZColumn,
                                             const Int_t phiSlice);

  void GetChargeDensity(TMatrixD** matricesChargeA, TMatrixD** matricesChargeC, TH3* spaceChargeHistogram3D,
                        const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice);

  void GetInverseLocalDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[]);

  void GetLocalDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[]);

  void SetIrregularGridSize(Int_t size) { fIrregularGridSize = size; }

  Int_t GetIrregularGridSize() { return fIrregularGridSize; }

  Int_t GetRBFKernelType() { return fRBFKernelType; }

  void SetPotentialBoundaryAndChargeFormula(TFormula* vTestFunction, TFormula* rhoTestFunction);
  TFormula* GetPotentialVFormula() const { return fFormulaPotentialV; }
  TFormula* GetChargeRhoFormula() const { return fFormulaChargeRho; }

  void SetBoundaryIFCA(TF1* f1) { fFormulaBoundaryIFCA = new TF1(*f1); }

  void SetBoundaryIFCC(TF1* f1) { fFormulaBoundaryIFCC = new TF1(*f1); }

  void SetBoundaryOFCA(TF1* f1) { fFormulaBoundaryOFCA = new TF1(*f1); }

  void SetBoundaryOFCC(TF1* f1) { fFormulaBoundaryOFCC = new TF1(*f1); }

  void SetBoundaryROCA(TF1* f1) { fFormulaBoundaryROCA = new TF1(*f1); }

  void SetBoundaryROCC(TF1* f1) { fFormulaBoundaryROCC = new TF1(*f1); }

  void SetBoundaryCE(TF1* f1) { fFormulaBoundaryCE = new TF1(*f1); }

  void SetElectricFieldFormula(TFormula* formulaEr, TFormula* formulaEPhi, TFormula* formulaEz)
  {
    fFormulaEr = formulaEr;
    fFormulaEPhi = formulaEPhi;
    fFormulaEz = formulaEz;
  }
  TFormula* GetErFormula() const { return fFormulaEr; }
  TFormula* GetEPhiFormula() const { return fFormulaEPhi; }
  TFormula* GetEzFormula() const { return fFormulaEz; }

  Float_t GetSpaceChargeDensity(Float_t r, Float_t phi, Float_t z);
  Float_t GetPotential(Float_t r, Float_t phi, Float_t z);
  void GetElectricFieldCyl(const Float_t x[], Short_t roc, Double_t dx[]);
  struct Profile {
    Double_t poissonSolverTime;
    Double_t electricFieldTime;
    Double_t localDistortionTime;
    Double_t globalDistortionTime;
    Double_t interpolationInitTime;
    Int_t iteration;
  };

  Profile GetProfile() { return myProfile; }
  void SetIntegrationStrategy(Int_t integrationStrategy) { fIntegrationStrategy = integrationStrategy; }

 private:
  static const Int_t kNMaxPhi = 360; //!<!
  Profile myProfile;                 //!<!
  Int_t fNRRows = 129;               ///< the maximum on row-slices so far ~ 2cm slicing
  Int_t fNPhiSlices = 180;           ///< the maximum of phi-slices so far = (8 per sector)
  Int_t fNZColumns = 129;            ///< the maximum on column-slices so  ~ 2cm slicing
  Float_t fC0 = 0.f;                 ///< coefficient C0 (compare Jim Thomas's notes for definitions)
  Float_t fC1 = 0.f;                 ///< coefficient C1 (compare Jim Thomas's notes for definitions)
  Float_t fCorrectionFactor = 1.f;   ///< Space Charge Correction factor in comparison to initialized

  Bool_t fInitLookUp = kFALSE; ///< flag to check if the Look Up table was created
  Double_t* fListR;                  //[fNRRows] list of r-coordinate of grids
  Double_t* fListPhi;                //[fNPhiSlices] list of \f$ \phi\f$ -coordinate of grids
  Double_t* fListZ;                  //[fNZColumns]
  Double_t* fListZA;                 //[fNZColumns]  list of z-coordinate of grids
  Double_t* fListZC;                 //[fNZColumns] list of z-coordinate of grids
  // / TODO: fListPotentialBoundary arrays are never used in the code. Remove?
  // Double_t* fListPotentialBoundaryA; //[(fNRRows + fNZColumns) * 2 * fNPhiSlices]
  // Double_t* fListPotentialBoundaryC; //[(fNRRows + fNZColumns) * 2 * fNPhiSlices]

  Int_t fCorrectionType = 1;      ///< use regular or irregular grid method
  Int_t fInterpolationOrder = 5;  ///<  Order of interpolation (1-> tri linear, 2->Lagrange interpolation order 2, 3> cubic spline)
  Int_t fIrregularGridSize = 3;   ///<  Size of irregular grid cubes for interpolation (min 3)
  Int_t fRBFKernelType = 0;       ///<  RBF kernel type
  Int_t fIntegrationStrategy = 0; ///< Strategy for integration

  TMatrixD** fMatrixIntDistDrEzA;    //!<! Matrices for storing Global distortion  \f$ R \f$ direction for Side A
  TMatrixD** fMatrixIntDistDPhiREzA; //!<! Matrices for storing Global \f$ \phi R \f$ Distortion for Side A
  TMatrixD** fMatrixIntDistDzA;      //!<! Matrices for storing Global \f$ z \f$ Distortion for Side A

  TMatrixD** fMatrixIntDistDrEzC;    //!<! Matrices for storing Global  \f$ R \f$ direction for Side C
  TMatrixD** fMatrixIntDistDPhiREzC; //!<! Matrices for storing Global \f$ \phi R \f$ Distortion for Side C
  TMatrixD** fMatrixIntDistDzC;      //!<! Matrices for storing Global \f$ z \f$ Distortion for Side C

  TMatrixD** fMatrixErOverEzA;   //!<! Matrices for storing Er Over Ez for intermediate value for side A
  TMatrixD** fMatrixEPhiOverEzA; //!<! Matrices for storing EPhi Over Ez for intermediate value for side A
  TMatrixD** fMatrixDeltaEzA;    //!<! Matrices for storing delta Ez for intermediate value for side A

  TMatrixD** fMatrixErOverEzC;   //!<! Matrices for storing Er Over Ez for intermediate value for Side C
  TMatrixD** fMatrixEPhiOverEzC; //!<! Matrices for storing EPhi Over Ez for intermediate value for Side C
  TMatrixD** fMatrixDeltaEzC;    //!<! Matrices for storing delta Ez for intermediate value for side A

  TMatrixD** fMatrixIntCorrDrEzA;    //!<! Matrices for storing Global  \f$  R \f$ correction for side A
  TMatrixD** fMatrixIntCorrDPhiREzA; //!<! Matrices for storing Global  \f$ \phi R \f$  correction for side A
  TMatrixD** fMatrixIntCorrDzA;      //!<! Matrices for storing Global  \f$ X \f$ correction for side A

  TMatrixD** fMatrixIntCorrDrEzC;    //!<!  Matrices for storing Global  \f$  R \f$ correction for side C
  TMatrixD** fMatrixIntCorrDPhiREzC; //!<! Matrices for storing Global  \f$ \phi R \f$  correction for side C
  TMatrixD** fMatrixIntCorrDzC;      //!<! Matrices for storing Global  \f$ X \f$ correction for side C

  TMatrixD** fMatrixIntCorrDrEzIrregularA;    //!<! Matrices for storing global  \f$ R \f$ correction irregular type for side A
  TMatrixD** fMatrixIntCorrDPhiREzIrregularA; //!<! Matrices for storing Global \f$ \phi R \f$ correction irregular type for side A
  TMatrixD** fMatrixIntCorrDzIrregularA;      //!<! Matrices for storing Global \f$ z \f$ correction irregular type for side A

  TMatrixD** fMatrixRListIrregularA;   //!<! Matrices for storing distorted \f$ R \f$ side A
  TMatrixD** fMatrixPhiListIrregularA; //!<! Matrices for storing distorted  \f$ \phi  \f$ side A
  TMatrixD** fMatrixZListIrregularA;   //!<! Matrices for storing distorted \f$ z \f$ side A

  TMatrixD** fMatrixIntCorrDrEzIrregularC;    //!<! Matrices for storing Global  \f$ R \f$ correction irregular type for side C
  TMatrixD** fMatrixIntCorrDPhiREzIrregularC; //!<! Matrices for storing Global \f$ \phi R \f$  correction irregular type for side C
  TMatrixD** fMatrixIntCorrDzIrregularC;      //!<! Matrices for storing Global \f$ z \f$  correction irregular type for side C

  TMatrixD** fMatrixRListIrregularC;   //!<! Matrices for storing distorted \f$ R \f$ side C
  TMatrixD** fMatrixPhiListIrregularC; //!<! Matrices for storing distorted  \f$ \phi  \f$ side C
  TMatrixD** fMatrixZListIrregularC;   //!<! Matrices for storing distorted \f$ z \f$ side C

  // look up for charge densities
  TMatrixD** fMatrixChargeA;        //!<! Matrices for storing input charge densities side A
  TMatrixD** fMatrixChargeC;        //!<! Matrices for storing input charge densities side C
  TMatrixD** fMatrixChargeInverseA; //!<! Matrices for storing charge densities from backward algorithm side A
  TMatrixD** fMatrixChargeInverseC; //!<! Matrices for storing charge densities from backward algorithm side C

  TMatrixD** fMatrixPotentialA; //!<! Matrices for storing potential side A
  TMatrixD** fMatrixPotentialC; //!<! Matrices for storing potential side C

  AliTPC3DCylindricalInterpolator* fInterpolatorChargeA = nullptr;        //-> interpolator for charge densities side A
  AliTPC3DCylindricalInterpolator* fInterpolatorChargeC = nullptr;        //-> interpolator for charge densities side C
  AliTPC3DCylindricalInterpolator* fInterpolatorPotentialA = nullptr;     //-> interpolator for charge densities side A
  AliTPC3DCylindricalInterpolator* fInterpolatorPotentialC = nullptr;     //-> interpolator for charge densities side C
  AliTPC3DCylindricalInterpolator* fInterpolatorInverseChargeA = nullptr; //-> interpolator for inverse charge densities side A
  AliTPC3DCylindricalInterpolator* fInterpolatorInverseChargeC = nullptr; //-> interpolator for inverse charge densities side C

  AliTPCLookUpTable3DInterpolatorD* fLookupIntDistA = nullptr;                   //-> look-up table for global distortion side A
  AliTPCLookUpTable3DInterpolatorD* fLookupIntCorrA = nullptr;                   //-> look-up table for global correction side A
  AliTPCLookUpTable3DInterpolatorD* fLookupIntDistC = nullptr;                   //-> look-up table for global distortion side C
  AliTPCLookUpTable3DInterpolatorD* fLookupIntCorrC = nullptr;                   //-> look-up table for global correction side C
  AliTPCLookUpTable3DInterpolatorIrregularD* fLookupIntCorrIrregularA = nullptr; //-> look-up table for global correction side A (method irregular)
  AliTPCLookUpTable3DInterpolatorIrregularD* fLookupIntCorrIrregularC = nullptr; //-> look-up table for global correction side C (method irregular)
  AliTPCLookUpTable3DInterpolatorD* fLookupIntENoDriftA = nullptr;               //-> look-up table for no drift integration side A
  AliTPCLookUpTable3DInterpolatorD* fLookupIntENoDriftC = nullptr;               //-> look-up table for no drift integration side C
  AliTPCLookUpTable3DInterpolatorD* fLookupDistA = nullptr;                      //-> look-up table for local distortion side A
  AliTPCLookUpTable3DInterpolatorD* fLookupDistC = nullptr;                      //-> look-up table for local distortion side C
  AliTPCLookUpTable3DInterpolatorD* fLookupInverseDistA = nullptr;               //-> look-up table for local distortion (from inverse) side A
  AliTPCLookUpTable3DInterpolatorD* fLookupInverseDistC = nullptr;               //-> look-up table for local distortion (from inverse) side C

  AliTPCLookUpTable3DInterpolatorD* fLookupElectricFieldA = nullptr; //-> look-up table for electric field side A
  AliTPCLookUpTable3DInterpolatorD* fLookupElectricFieldC = nullptr; //-> look-up table for electric field side C

  TH3* fHistogram3DSpaceCharge = nullptr;  //!<! Histogram with the input space charge histogram - used as an optional input
  TH3* fHistogram3DSpaceChargeA = nullptr; //!<! Histogram with the input space charge histogram - used as an optional input side A
  TH3* fHistogram3DSpaceChargeC = nullptr; //!<! Histogram with the input space charge histogram - used as an optional input side C
  TF1* fFormulaBoundaryIFCA = nullptr;     //!<! function define boundary values for IFC side A V(z) assuming symmetry in phi and r.
  TF1* fFormulaBoundaryIFCC = nullptr;     //!<! function define boundary values for IFC side C V(z) assuming symmetry in phi and r.
  TF1* fFormulaBoundaryOFCA = nullptr;     //!<! function define boundary values for OFC side A V(z) assuming symmetry in phi and r.
  TF1* fFormulaBoundaryOFCC = nullptr;     //!<! function define boundary values for IFC side C V(z) assuming symmetry in phi and r.
  TF1* fFormulaBoundaryROCA = nullptr;     //!<! function define boundary values for ROC side A V(r) assuming symmetry in phi and z.
  TF1* fFormulaBoundaryROCC = nullptr;     //!<! function define boundary values for ROC side V V(t) assuming symmetry in phi and z.
  TF1* fFormulaBoundaryCE = nullptr;       //!<! function define boundary values for CE V(z) assuming symmetry in phi and z.

  TFormula* fFormulaPotentialV = nullptr; //!<! potential V(r,rho,z) function
  TFormula* fFormulaChargeRho = nullptr;  //!<! charge density Rho(r,rho,z) function

  // analytic formula for E
  TFormula* fFormulaEPhi = nullptr; //!<! ePhi EPhi(r,rho,z) electric field (phi) function
  TFormula* fFormulaEr = nullptr;   //!<! er Er(r,rho,z) electric field (r) function
  TFormula* fFormulaEz = nullptr;   //!<! ez Ez(r,rho,z) electric field (z) function

  AliTPCPoissonSolver* fPoissonSolver = nullptr; //-> Pointer to a poisson solver

  void ElectricField(TMatrixD** matricesV, TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
                     const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlices, const Float_t gridSizeR,
                     const Float_t gridSizePhi, const Float_t gridSizeZ, const Int_t symmetry,
                     const Float_t innerRadius);

  void
    LocalDistCorrDz(TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz, TMatrixD** matricesDistDrDz,
                    TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz, TMatrixD** matricesCorrDrDz,
                    TMatrixD** matricesCorrDPhiRDz, TMatrixD** matricesCorrDz, const Int_t nRRow, const Int_t nZColumn,
                    const Int_t phiSlice, const Float_t gridSizeZ, const Double_t ezField);

  void IntegrateDistCorrDriftLineDz(AliTPCLookUpTable3DInterpolatorD* lookupLocalDist, TMatrixD** matricesGDistDrDz,
                                    TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz,
                                    AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr, TMatrixD** matricesGCorrDrDz,
                                    TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
                                    TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz,
                                    TMatrixD** matricesGCorrIrregularDz,
                                    TMatrixD** matricesRIrregular, TMatrixD** matricesPhiIrregular,
                                    TMatrixD** matricesZIrregular,
                                    const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
                                    const Double_t* rList, const Double_t* phiList, const Double_t* zList);

  // outdated, to be removed once modifications in aliroot are pushed
  void IntegrateDistCorrDriftLineDz(
    TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction,
    const Double_t ezField, TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz,
    TMatrixD** matricesGDistDz,
    TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
    TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz,
    TMatrixD** matricesGCorrIrregularDz, TMatrixD** matricesRIrregular, TMatrixD** matricesPhiIrregular,
    TMatrixD** matricesZIrregular, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
    const Double_t* rList,
    const Double_t* phiList, const Double_t* zList);

  void IntegrateDistCorrDriftLineDz(
    TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction, TFormula* ezF,
    const Double_t ezField, TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz,
    TMatrixD** matricesGDistDz,
    TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
    TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz,
    TMatrixD** matricesGCorrIrregularDz, TMatrixD** matricesRIrregular, TMatrixD** matricesPhiIrregular,
    TMatrixD** matricesZIrregular, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
    const Double_t* rList,
    const Double_t* phiList, const Double_t* zList);

  void IntegrateDistCorrDriftLineDzWithLookUp(AliTPCLookUpTable3DInterpolatorD* lookupLocalDist, TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz, AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr, TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, Double_t* rList, Double_t* phiList, Double_t* zList);

  void FillLookUpTable(AliTPCLookUpTable3DInterpolatorD* lookupGlobal, TMatrixD** lookupRDz, TMatrixD** lookupPhiRDz,
                       TMatrixD** lookupDz, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
                       const Double_t* rList, const Double_t* phiList, const Double_t* zList);

  Double_t InterpolatePhi(TH3* h3, const Double_t r, const Double_t phi, const Double_t z);

  void InverseGlobalToLocalDistortionGlobalInvTable(TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz,
                                                    TMatrixD** matricesDistDz, Double_t* rList, Double_t* zList,
                                                    Double_t* phiList, const Int_t nRRow, const Int_t nZColumn,
                                                    const Int_t phiSlice, const Int_t nStep, const Bool_t useCylAC,
                                                    Int_t stepR, Int_t stepZ, Int_t stepPhi, Int_t type);

  void InverseLocalDistortionToElectricField(TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
                                             TMatrixD** matricesInvLocalIntErDz, TMatrixD** matricesInvLocalIntEPhiDz,
                                             TMatrixD** matricesInvLocalIntEz, TMatrixD** matricesDistDrDz,
                                             TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz,
                                             Double_t* rList, Double_t* zList, Double_t* phiList, const Int_t nRRow,
                                             const Int_t nZColumn, const Int_t phiSlice);

  void InverseElectricFieldToCharge(TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi,
                                    TMatrixD** matricesEz, Double_t* rList, Double_t* zList, Double_t* phiList,
                                    const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice);

  void CalculateEField(TMatrixD** matricesV, TMatrixD** matricesErOverEz, TMatrixD** matricesEPhiOverEz,
                       TMatrixD** matricesDeltaEz, const Int_t nRRow, const Int_t nZColumn, const Int_t nPhiSlice,
                       const Int_t symmetry, Bool_t rocDisplacement = kFALSE);

  void
    IntegrateEz(TMatrixD** matricesExOverEz, TMatrixD** matricesEx, const Int_t nRRow, const Int_t nZColumn,
                const Int_t nPhiSlice, const Double_t ezField);

  void InitAllocateMemory();

  /// \cond CLASSIMP
  ClassDef(AliTPCSpaceCharge3DCalc, 1);
  /// \endcond
};

#endif
