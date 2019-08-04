// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCSpaceCharge3DCalc.cxx
/// \brief This class provides distortion and correction map with integration following electron drift
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Nov 20, 2017

#include "TStopwatch.h"
#include "TMath.h"
#include "AliTPCSpaceCharge3DCalc.h"

/// \cond CLASSIMP
ClassImp(AliTPCSpaceCharge3DCalc);
/// \endcond

/// Construction for AliTPCSpaceCharge3DCalc class
/// Default values
/// ~~~
/// fInterpolationOrder = 5; // interpolation cubic spline with 5 points
/// fNRRows = 129;
/// fNPhiSlices = 180; // the maximum of phi-slices so far = (8 per sector)
/// fNZColumns = 129; // the maximum on column-slices so  ~ 2cm slicing
/// ~~~
AliTPCSpaceCharge3DCalc::AliTPCSpaceCharge3DCalc()
{
  InitAllocateMemory();
}

/// Member values from params
///
/// \param nRRow Int_t number of grid in r direction
/// \param nZColumn Int_t number of grid in z direction
/// \param nPhiSlice Int_t number of grid in \f$ \phi \f$ direction
///
AliTPCSpaceCharge3DCalc::AliTPCSpaceCharge3DCalc(Int_t nRRow,
                                                 Int_t nZColumn, Int_t nPhiSlice)
  : fNRRows(nRRow),
    fNPhiSlices(nPhiSlice),
    fNZColumns(nZColumn)
{
  InitAllocateMemory();
}

/// Construction for AliTPCSpaceCharge3DCalc class
/// Member values from params
///
/// \param nRRow Int_t number of grid in r direction
/// \param nZColumn Int_t number of grid in z direction
/// \param nPhiSlice Int_t number of grid in \f$ \phi \f$ direction
/// \param interpolationOrder Int_t order of interpolation
/// \param strategy Int_t strategy for global distortion
/// \param rbfKernelType Int_t strategy for global distortion
///
AliTPCSpaceCharge3DCalc::AliTPCSpaceCharge3DCalc(
  Int_t nRRow, Int_t nZColumn, Int_t nPhiSlice, Int_t interpolationOrder,
  Int_t irregularGridSize, Int_t rbfKernelType)
  : fNRRows(nRRow),
    fNPhiSlices(nPhiSlice),
    fNZColumns(nZColumn),
    fInterpolationOrder(interpolationOrder),
    fIrregularGridSize(irregularGridSize),
    fRBFKernelType(rbfKernelType)
{
  InitAllocateMemory();
}

/// Memory allocation for working/output memory
///
void AliTPCSpaceCharge3DCalc::InitAllocateMemory()
{
  fPoissonSolver = new AliTPCPoissonSolver();

  fListR = new Double_t[fNRRows];
  fListPhi = new Double_t[fNPhiSlices];
  fListZ = new Double_t[fNZColumns];
  fListZA = new Double_t[fNZColumns];
  fListZC = new Double_t[fNZColumns];

  // allocate for boundary
  Int_t len = 2 * fNPhiSlices * (fNZColumns + fNRRows) - (4 * fNPhiSlices);
  fListPotentialBoundaryA = new Double_t[len];
  fListPotentialBoundaryC = new Double_t[len];

  Int_t phiSlicesPerSector = fNPhiSlices / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNRRows - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZColumns - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhiSlices;

  for (Int_t k = 0; k < fNPhiSlices; k++) {
    fListPhi[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < fNRRows; i++) {
    fListR[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < fNZColumns; j++) {
    fListZ[j] = (j * gridSizeZ);
  }

  for (Int_t j = 0; j < fNZColumns; j++) {
    fListZA[j] = (j * gridSizeZ);
  }

  for (Int_t j = 0; j < fNZColumns; j++) {
    fListZC[j] = (j * gridSizeZ);
  }

  for (Int_t k = 0; k < fNPhiSlices; k++) {

    fMatrixIntDistDrEzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntDistDPhiREzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntDistDzA[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixIntDistDrEzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntDistDPhiREzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntDistDzC[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixIntCorrDrEzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntCorrDPhiREzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntCorrDzA[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixIntCorrDrEzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntCorrDPhiREzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixIntCorrDzC[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixErOverEzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixEPhiOverEzA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixDeltaEzA[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixErOverEzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixEPhiOverEzC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixDeltaEzC[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixIntCorrDrEzIrregularA[k] = new TMatrixD(fNRRows, fNZColumns);    //[kNPhi]
    fMatrixIntCorrDPhiREzIrregularA[k] = new TMatrixD(fNRRows, fNZColumns); //[kNPhi]
    fMatrixIntCorrDzIrregularA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixRListIrregularA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixPhiListIrregularA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixZListIrregularA[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixIntCorrDrEzIrregularC[k] = new TMatrixD(fNRRows, fNZColumns);    //[kNPhi]
    fMatrixIntCorrDPhiREzIrregularC[k] = new TMatrixD(fNRRows, fNZColumns); //[kNPhi]
    fMatrixIntCorrDzIrregularC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixRListIrregularC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixPhiListIrregularC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixZListIrregularC[k] = new TMatrixD(fNRRows, fNZColumns);

    fMatrixChargeA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixChargeC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixChargeInverseA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixChargeInverseC[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixPotentialA[k] = new TMatrixD(fNRRows, fNZColumns);
    fMatrixPotentialC[k] = new TMatrixD(fNRRows, fNZColumns);
  }

  fLookupIntDistA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixIntDistDrEzA, fListR, fNPhiSlices, fMatrixIntDistDPhiREzA, fListPhi,
      fNZColumns, fMatrixIntDistDzA, fListZA, fInterpolationOrder);
  fLookupIntDistC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixIntDistDrEzC, fListR, fNPhiSlices, fMatrixIntDistDPhiREzC, fListPhi,
      fNZColumns, fMatrixIntDistDzC, fListZC, fInterpolationOrder);
  fLookupIntCorrA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixIntCorrDrEzA, fListR, fNPhiSlices, fMatrixIntCorrDPhiREzA, fListPhi,
      fNZColumns, fMatrixIntCorrDzA, fListZA, fInterpolationOrder);
  fLookupIntCorrC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixIntCorrDrEzC, fListR, fNPhiSlices, fMatrixIntCorrDPhiREzC, fListPhi,
      fNZColumns, fMatrixIntCorrDzC, fListZC, fInterpolationOrder);

  fLookupIntENoDriftA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixErOverEzA, fListR, fNPhiSlices, fMatrixEPhiOverEzA, fListPhi,
      fNZColumns, fMatrixDeltaEzA, fListZA, fInterpolationOrder);
  fLookupIntENoDriftC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, fMatrixErOverEzC, fListR, fNPhiSlices, fMatrixEPhiOverEzC, fListPhi,
      fNZColumns, fMatrixDeltaEzC, fListZC, fInterpolationOrder);
  fLookupIntCorrIrregularA =
    new AliTPCLookUpTable3DInterpolatorIrregularD(
      fNRRows, fMatrixIntCorrDrEzIrregularA, fMatrixRListIrregularA, fNPhiSlices,
      fMatrixIntCorrDPhiREzIrregularA, fMatrixPhiListIrregularA, fNZColumns,
      fMatrixIntCorrDzIrregularA, fMatrixZListIrregularA, 2, GetIrregularGridSize(),
      GetIrregularGridSize(), GetIrregularGridSize(), 1);

  fLookupIntCorrIrregularC =
    new AliTPCLookUpTable3DInterpolatorIrregularD(
      fNRRows, fMatrixIntCorrDrEzIrregularC, fMatrixRListIrregularC, fNPhiSlices,
      fMatrixIntCorrDPhiREzIrregularC, fMatrixPhiListIrregularC, fNZColumns,
      fMatrixIntCorrDzIrregularC, fMatrixZListIrregularC, 2, GetIrregularGridSize(),
      GetIrregularGridSize(), GetIrregularGridSize(), 1);

  fInterpolatorChargeA = new AliTPC3DCylindricalInterpolator();
  fInterpolatorChargeC = new AliTPC3DCylindricalInterpolator();
  fInterpolatorPotentialA = new AliTPC3DCylindricalInterpolator();
  fInterpolatorPotentialC = new AliTPC3DCylindricalInterpolator();
  fInterpolatorInverseChargeA = new AliTPC3DCylindricalInterpolator();
  fInterpolatorInverseChargeC = new AliTPC3DCylindricalInterpolator();

  fInterpolatorChargeA->SetNR(fNRRows);
  fInterpolatorChargeA->SetNZ(fNZColumns);
  fInterpolatorChargeA->SetNPhi(fNPhiSlices);
  fInterpolatorChargeA->SetRList(fListR);
  fInterpolatorChargeA->SetZList(fListZA);
  fInterpolatorChargeA->SetPhiList(fListPhi);
  fInterpolatorChargeA->SetOrder(fInterpolationOrder);

  fInterpolatorChargeC->SetNR(fNRRows);
  fInterpolatorChargeC->SetNZ(fNZColumns);
  fInterpolatorChargeC->SetNPhi(fNPhiSlices);
  fInterpolatorChargeC->SetRList(fListR);
  fInterpolatorChargeC->SetZList(fListZC);
  fInterpolatorChargeC->SetPhiList(fListPhi);
  fInterpolatorChargeC->SetOrder(fInterpolationOrder);

  fInterpolatorPotentialA->SetNR(fNRRows);
  fInterpolatorPotentialA->SetNZ(fNZColumns);
  fInterpolatorPotentialA->SetNPhi(fNPhiSlices);
  fInterpolatorPotentialA->SetRList(fListR);
  fInterpolatorPotentialA->SetZList(fListZA);
  fInterpolatorPotentialA->SetPhiList(fListPhi);
  fInterpolatorPotentialA->SetOrder(fInterpolationOrder);

  fInterpolatorPotentialC->SetNR(fNRRows);
  fInterpolatorPotentialC->SetNZ(fNZColumns);
  fInterpolatorPotentialC->SetNPhi(fNPhiSlices);
  fInterpolatorPotentialC->SetRList(fListR);
  fInterpolatorPotentialC->SetZList(fListZA);
  fInterpolatorPotentialC->SetPhiList(fListPhi);
  fInterpolatorPotentialC->SetOrder(fInterpolationOrder);

  fInterpolatorInverseChargeA->SetNR(fNRRows);
  fInterpolatorInverseChargeA->SetNZ(fNZColumns);
  fInterpolatorInverseChargeA->SetNPhi(fNPhiSlices);
  fInterpolatorInverseChargeA->SetRList(fListR);
  fInterpolatorInverseChargeA->SetZList(fListZA);
  fInterpolatorInverseChargeA->SetPhiList(fListPhi);
  fInterpolatorInverseChargeA->SetOrder(fInterpolationOrder);

  fInterpolatorInverseChargeC->SetNR(fNRRows);
  fInterpolatorInverseChargeC->SetNZ(fNZColumns);
  fInterpolatorInverseChargeC->SetNPhi(fNPhiSlices);
  fInterpolatorInverseChargeC->SetRList(fListR);
  fInterpolatorInverseChargeC->SetZList(fListZC);
  fInterpolatorInverseChargeC->SetPhiList(fListPhi);
  fInterpolatorInverseChargeC->SetOrder(fInterpolationOrder);

  fLookupDistA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupDistC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupInverseDistA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupInverseDistC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupElectricFieldA =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupElectricFieldC =
    new AliTPCLookUpTable3DInterpolatorD(
      fNRRows, nullptr, fListR, fNPhiSlices, nullptr, fListPhi, fNZColumns, nullptr, fListZA,
      fInterpolationOrder);

  fLookupIntCorrIrregularA->SetKernelType(fRBFKernelType);
  fLookupIntCorrIrregularC->SetKernelType(fRBFKernelType);

  fFormulaBoundaryIFCA = nullptr; //-> function define boundary values for IFC side A V(z) assuming symmetry in phi and r.
  fFormulaBoundaryIFCC = nullptr; //-> function define boundary values for IFC side C V(z) assuming symmetry in phi and r.
  fFormulaBoundaryOFCA = nullptr; //-> function define boundary values for OFC side A V(z) assuming symmetry in phi and r.
  fFormulaBoundaryOFCC = nullptr; ///<- function define boundary values for IFC side C V(z) assuming symmetry in phi and r.
  fFormulaBoundaryROCA = nullptr; ///<- function define boundary values for ROC side A V(r) assuming symmetry in phi and z.
  fFormulaBoundaryROCC = nullptr; ///<- function define boundary values for ROC side V V(t) assuming symmetry in phi and z.
  fFormulaBoundaryCE = nullptr;   ///<- function define boundary values for CE V(z) assuming symmetry in phi and z.

  fFormulaPotentialV = nullptr; ///<- potential V(r,rho,z) function
  fFormulaChargeRho = nullptr;  ///<- charge density Rho(r,rho,z) function

  // analytic formula for E
  fFormulaEPhi = nullptr; ///<- ePhi EPhi(r,rho,z) electric field (phi) function
  fFormulaEr = nullptr;   ///<- er Er(r,rho,z) electric field (r) function
  fFormulaEz = nullptr;   ///<- ez Ez(r,rho,z) electric field (z) function
}
/// Destruction for AliTPCSpaceCharge3DCalc
/// Deallocate memory for lookup table and charge distribution
///
AliTPCSpaceCharge3DCalc::~AliTPCSpaceCharge3DCalc()
{

  if (fPoissonSolver != nullptr) {
    delete fPoissonSolver;
  }

  for (Int_t k = 0; k < fNPhiSlices; k++) {
    delete fMatrixIntDistDrEzA[k];
    delete fMatrixIntDistDPhiREzA[k];
    delete fMatrixIntDistDzA[k];
    delete fMatrixIntDistDrEzC[k];
    delete fMatrixIntDistDPhiREzC[k];
    delete fMatrixIntDistDzC[k];
    delete fMatrixIntCorrDrEzA[k];
    delete fMatrixIntCorrDPhiREzA[k];
    delete fMatrixIntCorrDzA[k];
    delete fMatrixIntCorrDrEzC[k];
    delete fMatrixIntCorrDPhiREzC[k];
    delete fMatrixIntCorrDzC[k];
    delete fMatrixErOverEzA[k];
    delete fMatrixEPhiOverEzA[k];
    delete fMatrixDeltaEzA[k];
    delete fMatrixErOverEzC[k];
    delete fMatrixEPhiOverEzC[k];
    delete fMatrixDeltaEzC[k];
    delete fMatrixIntCorrDrEzIrregularA[k];
    delete fMatrixIntCorrDPhiREzIrregularA[k];
    delete fMatrixIntCorrDzIrregularA[k];
    delete fMatrixRListIrregularA[k];
    delete fMatrixPhiListIrregularA[k];
    delete fMatrixZListIrregularA[k];
    delete fMatrixIntCorrDrEzIrregularC[k];
    delete fMatrixIntCorrDPhiREzIrregularC[k];
    delete fMatrixIntCorrDzIrregularC[k];
    delete fMatrixRListIrregularC[k];
    delete fMatrixPhiListIrregularC[k];
    delete fMatrixZListIrregularC[k];
    delete fMatrixChargeA[k];
    delete fMatrixChargeC[k];
    delete fMatrixChargeInverseA[k];
    delete fMatrixChargeInverseC[k];

    delete fMatrixPotentialA[k];
    delete fMatrixPotentialC[k];
  }
  delete[] fListR;
  delete[] fListPhi;
  delete[] fListZ;
  delete[] fListZA;
  delete[] fListZC;

  delete fLookupIntDistA;
  delete fLookupIntDistC;
  delete fLookupIntENoDriftA;
  delete fLookupIntENoDriftC;
  delete fLookupIntCorrA;
  delete fLookupIntCorrC;
  delete fLookupIntCorrIrregularA;
  delete fLookupIntCorrIrregularC;
  delete fLookupDistA;
  delete fLookupDistC;
  delete fLookupInverseDistA;
  delete fLookupInverseDistC;
  delete fLookupElectricFieldA;
  delete fLookupElectricFieldC;
  delete fInterpolatorChargeA;
  delete fInterpolatorPotentialA;
  delete fInterpolatorChargeC;
  delete fInterpolatorPotentialC;
  delete fInterpolatorInverseChargeA;
  delete fInterpolatorInverseChargeC;

  delete[] fListPotentialBoundaryA;
  delete[] fListPotentialBoundaryC;
}

/// Creating look-up tables of Correction/Distortion by integration following
/// drift line, input from space charge 3d histogram (fSpaceCharge3D) and boundary values are filled with zeroes
///
/// TODO: provide an interface for setting boundary values
///
/// The algorithm and implementations of this function is the following:
///
/// Do for each side A,C
///
/// 1) Solving \f$ \nabla^2 \Phi(r,\phi,z) = -  \rho(r,\phi,z)\f$
/// ~~~ Calling poisson solver
/// fPoissonSolver->PoissonSolver3D( matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration, symmetry ) ;
/// ~~~
///
/// 2) Get the electric field \f$ \vec{E} = - \nabla \Phi(r,\phi,z) \f$
/// ~~~
/// ElectricField( matricesV, matricesEr,  matricesEPhi, matricesEz, nRRow, nZColumn, phiSlice,
/// gridSizeR, gridSizePhi ,gridSizeZ,symmetry, AliTPCPoissonSolver::fgkIFCRadius);
/// ~~~
///
/// 3) Calculate local distortion and correction, using Langevin formula
/// ~~~ cxx
/// LocalDistCorrDz (matricesEr, matricesEPhi,  matricesEz,
///	matricesDistDrDz,  matricesDistDPhiRDz, matricesDistDz,
///	matricesCorrDrDz,  matricesCorrDPhiRDz, matricesCorrDz,
///	nRRow,  nZColumn, phiSlice, gridSizeZ, ezField);
/// ~~~
///
/// 4) Integrate distortion by following the drift line
///
/// 5) Fill look up table for Integral distortion
///
/// 6) Fill look up table for Integral correction
///
/// \param nRRow Int_t Number of nRRow in r-direction
/// \param nZColumn Int_t Number of nZColumn in z-direction
/// \param phiSlice Int_t Number of phi slice in \f$ phi \f$ direction
/// \param maxIteration Int_t Maximum iteration for poisson solver
/// \param stoppingConvergence Convergence error stopping condition for poisson solver
///
/// \post Lookup tables for distortion:
/// ~~~
/// fLookUpIntDistDrEz,fLookUpIntDistDPhiREz,fLookUpIntDistDz
/// ~~~
/// and correction:
/// ~~~
/// fLookUpIntCorrDrEz,fLookUpIntCorrDPhiREz,fLookUpIntCorrDz
/// ~~~
/// are initialized
///
void AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz(
  Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stoppingConvergence)
{
  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  // local variables
  Float_t radius0, phi0, z0;

  // memory allocation for temporary matrices:
  // potential (boundary values), charge distribution
  TMatrixD **matricesV, *matricesCharge[phiSlice];
  TMatrixD *matricesEr[phiSlice], *matricesEPhi[phiSlice], *matricesEz[phiSlice];
  TMatrixD *matricesDistDrDz[phiSlice], *matricesDistDPhiRDz[phiSlice], *matricesDistDz[phiSlice];
  TMatrixD *matricesCorrDrDz[phiSlice], *matricesCorrDPhiRDz[phiSlice], *matricesCorrDz[phiSlice];
  TMatrixD *matricesGDistDrDz[phiSlice], *matricesGDistDPhiRDz[phiSlice], *matricesGDistDz[phiSlice];
  TMatrixD *matricesGCorrDrDz[phiSlice], *matricesGCorrDPhiRDz[phiSlice], *matricesGCorrDz[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    //matricesV[k] = new TMatrixD(nRRow, nZColumn);
    matricesCharge[k] = new TMatrixD(nRRow, nZColumn);
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDz[k] = new TMatrixD(nRRow, nZColumn);
  }

  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];

  // pointer to current TF1 for potential boundary values
  TF1* f1BoundaryIFC = nullptr;
  TF1* f1BoundaryOFC = nullptr;
  TF1* f1BoundaryROC = nullptr;
  TStopwatch w;

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = j * gridSizeZ;
  }

  // allocate look up local distortion
  AliTPCLookUpTable3DInterpolatorD* lookupLocalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesDistDrDz, rList, phiSlice, matricesDistDPhiRDz, phiList, nZColumn, matricesDistDz,
      zList, fInterpolationOrder);

  // allocate look up local correction
  AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesCorrDrDz, rList, phiSlice, matricesCorrDPhiRDz, phiList, nZColumn, matricesCorrDz,
      zList, fInterpolationOrder);

  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGDistDrDz, rList, phiSlice, matricesGDistDPhiRDz, phiList, nZColumn, matricesGDistDz,
      zList, fInterpolationOrder);
  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGCorrDrDz, rList, phiSlice, matricesGCorrDPhiRDz, phiList, nZColumn, matricesGCorrDz,
      zList, fInterpolationOrder);

  // should be set, in another place
  const Int_t symmetry = 0; // fSymmetry

  // for irregular
  TMatrixD** matricesIrregularDrDz = nullptr;
  TMatrixD** matricesIrregularDPhiRDz = nullptr;
  TMatrixD** matricesIrregularDz = nullptr;
  TMatrixD** matricesPhiIrregular = nullptr;
  TMatrixD** matricesRIrregular = nullptr;
  TMatrixD** matricesZIrregular = nullptr;

  // for charge
  TMatrixD** matricesLookUpCharge = nullptr;
  AliTPC3DCylindricalInterpolator* chargeInterpolator = nullptr;
  AliTPC3DCylindricalInterpolator* potentialInterpolator = nullptr;
  Double_t* potentialBoundary = nullptr;
  TMatrixD* matrixV;
  TMatrixD* matrixCharge;
  // for potential
  TMatrixD** matricesVPotential;

  Int_t pIndex = 0;

  // do if look up table haven't be initialized
  if (!fInitLookUp) {
    // initialize for working memory
    for (Int_t side = 0; side < 2; side++) {
      // zeroing global distortion/correction
      for (Int_t k = 0; k < phiSlice; k++) {
        matricesDistDrDz[k]->Zero();
        matricesDistDPhiRDz[k]->Zero();
        matricesDistDz[k]->Zero();
        matricesCorrDrDz[k]->Zero();
        matricesCorrDPhiRDz[k]->Zero();
        matricesCorrDz[k]->Zero();

        matricesGDistDrDz[k]->Zero();
        matricesGDistDPhiRDz[k]->Zero();
        matricesGDistDz[k]->Zero();
        matricesGCorrDrDz[k]->Zero();
        matricesGCorrDPhiRDz[k]->Zero();
        matricesGCorrDz[k]->Zero();
      }
      if (side == 0) {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularA;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularA;
        matricesIrregularDz = fMatrixIntCorrDzIrregularA;

        matricesPhiIrregular = fMatrixPhiListIrregularA;
        matricesRIrregular = fMatrixRListIrregularA;
        matricesZIrregular = fMatrixZListIrregularA;
        matricesLookUpCharge = fMatrixChargeA;

        matricesV = fMatrixPotentialA;
        chargeInterpolator = fInterpolatorChargeA;
        potentialInterpolator = fInterpolatorPotentialA;
        fLookupDistA->SetLookUpR(matricesDistDrDz);
        fLookupDistA->SetLookUpPhi(matricesDistDPhiRDz);
        fLookupDistA->SetLookUpZ(matricesDistDz);

        fLookupElectricFieldA->SetLookUpR(matricesEr);
        fLookupElectricFieldA->SetLookUpPhi(matricesEPhi);
        fLookupElectricFieldA->SetLookUpZ(matricesEz);

        potentialBoundary = fListPotentialBoundaryA;
        f1BoundaryIFC = fFormulaBoundaryIFCA;
        f1BoundaryOFC = fFormulaBoundaryOFCA;
        f1BoundaryROC = fFormulaBoundaryROCA;
      } else {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularC;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularC;
        matricesIrregularDz = fMatrixIntCorrDzIrregularC;
        matricesPhiIrregular = fMatrixPhiListIrregularC;
        matricesRIrregular = fMatrixRListIrregularC;
        matricesZIrregular = fMatrixZListIrregularC;
        matricesLookUpCharge = fMatrixChargeC;
        matricesV = fMatrixPotentialC;
        chargeInterpolator = fInterpolatorChargeC;
        potentialInterpolator = fInterpolatorPotentialC;
        fLookupDistC->SetLookUpR(matricesDistDrDz);
        fLookupDistC->SetLookUpPhi(matricesDistDPhiRDz);
        fLookupDistC->SetLookUpZ(matricesDistDz);
        fLookupElectricFieldC->SetLookUpR(matricesEr);
        fLookupElectricFieldC->SetLookUpPhi(matricesEPhi);
        fLookupElectricFieldC->SetLookUpZ(matricesEz);

        potentialBoundary = fListPotentialBoundaryC;
        f1BoundaryIFC = fFormulaBoundaryIFCC;
        f1BoundaryOFC = fFormulaBoundaryOFCC;
        f1BoundaryROC = fFormulaBoundaryROCC;
      }

      // fill the potential boundary
      // guess the initial potential
      // fill also charge
      //pIndex = 0;

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step = 0: Fill Boundary and Charge Densities"));
      for (Int_t k = 0; k < phiSlice; k++) {
        phi0 = k * gridSizePhi;
        matrixV = matricesV[k];
        matrixCharge = matricesCharge[k];
        for (Int_t i = 0; i < nRRow; i++) {
          radius0 = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
          for (Int_t j = 0; j < nZColumn; j++) {
            z0 = j * gridSizeZ;
            (*matrixCharge)(i, j) = chargeInterpolator->GetValue(rList[i], phiList[k], zList[j]);
            (*matrixV)(i, j) = 0.0; // fill zeros

            if (fFormulaPotentialV == nullptr) {
              // boundary IFC
              if (i == 0) {
                if (f1BoundaryIFC != nullptr) {
                  (*matrixV)(i, j) = f1BoundaryIFC->Eval(z0);
                }
              }
              if (i == (nRRow - 1)) {
                if (f1BoundaryOFC != nullptr) {
                  (*matrixV)(i, j) = f1BoundaryOFC->Eval(z0);
                }
              }
              if (j == 0) {
                if (fFormulaBoundaryCE) {
                  (*matrixV)(i, j) = fFormulaBoundaryCE->Eval(radius0);
                }
              }
              if (j == (nZColumn - 1)) {
                if (f1BoundaryROC != nullptr) {
                  (*matrixV)(i, j) = f1BoundaryROC->Eval(radius0);
                }
              }
            } else {
              if ((i == 0) || (i == (nRRow - 1)) || (j == 0) || (j == (nZColumn - 1))) {
                (*matrixV)(i, j) = fFormulaPotentialV->Eval(radius0, phi0, z0);
              }
            }
          }
        }
      }
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 0: Preparing Charge interpolator: %f\n", w.CpuTime()));
      AliTPCPoissonSolver::fgConvergenceError = stoppingConvergence;

      //fPoissonSolver->SetStrategy(AliTPCPoissonSolver::kMultiGrid);
      //(fPoissonSolver->fMgParameters).cycleType = AliTPCPoissonSolver::kFCycle;
      //(fPoissonSolver->fMgParameters).isFull3D = kFALSE;
      //(fPoissonSolver->fMgParameters).nMGCycle = maxIteration;
      //(fPoissonSolver->fMgParameters).maxLoop = 6;

      w.Start();
      fPoissonSolver->PoissonSolver3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration,
                                      symmetry);
      w.Stop();

      potentialInterpolator->SetValue(matricesV);
      potentialInterpolator->InitCubicSpline();

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 1: Poisson solver: %f\n", w.CpuTime()));
      if (side == 0) {
        myProfile.poissonSolverTime = w.CpuTime();
      }
      if (side == 0) {
        myProfile.iteration = fPoissonSolver->fIterations;
      }

      w.Start();
      ElectricField(matricesV,
                    matricesEr, matricesEPhi, matricesEz, nRRow, nZColumn, phiSlice,
                    gridSizeR, gridSizePhi, gridSizeZ, symmetry, AliTPCPoissonSolver::fgkIFCRadius);
      w.Stop();

      myProfile.electricFieldTime = w.CpuTime();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 2: Electric Field Calculation: %f\n", w.CpuTime()));
      w.Start();
      LocalDistCorrDz(matricesEr, matricesEPhi, matricesEz,
                      matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz,
                      matricesCorrDrDz, matricesCorrDPhiRDz, matricesCorrDz,
                      nRRow, nZColumn, phiSlice, gridSizeZ, ezField);
      w.Stop();
      myProfile.localDistortionTime = w.CpuTime();

      // copy to interpolator
      if (side == 0) {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistA->CopyFromMatricesToInterpolator();
        fLookupElectricFieldA->CopyFromMatricesToInterpolator();
      } else {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistC->CopyFromMatricesToInterpolator();
        fLookupElectricFieldC->CopyFromMatricesToInterpolator();
      }

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 3: Local distortion and correction: %f\n", w.CpuTime()));
      w.Start();
      if (fIntegrationStrategy == kNaive) {
        IntegrateDistCorrDriftLineDz(
          lookupLocalDist,
          matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
          lookupLocalCorr,
          matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
          matricesIrregularDrDz, matricesIrregularDPhiRDz, matricesIrregularDz,
          matricesRIrregular, matricesPhiIrregular, matricesZIrregular,
          nRRow, nZColumn, phiSlice, rList, phiList, zList);
      } else {
        IntegrateDistCorrDriftLineDzWithLookUp(
          lookupLocalDist,
          matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
          lookupLocalCorr,
          matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
          nRRow, nZColumn, phiSlice, rList, phiList, zList);
      }

      w.Stop();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 4: Global correction/distortion: %f\n", w.CpuTime()));
      myProfile.globalDistortionTime = w.CpuTime();

      //// copy to 1D interpolator /////
      lookupGlobalDist->CopyFromMatricesToInterpolator();
      if (fCorrectionType == 0) {
        lookupGlobalCorr->CopyFromMatricesToInterpolator();
      }
      ////

      w.Stop();

      if (side == 0) {

        w.Start();
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzA, fMatrixIntDistDPhiREzA, fMatrixIntDistDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        if (fCorrectionType == 0) {
          FillLookUpTable(lookupGlobalCorr,
                          fMatrixIntCorrDrEzA, fMatrixIntCorrDPhiREzA, fMatrixIntCorrDzA,
                          nRRow, nZColumn, phiSlice, rList, phiList, zList);
        }

        fLookupIntDistA->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrA->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularA->CopyFromMatricesToInterpolator();
        }

        w.Stop();
        myProfile.interpolationInitTime = w.CpuTime();
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 5: Filling up the look up: %f\n", w.CpuTime()));
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " A side done");
      }
      if (side == 1) {
        w.Start();
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzC, fMatrixIntDistDPhiREzC, fMatrixIntDistDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        if (fCorrectionType == 0) {
          FillLookUpTable(lookupGlobalCorr,
                          fMatrixIntCorrDrEzC, fMatrixIntCorrDPhiREzC, fMatrixIntCorrDzC,
                          nRRow, nZColumn, phiSlice, rList, phiList, zList);
        }

        fLookupIntDistC->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrC->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularC->CopyFromMatricesToInterpolator();
        }
        w.Stop();
        myProfile.interpolationInitTime = w.CpuTime();
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " C side done");
      }
    }

    fInitLookUp = kTRUE;
  }

  // memory de-allocation for temporary matrices
  for (Int_t k = 0; k < phiSlice; k++) {
    //delete matricesV[k];
    delete matricesCharge[k];
    delete matricesEr[k];
    delete matricesEPhi[k];
    delete matricesEz[k];
    delete matricesDistDrDz[k];
    delete matricesDistDPhiRDz[k];
    delete matricesDistDz[k];

    delete matricesCorrDrDz[k];
    delete matricesCorrDPhiRDz[k];
    delete matricesCorrDz[k];
    delete matricesGDistDrDz[k];
    delete matricesGDistDPhiRDz[k];
    delete matricesGDistDz[k];

    delete matricesGCorrDrDz[k];
    delete matricesGCorrDPhiRDz[k];
    delete matricesGCorrDz[k];
  }
  delete lookupLocalDist;
  delete lookupLocalCorr;
  delete lookupGlobalDist;
  delete lookupGlobalCorr;
}

// outdated, to be removed once modifications in aliroot are pushed
/// Creating look-up tables of Correction/Distortion by integration following
/// drift line with known distributions for potential and space charge.
///
///
/// \param nRRow  Int_t  number of grid in row direction
///  \param nZColumn Int_t number of grid in z direction
/// \param phiSlice   Int_t number of slices in phi direction
/// \param maxIteration Int_t max iteration for convergence
/// \param stopConvergence Double_t stopping criteria for convergence
/// \param matricesDistDrDzA TMatrixD**  local r distortion (output) A side
/// \param matricesDistDPhiRDzA TMatrixD** local r phi distortion (output) A side
/// \param matricesDistDzA TMatrixD**  local z distortion (output) A side
/// \param matricesCorrDrDzA TMatrixD** local r correction (output) A side
/// \param matricesCorrDPhiRDzA TMatrixD** local r phi correction (output) A side
/// \param matricesCorrDzA   TMatrixD** local z correction (output) A side
/// \param matricesDistDrDzC   TMatrixD**   local r distortion (output) C side
/// \param matricesDistDPhiRDzC   TMatrixD**  local r phi distortion (output) C side
/// \param matricesDistDzC TMatrixD** local z distortion (output) C side
/// \param matricesCorrDrDzC TMatrixD** local r phi correction (output) C side
/// \param matricesCorrDPhiRDzC TMatrixD** local r phi correction (output) C side
/// \param matricesCorrDzC  TMatrixD** local z correction (output) C side
///
/// \post Lookup tables for distortion:
/// ~~~
/// fLookUpIntDistDrEz,fLookUpIntDistDPhiREz,fLookUpIntDistDz
/// ~~~
/// and correction:
/// ~~~
/// fLookUpIntCorrDrEz,fLookUpIntCorrDPhiREz,fLookUpIntCorrDz
/// ~~~
/// are initialized
///
void AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz(
  Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stopConvergence,
  TMatrixD** matricesErA, TMatrixD** matricesEPhiA, TMatrixD** matricesEzA,
  TMatrixD** matricesErC, TMatrixD** matricesEPhiC, TMatrixD** matricesEzC,
  TMatrixD** matricesDistDrDzA, TMatrixD** matricesDistDPhiRDzA, TMatrixD** matricesDistDzA,
  TMatrixD** matricesCorrDrDzA, TMatrixD** matricesCorrDPhiRDzA, TMatrixD** matricesCorrDzA,
  TMatrixD** matricesDistDrDzC, TMatrixD** matricesDistDPhiRDzC, TMatrixD** matricesDistDzC,
  TMatrixD** matricesCorrDrDzC, TMatrixD** matricesCorrDPhiRDzC, TMatrixD** matricesCorrDzC,
  TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction)
{
  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  // local variables
  Float_t radius0, phi0, z0;

  // memory allocation for temporary matrices:
  // potential (boundary values), charge distribution
  TMatrixD *matricesV[phiSlice], *matricesCharge[phiSlice];
  TMatrixD *matricesEr[phiSlice], *matricesEPhi[phiSlice], *matricesEz[phiSlice];
  TMatrixD *matricesDistDrDz[phiSlice], *matricesDistDPhiRDz[phiSlice], *matricesDistDz[phiSlice];
  TMatrixD *matricesCorrDrDz[phiSlice], *matricesCorrDPhiRDz[phiSlice], *matricesCorrDz[phiSlice];
  TMatrixD *matricesGDistDrDz[phiSlice], *matricesGDistDPhiRDz[phiSlice], *matricesGDistDz[phiSlice];
  TMatrixD *matricesGCorrDrDz[phiSlice], *matricesGCorrDPhiRDz[phiSlice], *matricesGCorrDz[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    matricesV[k] = new TMatrixD(nRRow, nZColumn);
    matricesCharge[k] = new TMatrixD(nRRow, nZColumn);
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDz[k] = new TMatrixD(nRRow, nZColumn);
  }

  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];

  // pointer to current TF1 for potential boundary values
  TF1* f1BoundaryIFC = nullptr;
  TF1* f1BoundaryOFC = nullptr;
  TF1* f1BoundaryROC = nullptr;
  TStopwatch w;

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = j * gridSizeZ;
  }

  // allocate look up local distortion
  AliTPCLookUpTable3DInterpolatorD* lookupLocalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesDistDrDz, rList, phiSlice, matricesDistDPhiRDz, phiList, nZColumn, matricesDistDz,
      zList, fInterpolationOrder);

  // allocate look up local correction
  AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesCorrDrDz, rList, phiSlice, matricesCorrDPhiRDz, phiList, nZColumn, matricesCorrDz,
      zList, fInterpolationOrder);

  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGDistDrDz, rList, phiSlice, matricesGDistDPhiRDz, phiList, nZColumn, matricesGDistDz,
      zList, fInterpolationOrder);
  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGCorrDrDz, rList, phiSlice, matricesGCorrDPhiRDz, phiList, nZColumn, matricesGCorrDz,
      zList, fInterpolationOrder);

  // should be set, in another place
  const Int_t symmetry = 0; // fSymmetry

  // for irregular
  TMatrixD** matricesIrregularDrDz = nullptr;
  TMatrixD** matricesIrregularDPhiRDz = nullptr;
  TMatrixD** matricesIrregularDz = nullptr;
  TMatrixD** matricesPhiIrregular = nullptr;
  TMatrixD** matricesRIrregular = nullptr;
  TMatrixD** matricesZIrregular = nullptr;

  // for charge
  TMatrixD** matricesLookUpCharge = nullptr;
  AliTPC3DCylindricalInterpolator* chargeInterpolator = nullptr;
  AliTPC3DCylindricalInterpolator* potentialInterpolator = nullptr;
  Double_t* potentialBoundary = nullptr;
  TMatrixD* matrixV;
  TMatrixD* matrixCharge;
  Int_t pIndex = 0;

  // do if look up table haven't be initialized
  if (!fInitLookUp) {
    // initialize for working memory
    for (Int_t side = 0; side < 2; side++) {
      // zeroing global distortion/correction
      for (Int_t k = 0; k < phiSlice; k++) {
        matricesDistDrDz[k]->Zero();
        matricesDistDPhiRDz[k]->Zero();
        matricesDistDz[k]->Zero();
        matricesCorrDrDz[k]->Zero();
        matricesCorrDPhiRDz[k]->Zero();
        matricesCorrDz[k]->Zero();

        matricesGDistDrDz[k]->Zero();
        matricesGDistDPhiRDz[k]->Zero();
        matricesGDistDz[k]->Zero();
        matricesGCorrDrDz[k]->Zero();
        matricesGCorrDPhiRDz[k]->Zero();
        matricesGCorrDz[k]->Zero();
      }
      if (side == 0) {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularA;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularA;
        matricesIrregularDz = fMatrixIntCorrDzIrregularA;

        matricesPhiIrregular = fMatrixPhiListIrregularA;
        matricesRIrregular = fMatrixRListIrregularA;
        matricesZIrregular = fMatrixZListIrregularA;
        matricesLookUpCharge = fMatrixChargeA;
        chargeInterpolator = fInterpolatorChargeA;
        potentialInterpolator = fInterpolatorPotentialA;
        fLookupDistA->SetLookUpR(matricesDistDrDzA);
        fLookupDistA->SetLookUpPhi(matricesDistDPhiRDzA);
        fLookupDistA->SetLookUpZ(matricesDistDzA);
        lookupLocalDist->SetLookUpR(matricesDistDrDzA);
        lookupLocalDist->SetLookUpPhi(matricesDistDPhiRDzA);
        lookupLocalDist->SetLookUpZ(matricesDistDzA);

        lookupLocalCorr->SetLookUpR(matricesCorrDrDzA);
        lookupLocalCorr->SetLookUpPhi(matricesCorrDPhiRDzA);
        lookupLocalCorr->SetLookUpZ(matricesCorrDzA);

        fLookupElectricFieldA->SetLookUpR(matricesErA);
        fLookupElectricFieldA->SetLookUpPhi(matricesEPhiA);
        fLookupElectricFieldA->SetLookUpZ(matricesEzA);

        potentialBoundary = fListPotentialBoundaryA;
        f1BoundaryIFC = fFormulaBoundaryIFCA;
        f1BoundaryOFC = fFormulaBoundaryOFCA;
        f1BoundaryROC = fFormulaBoundaryROCA;
      } else {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularC;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularC;
        matricesIrregularDz = fMatrixIntCorrDzIrregularC;
        matricesPhiIrregular = fMatrixPhiListIrregularC;
        matricesRIrregular = fMatrixRListIrregularC;
        matricesZIrregular = fMatrixZListIrregularC;
        matricesLookUpCharge = fMatrixChargeC;
        chargeInterpolator = fInterpolatorChargeC;
        potentialInterpolator = fInterpolatorPotentialC;
        fLookupDistC->SetLookUpR(matricesDistDrDzC);
        fLookupDistC->SetLookUpPhi(matricesDistDPhiRDzC);
        fLookupDistC->SetLookUpZ(matricesDistDzC);
        fLookupElectricFieldC->SetLookUpR(matricesErC);
        fLookupElectricFieldC->SetLookUpPhi(matricesEPhiC);
        fLookupElectricFieldC->SetLookUpZ(matricesEzC);

        lookupLocalDist->SetLookUpR(matricesDistDrDzC);
        lookupLocalDist->SetLookUpPhi(matricesDistDPhiRDzC);
        lookupLocalDist->SetLookUpZ(matricesDistDzC);

        lookupLocalCorr->SetLookUpR(matricesCorrDrDzC);
        lookupLocalCorr->SetLookUpPhi(matricesCorrDPhiRDzC);
        lookupLocalCorr->SetLookUpZ(matricesCorrDzC);

        potentialBoundary = fListPotentialBoundaryC;
        f1BoundaryIFC = fFormulaBoundaryIFCC;
        f1BoundaryOFC = fFormulaBoundaryOFCC;
        f1BoundaryROC = fFormulaBoundaryROCC;
      }

      // fill the potential boundary
      // guess the initial potential
      // fill also charge
      //pIndex = 0;

      //Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz","Step = 0: Fill Boundary and Charge Densities");
      for (Int_t k = 0; k < phiSlice; k++) {
        phi0 = k * gridSizePhi;
        matrixV = matricesV[k];
        matrixCharge = matricesCharge[k];
        for (Int_t i = 0; i < nRRow; i++) {
          radius0 = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
          for (Int_t j = 0; j < nZColumn; j++) {
            z0 = j * gridSizeZ;
            (*matrixCharge)(i, j) = chargeInterpolator->GetValue(rList[i], phiList[k], zList[j]);
            (*matrixV)(i, j) = fFormulaPotentialV->Eval(radius0, phi0, z0);
          }
        }
      }
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 0: Preparing Charge interpolator: %f\n", w.CpuTime()));
      //AliTPCPoissonSolver::fgConvergenceError = stoppingConvergence;

      //fPoissonSolver->SetStrategy(AliTPCPoissonSolver::kMultiGrid);
      //(fPoissonSolver->fMgParameters).cycleType = AliTPCPoissonSolver::kFCycle;
      //(fPoissonSolver->fMgParameters).isFull3D = kFALSE;
      //(fPoissonSolver->fMgParameters).nMGCycle = maxIteration;
      //(fPoissonSolver->fMgParameters).maxLoop = 6;

      w.Start();
      //fPoissonSolver->PoissonSolver3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration,
      //                                symmetry);
      w.Stop();

      potentialInterpolator->SetValue(matricesV);
      potentialInterpolator->InitCubicSpline();

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 1: Poisson solver: %f\n", w.CpuTime()));
      w.Start();
      //ElectricField(matricesV,
      //              matricesEr, matricesEPhi, matricesEz, nRRow, nZColumn, phiSlice,
      //              gridSizeR, gridSizePhi, gridSizeZ, symmetry, AliTPCPoissonSolver::fgkIFCRadius);
      w.Stop();

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 2: Electric Field Calculation: %f\n", w.CpuTime()));
      w.Start();
      //LocalDistCorrDz(matricesEr, matricesEPhi, matricesEz,
      //                      matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz,
      //                      matricesCorrDrDz, matricesCorrDPhiRDz, matricesCorrDz,
      //                      nRRow, nZColumn, phiSlice, gridSizeZ, ezField);
      w.Stop();

      // copy to interpolator
      if (side == 0) {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistA->CopyFromMatricesToInterpolator();
        fLookupElectricFieldA->CopyFromMatricesToInterpolator();
      } else {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistC->CopyFromMatricesToInterpolator();
        fLookupElectricFieldC->CopyFromMatricesToInterpolator();
      }

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 3: Local distortion and correction: %f\n", w.CpuTime()));
      w.Start();

      IntegrateDistCorrDriftLineDz(intErDzTestFunction, intEPhiRDzTestFunction, intDzTestFunction, ezField,
                                   matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
                                   matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
                                   matricesIrregularDrDz, matricesIrregularDPhiRDz, matricesIrregularDz,
                                   matricesRIrregular, matricesPhiIrregular, matricesZIrregular,
                                   nRRow, nZColumn, phiSlice, rList, phiList, zList);

      w.Stop();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "Step 4: Global correction/distortion: %f\n", w.CpuTime());
      w.Start();

      //// copy to 1D interpolator /////
      lookupGlobalDist->CopyFromMatricesToInterpolator();
      lookupGlobalCorr->CopyFromMatricesToInterpolator();
      ////

      w.Stop();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "Step 5: Filling up the look up: %f\n", w.CpuTime());

      if (side == 0) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzA, fMatrixIntDistDPhiREzA, fMatrixIntDistDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzA, fMatrixIntCorrDPhiREzA, fMatrixIntCorrDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistA->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrA->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularA->CopyFromMatricesToInterpolator();
        }

        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " A side done");
      }
      if (side == 1) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzC, fMatrixIntDistDPhiREzC, fMatrixIntDistDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzC, fMatrixIntCorrDPhiREzC, fMatrixIntCorrDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistC->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrC->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularC->CopyFromMatricesToInterpolator();
        }
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " C side done");
      }
    }

    fInitLookUp = kTRUE;
  }

  // memory de-allocation for temporary matrices
  for (Int_t k = 0; k < phiSlice; k++) {
    delete matricesV[k];
    delete matricesCharge[k];
    delete matricesEr[k];
    delete matricesEPhi[k];
    delete matricesEz[k];
    delete matricesDistDrDz[k];
    delete matricesDistDPhiRDz[k];
    delete matricesDistDz[k];

    delete matricesCorrDrDz[k];
    delete matricesCorrDPhiRDz[k];
    delete matricesCorrDz[k];
    delete matricesGDistDrDz[k];
    delete matricesGDistDPhiRDz[k];
    delete matricesGDistDz[k];

    delete matricesGCorrDrDz[k];
    delete matricesGCorrDPhiRDz[k];
    delete matricesGCorrDz[k];
  }
  delete lookupLocalDist;
  delete lookupLocalCorr;
  delete lookupGlobalDist;
  delete lookupGlobalCorr;
}

/// Creating look-up tables of Correction/Distortion by integration following
/// drift line with known distributions for potential and space charge.
///
///
/// \param nRRow	Int_t  number of grid in row direction
///	\param nZColumn Int_t number of grid in z direction
/// \param phiSlice     Int_t number of slices in phi direction
/// \param maxIteration Int_t max iteration for convergence
/// \param stopConvergence Double_t stopping criteria for convergence
/// \param matricesDistDrDzA TMatrixD**  local r distortion (output) A side
/// \param matricesDistDPhiRDzA TMatrixD** local r phi distortion (output) A side
/// \param matricesDistDzA TMatrixD**  local z distortion (output) A side
/// \param matricesCorrDrDzA TMatrixD** local r correction (output) A side
/// \param matricesCorrDPhiRDzA TMatrixD** local r phi correction (output) A side
/// \param matricesCorrDzA  TMatrixD** local z correction (output) A side
/// \param matricesDistDrDzC    TMatrixD**   local r distortion (output) C side
/// \param matricesDistDPhiRDzC     TMatrixD**  local r phi distortion (output) C side
/// \param matricesDistDzC TMatrixD** local z distortion (output) C side
/// \param matricesCorrDrDzC TMatrixD** local r phi correction (output) C side
/// \param matricesCorrDPhiRDzC TMatrixD** local r phi correction (output) C side
/// \param matricesCorrDzC	TMatrixD** local z correction (output) C side
///
/// \post Lookup tables for distortion:
/// ~~~
/// fLookUpIntDistDrEz,fLookUpIntDistDPhiREz,fLookUpIntDistDz
/// ~~~
/// and correction:
/// ~~~
/// fLookUpIntCorrDrEz,fLookUpIntCorrDPhiREz,fLookUpIntCorrDz
/// ~~~
/// are initialized
///
void AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz(
  Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration, Double_t stopConvergence,
  TMatrixD** matricesErA, TMatrixD** matricesEPhiA, TMatrixD** matricesEzA,
  TMatrixD** matricesErC, TMatrixD** matricesEPhiC, TMatrixD** matricesEzC,
  TMatrixD** matricesDistDrDzA, TMatrixD** matricesDistDPhiRDzA, TMatrixD** matricesDistDzA,
  TMatrixD** matricesCorrDrDzA, TMatrixD** matricesCorrDPhiRDzA, TMatrixD** matricesCorrDzA,
  TMatrixD** matricesDistDrDzC, TMatrixD** matricesDistDPhiRDzC, TMatrixD** matricesDistDzC,
  TMatrixD** matricesCorrDrDzC, TMatrixD** matricesCorrDPhiRDzC, TMatrixD** matricesCorrDzC,
  TFormula* intErDzTestFunction, TFormula* intEPhiRDzTestFunction, TFormula* intDzTestFunction, TFormula* ezFunction)
{
  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  // local variables
  Float_t radius0, phi0, z0;

  // memory allocation for temporary matrices:
  // potential (boundary values), charge distribution
  TMatrixD *matricesV[phiSlice], *matricesCharge[phiSlice];
  TMatrixD *matricesEr[phiSlice], *matricesEPhi[phiSlice], *matricesEz[phiSlice];
  TMatrixD *matricesDistDrDz[phiSlice], *matricesDistDPhiRDz[phiSlice], *matricesDistDz[phiSlice];
  TMatrixD *matricesCorrDrDz[phiSlice], *matricesCorrDPhiRDz[phiSlice], *matricesCorrDz[phiSlice];
  TMatrixD *matricesGDistDrDz[phiSlice], *matricesGDistDPhiRDz[phiSlice], *matricesGDistDz[phiSlice];
  TMatrixD *matricesGCorrDrDz[phiSlice], *matricesGCorrDPhiRDz[phiSlice], *matricesGCorrDz[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    matricesV[k] = new TMatrixD(nRRow, nZColumn);
    matricesCharge[k] = new TMatrixD(nRRow, nZColumn);
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGDistDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDrDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDPhiRDz[k] = new TMatrixD(nRRow, nZColumn);
    matricesGCorrDz[k] = new TMatrixD(nRRow, nZColumn);
  }

  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];

  // pointer to current TF1 for potential boundary values
  TF1* f1BoundaryIFC = nullptr;
  TF1* f1BoundaryOFC = nullptr;
  TF1* f1BoundaryROC = nullptr;
  TStopwatch w;

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = j * gridSizeZ;
  }

  // allocate look up local distortion
  AliTPCLookUpTable3DInterpolatorD* lookupLocalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesDistDrDz, rList, phiSlice, matricesDistDPhiRDz, phiList, nZColumn, matricesDistDz,
      zList, fInterpolationOrder);

  // allocate look up local correction
  AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesCorrDrDz, rList, phiSlice, matricesCorrDPhiRDz, phiList, nZColumn, matricesCorrDz,
      zList, fInterpolationOrder);

  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalDist =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGDistDrDz, rList, phiSlice, matricesGDistDPhiRDz, phiList, nZColumn, matricesGDistDz,
      zList, fInterpolationOrder);
  // allocate look up for global distortion
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalCorr =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGCorrDrDz, rList, phiSlice, matricesGCorrDPhiRDz, phiList, nZColumn, matricesGCorrDz,
      zList, fInterpolationOrder);

  // should be set, in another place
  const Int_t symmetry = 0; // fSymmetry

  // for irregular
  TMatrixD** matricesIrregularDrDz = nullptr;
  TMatrixD** matricesIrregularDPhiRDz = nullptr;
  TMatrixD** matricesIrregularDz = nullptr;
  TMatrixD** matricesPhiIrregular = nullptr;
  TMatrixD** matricesRIrregular = nullptr;
  TMatrixD** matricesZIrregular = nullptr;

  // for charge
  TMatrixD** matricesLookUpCharge = nullptr;
  AliTPC3DCylindricalInterpolator* chargeInterpolator = nullptr;
  AliTPC3DCylindricalInterpolator* potentialInterpolator = nullptr;
  Double_t* potentialBoundary = nullptr;
  TMatrixD* matrixV;
  TMatrixD* matrixCharge;
  Int_t pIndex = 0;

  // do if look up table haven't be initialized
  if (!fInitLookUp) {
    // initialize for working memory
    for (Int_t side = 0; side < 2; side++) {
      // zeroing global distortion/correction
      for (Int_t k = 0; k < phiSlice; k++) {
        matricesDistDrDz[k]->Zero();
        matricesDistDPhiRDz[k]->Zero();
        matricesDistDz[k]->Zero();
        matricesCorrDrDz[k]->Zero();
        matricesCorrDPhiRDz[k]->Zero();
        matricesCorrDz[k]->Zero();

        matricesGDistDrDz[k]->Zero();
        matricesGDistDPhiRDz[k]->Zero();
        matricesGDistDz[k]->Zero();
        matricesGCorrDrDz[k]->Zero();
        matricesGCorrDPhiRDz[k]->Zero();
        matricesGCorrDz[k]->Zero();
      }
      if (side == 0) {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularA;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularA;
        matricesIrregularDz = fMatrixIntCorrDzIrregularA;

        matricesPhiIrregular = fMatrixPhiListIrregularA;
        matricesRIrregular = fMatrixRListIrregularA;
        matricesZIrregular = fMatrixZListIrregularA;
        matricesLookUpCharge = fMatrixChargeA;
        chargeInterpolator = fInterpolatorChargeA;
        potentialInterpolator = fInterpolatorPotentialA;
        fLookupDistA->SetLookUpR(matricesDistDrDzA);
        fLookupDistA->SetLookUpPhi(matricesDistDPhiRDzA);
        fLookupDistA->SetLookUpZ(matricesDistDzA);
        lookupLocalDist->SetLookUpR(matricesDistDrDzA);
        lookupLocalDist->SetLookUpPhi(matricesDistDPhiRDzA);
        lookupLocalDist->SetLookUpZ(matricesDistDzA);

        lookupLocalCorr->SetLookUpR(matricesCorrDrDzA);
        lookupLocalCorr->SetLookUpPhi(matricesCorrDPhiRDzA);
        lookupLocalCorr->SetLookUpZ(matricesCorrDzA);

        fLookupElectricFieldA->SetLookUpR(matricesErA);
        fLookupElectricFieldA->SetLookUpPhi(matricesEPhiA);
        fLookupElectricFieldA->SetLookUpZ(matricesEzA);

        potentialBoundary = fListPotentialBoundaryA;
        f1BoundaryIFC = fFormulaBoundaryIFCA;
        f1BoundaryOFC = fFormulaBoundaryOFCA;
        f1BoundaryROC = fFormulaBoundaryROCA;
      } else {
        matricesIrregularDrDz = fMatrixIntCorrDrEzIrregularC;
        matricesIrregularDPhiRDz = fMatrixIntCorrDPhiREzIrregularC;
        matricesIrregularDz = fMatrixIntCorrDzIrregularC;
        matricesPhiIrregular = fMatrixPhiListIrregularC;
        matricesRIrregular = fMatrixRListIrregularC;
        matricesZIrregular = fMatrixZListIrregularC;
        matricesLookUpCharge = fMatrixChargeC;
        chargeInterpolator = fInterpolatorChargeC;
        potentialInterpolator = fInterpolatorPotentialC;
        fLookupDistC->SetLookUpR(matricesDistDrDzC);
        fLookupDistC->SetLookUpPhi(matricesDistDPhiRDzC);
        fLookupDistC->SetLookUpZ(matricesDistDzC);
        fLookupElectricFieldC->SetLookUpR(matricesErC);
        fLookupElectricFieldC->SetLookUpPhi(matricesEPhiC);
        fLookupElectricFieldC->SetLookUpZ(matricesEzC);

        lookupLocalDist->SetLookUpR(matricesDistDrDzC);
        lookupLocalDist->SetLookUpPhi(matricesDistDPhiRDzC);
        lookupLocalDist->SetLookUpZ(matricesDistDzC);

        lookupLocalCorr->SetLookUpR(matricesCorrDrDzC);
        lookupLocalCorr->SetLookUpPhi(matricesCorrDPhiRDzC);
        lookupLocalCorr->SetLookUpZ(matricesCorrDzC);

        potentialBoundary = fListPotentialBoundaryC;
        f1BoundaryIFC = fFormulaBoundaryIFCC;
        f1BoundaryOFC = fFormulaBoundaryOFCC;
        f1BoundaryROC = fFormulaBoundaryROCC;
      }

      // fill the potential boundary
      // guess the initial potential
      // fill also charge
      //pIndex = 0;

      //Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz","%s",Form("Step = 0: Fill Boundary and Charge Densities"));
      for (Int_t k = 0; k < phiSlice; k++) {
        phi0 = k * gridSizePhi;
        matrixV = matricesV[k];
        matrixCharge = matricesCharge[k];
        for (Int_t i = 0; i < nRRow; i++) {
          radius0 = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
          for (Int_t j = 0; j < nZColumn; j++) {
            z0 = j * gridSizeZ;
            (*matrixCharge)(i, j) = chargeInterpolator->GetValue(rList[i], phiList[k], zList[j]);
            (*matrixV)(i, j) = fFormulaPotentialV->Eval(radius0, phi0, z0);
          }
        }
      }
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 0: Preparing Charge interpolator: %f\n", w.CpuTime()));
      //AliTPCPoissonSolver::fgConvergenceError = stoppingConvergence;

      //fPoissonSolver->SetStrategy(AliTPCPoissonSolver::kMultiGrid);
      //(fPoissonSolver->fMgParameters).cycleType = AliTPCPoissonSolver::kFCycle;
      //(fPoissonSolver->fMgParameters).isFull3D = kFALSE;
      //(fPoissonSolver->fMgParameters).nMGCycle = maxIteration;
      //(fPoissonSolver->fMgParameters).maxLoop = 6;

      w.Start();
      //fPoissonSolver->PoissonSolver3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration,
      //                                symmetry);
      w.Stop();

      potentialInterpolator->SetValue(matricesV);
      potentialInterpolator->InitCubicSpline();

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 1: Poisson solver: %f\n", w.CpuTime()));
      w.Start();
      //ElectricField(matricesV,
      //              matricesEr, matricesEPhi, matricesEz, nRRow, nZColumn, phiSlice,
      //              gridSizeR, gridSizePhi, gridSizeZ, symmetry, AliTPCPoissonSolver::fgkIFCRadius);
      w.Stop();

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 2: Electric Field Calculation: %f\n", w.CpuTime()));
      w.Start();
      //LocalDistCorrDz(matricesEr, matricesEPhi, matricesEz,
      //                      matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz,
      //                      matricesCorrDrDz, matricesCorrDPhiRDz, matricesCorrDz,
      //                      nRRow, nZColumn, phiSlice, gridSizeZ, ezField);
      w.Stop();

      // copy to interpolator
      if (side == 0) {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistA->CopyFromMatricesToInterpolator();
        fLookupElectricFieldA->CopyFromMatricesToInterpolator();
      } else {
        lookupLocalDist->CopyFromMatricesToInterpolator();
        lookupLocalCorr->CopyFromMatricesToInterpolator();
        fLookupDistC->CopyFromMatricesToInterpolator();
        fLookupElectricFieldC->CopyFromMatricesToInterpolator();
      }

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 3: Local distortion and correction: %f\n", w.CpuTime()));
      w.Start();

      IntegrateDistCorrDriftLineDz(intErDzTestFunction, intEPhiRDzTestFunction, intDzTestFunction, ezFunction, ezField,
                                   matricesGDistDrDz, matricesGDistDPhiRDz, matricesGDistDz,
                                   matricesGCorrDrDz, matricesGCorrDPhiRDz, matricesGCorrDz,
                                   matricesIrregularDrDz, matricesIrregularDPhiRDz, matricesIrregularDz,
                                   matricesRIrregular, matricesPhiIrregular, matricesZIrregular,
                                   nRRow, nZColumn, phiSlice, rList, phiList, zList);

      w.Stop();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 4: Global correction/distortion: %f\n", w.CpuTime()));
      w.Start();

      //// copy to 1D interpolator /////
      lookupGlobalDist->CopyFromMatricesToInterpolator();
      lookupGlobalCorr->CopyFromMatricesToInterpolator();
      ////

      w.Stop();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Step 5: Filling up the look up: %f\n", w.CpuTime()));

      if (side == 0) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzA, fMatrixIntDistDPhiREzA, fMatrixIntDistDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzA, fMatrixIntCorrDPhiREzA, fMatrixIntCorrDzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistA->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrA->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularA->CopyFromMatricesToInterpolator();
        }

        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " A side done");
      }
      if (side == 1) {
        FillLookUpTable(lookupGlobalDist,
                        fMatrixIntDistDrEzC, fMatrixIntDistDPhiREzC, fMatrixIntDistDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        FillLookUpTable(lookupGlobalCorr,
                        fMatrixIntCorrDrEzC, fMatrixIntCorrDPhiREzC, fMatrixIntCorrDzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);

        fLookupIntDistC->CopyFromMatricesToInterpolator();
        if (fCorrectionType == 0) {
          fLookupIntCorrC->CopyFromMatricesToInterpolator();
        } else {
          fLookupIntCorrIrregularC->CopyFromMatricesToInterpolator();
        }
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " C side done");
      }
    }

    fInitLookUp = kTRUE;
  }

  // memory de-allocation for temporary matrices
  for (Int_t k = 0; k < phiSlice; k++) {
    delete matricesV[k];
    delete matricesCharge[k];
    delete matricesEr[k];
    delete matricesEPhi[k];
    delete matricesEz[k];
    delete matricesDistDrDz[k];
    delete matricesDistDPhiRDz[k];
    delete matricesDistDz[k];

    delete matricesCorrDrDz[k];
    delete matricesCorrDPhiRDz[k];
    delete matricesCorrDz[k];
    delete matricesGDistDrDz[k];
    delete matricesGDistDPhiRDz[k];
    delete matricesGDistDz[k];

    delete matricesGCorrDrDz[k];
    delete matricesGCorrDPhiRDz[k];
    delete matricesGCorrDz[k];
  }
  delete lookupLocalDist;
  delete lookupLocalCorr;
  delete lookupGlobalDist;
  delete lookupGlobalCorr;
}
/// Creating look-up tables of Correction/Distortion by linear integration
/// on z line
///
/// \param nRRow	Int_t  number of grid in row direction
///	\param nZColumn Int_t number of grid in z direction
/// \param phiSlice     Int_t number of slices in phi direction
/// \param maxIteration Int_t max iteration for convergence
/// \param stoppingConvergence Double_t stopping criteria for convergence
/// \post Lookup tables for distortion:
/// ~~~
/// fLookUpIntDistDrEz,fLookUpIntDistDPhiREz,fLookUpIntDistDz
/// ~~~ fo
/// and correction:
/// ~~~
/// fLookUpIntCorrDrEz,fLookUpIntCorrDPhiREz,fLookUpIntCorrDz
/// ~~~
/// are initialized
///
void AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoisson(Int_t nRRow, Int_t nZColumn, Int_t phiSlice, Int_t maxIteration,
                                                       Double_t stoppingConvergence)
{
  // Compute grid size for all direction
  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  // local variables
  Float_t radius0, phi0, z0;

  // memory allocation for temporary matrices:
  // potential (boundary values), charge distribution

  TMatrixD *matricesV[phiSlice], *matricesCharge[phiSlice];
  TMatrixD *matricesEr[phiSlice], *matricesEPhi[phiSlice], *matricesEz[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesV[k] = new TMatrixD(nRRow, nZColumn);
    matricesCharge[k] = new TMatrixD(nRRow, nZColumn);
  }

  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = j * gridSizeZ;
  }
  // should be set, in another place
  const Int_t symmetry = 0;
  // do if look up table haven't be initialized
  if (!fInitLookUp) {
    for (Int_t side = 0; side < 2; side++) {
      for (Int_t k = 0; k < phiSlice; k++) {
        TMatrixD* mV = matricesV[k];
        TMatrixD* mCharge = matricesCharge[k];
        phi0 = phiList[k];
        for (Int_t i = 0; i < nRRow; i++) {
          radius0 = rList[i];
          for (Int_t j = 0; j < nZColumn; j++) {
            z0 = zList[j];
            if (side == 1) {
              z0 = -TMath::Abs(zList[j]);
            }
            if (fHistogram3DSpaceCharge != nullptr) {
              // * Boundary values and charge distribution setup
              (*mV)(i, j) = 0.0;
              (*mCharge)(i, j) = -1 * InterpolatePhi(fHistogram3DSpaceCharge, phi0, radius0, z0);
            }
          }
        }
      }
      AliTPCLookUpTable3DInterpolatorD* lookupEField =
        new AliTPCLookUpTable3DInterpolatorD(
          nRRow,
          matricesEr,
          rList, phiSlice,
          matricesEPhi,
          phiList, nZColumn,
          matricesEz,
          zList,
          fInterpolationOrder);

      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "Step 1: Solving poisson solver");
      fPoissonSolver->PoissonSolver3D(matricesV, matricesCharge, nRRow, nZColumn, phiSlice, maxIteration,
                                      symmetry);
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "Step 2: Calculate electric field");
      CalculateEField(
        matricesV,
        matricesEr,
        matricesEPhi,
        matricesEz,
        nRRow,
        nZColumn,
        phiSlice,
        maxIteration,
        symmetry);
      lookupEField->CopyFromMatricesToInterpolator();
      Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "Step 3: Fill the look up table");

      if (side == 0) {
        FillLookUpTable(lookupEField,
                        fMatrixErOverEzA, fMatrixEPhiOverEzA, fMatrixDeltaEzA,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);
        fLookupIntENoDriftA->CopyFromMatricesToInterpolator();
        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " A side done");
      }
      if (side == 1) {
        FillLookUpTable(lookupEField,
                        fMatrixErOverEzC, fMatrixEPhiOverEzC, fMatrixDeltaEzC,
                        nRRow, nZColumn, phiSlice, rList, phiList, zList);
        fLookupIntENoDriftC->CopyFromMatricesToInterpolator();

        Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", " C side done");
      }
      delete lookupEField;
    }
    fInitLookUp = kTRUE;
  }

  for (Int_t k = 0; k < phiSlice; k++) {
    delete matricesV[k];
    delete matricesCharge[k];
    delete matricesEr[k];
    delete matricesEPhi[k];
    delete matricesEz[k];
  }
}
/// Force creating look-up table of Correction/Distortion by integration following
/// drift line.
///
/// \param nRRow Int_t Number of nRRow in r-direction
/// \param nZColumn Int_t Number of nZColumn in z-direction
/// \param phiSlice Int_t Number of phi slices in \f$ phi \f$ direction
/// \param maxIteration Int_t Maximum iteration for poisson solver
/// \param stoppingConvergence Convergence error stopping condition for poisson solver
///
void AliTPCSpaceCharge3DCalc::ForceInitSpaceCharge3DPoissonIntegralDz(Int_t nRRow, Int_t nZColumn, Int_t phiSlice,
                                                                      Int_t maxIteration, Double_t stoppingConvergence)
{
  fInitLookUp = kFALSE;
  InitSpaceCharge3DPoissonIntegralDz(nRRow, nZColumn, phiSlice, maxIteration, stoppingConvergence);
}
/// Electric field Calculation:
///
///
/// \param matricesV
/// \param matricesEr
/// \param matricesEPhi
/// \param matricesEz
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
/// \param gridSizeR
/// \param gridSizePhi
/// \param gridSizeZ
/// \param symmetry
/// \param innerRadius
///
/// \pre   Matrix matricesV is assumed had been calculated  by Poisson solver
/// \post  Results of  E-fields are calculated by measuring gradient at potential distribution
///
///
///	* Differentiate potential on all direction (r,z and phi)
/// * Non-boundary -> Central difference (3 stencil) TODO: 5 Stencil
///
///   \f$  \nabla_{r} V(r_{i},\phi_{j},z_{k}) \approx -( V_{i+1,j,k} - V_{i-1,j,k}) / (2* h_{r}) \f$
///
///   \f$ -\nabla_{\phi} V(r_{i},\phi_{j},z_{k}) \approx -( V_{i,j-1,k} - V_{i,j+1,k}) / (2* r_{j} * h_{\phi}) \f$
///
///   \f$ -\nabla_{z} V(r_{i},\phi_{j},z_{k}) \approx -( V_{i,j,k+1} - V_{i,j,k-1}) / (2* h_{z}) \f$
///
///   ~~~ cxx
///   matrixEr(i,j) = -1 * ( arrayV(i+1,j) - arrayV(i-1,j) ) / (2*gridSizeR); // r direction
///		matrixEz(i,j) = -1 * ( arrayV(i,j+1) - arrayV(i,j-1) ) / (2*gridSizeZ) ; // z direction
///		matrixEPhi(i,j) = -1 * (signPlus * arrayVP(i,j) - signMinus * arrayVM(i,j) ) / (2*radius*gridSizePhi)
///   ~~~
///
/// * Boundary -> Forward/Backward difference (3 stencil) TODO: 5 Stencil
///
///   \f$ -\nabla_{r} V(r_{0},\phi_{j},z_{k}) \approx -( -0.5 V_{2,j,k} + 2 V_{1,j,k} - 1.5 * V_{0,j,k}) /  h_{r} \f$
///
///   \f$ -\nabla_{r} V(r_{nRRow - 1},\phi_{j},z_{k}) \approx -( 1.5 V_{nRRow-1,j,k} - 2.0 V_{nRRow-2,j,k} + 0.5 V_{nRRow -3,j,k}) / h_{\phi} \f$
///
void AliTPCSpaceCharge3DCalc::ElectricField(TMatrixD** matricesV, TMatrixD** matricesEr, TMatrixD** matricesEPhi,
                                            TMatrixD** matricesEz, const Int_t nRRow, const Int_t nZColumn,
                                            const Int_t phiSlice,
                                            const Float_t gridSizeR, const Float_t gridSizePhi,
                                            const Float_t gridSizeZ,
                                            const Int_t symmetry, const Float_t innerRadius)
{
  Float_t radius;
  Int_t mPlus, mMinus, signPlus, signMinus;
  for (Int_t m = 0; m < phiSlice; m++) {
    mPlus = m + 1;
    signPlus = 1;
    mMinus = m - 1;
    signMinus = 1;
    if (symmetry == 1) { // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
      }
      if (mMinus < 0) {
        mMinus = 1;
      }
    } else if (symmetry == -1) { // Anti-symmetry in phi
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
        signPlus = -1;
      }
      if (mMinus < 0) {
        mMinus = 1;
        signMinus = -1;
      }
    } else { // No Symmetries in phi, no boundaries, the calculations is continuous across all phi
      if (mPlus > phiSlice - 1) {
        mPlus = m + 1 - phiSlice;
      }
      if (mMinus < 0) {
        mMinus = m - 1 + phiSlice;
      }
    }

    TMatrixD& arrayVP = *matricesV[mPlus];
    TMatrixD& arrayVM = *matricesV[mMinus];
    TMatrixD& arrayV = *matricesV[m];
    TMatrixD& matrixEr = *matricesEr[m];
    TMatrixD& matrixEz = *matricesEz[m];
    TMatrixD& matrixEPhi = *matricesEPhi[m];

    // for non-boundary V
    for (Int_t i = 1; i < nRRow - 1; i++) {
      radius = innerRadius + i * gridSizeR;
      for (Int_t j = 1; j < nZColumn - 1; j++) {
        matrixEr(i, j) = -1 * (arrayV(i + 1, j) - arrayV(i - 1, j)) / (2 * gridSizeR); // r direction
        matrixEz(i, j) = -1 * (arrayV(i, j + 1) - arrayV(i, j - 1)) / (2 * gridSizeZ); // z direction
        matrixEPhi(i, j) = -1 * (signPlus * arrayVP(i, j) - signMinus * arrayVM(i, j)) /
                           (2 * radius * gridSizePhi); // phi direction
      }
    }

    // for boundary-r
    for (Int_t j = 0; j < nZColumn; j++) {
      matrixEr(0, j) = -1 * (-0.5 * arrayV(2, j) + 2.0 * arrayV(1, j) - 1.5 * arrayV(0, j)) /
                       gridSizeR; // forward difference
      matrixEr(nRRow - 1, j) =
        -1 * (1.5 * arrayV(nRRow - 1, j) - 2.0 * arrayV(nRRow - 2, j) + 0.5 * arrayV(nRRow - 3, j)) /
        gridSizeR; // backward difference
    }

    for (Int_t i = 0; i < nRRow; i += nRRow - 1) {
      radius = innerRadius + i * gridSizeR;
      for (Int_t j = 1; j < nZColumn - 1; j++) {
        matrixEz(i, j) = -1 * (arrayV(i, j + 1) - arrayV(i, j - 1)) / (2 * gridSizeZ); // z direction
        matrixEPhi(i, j) = -1 * (signPlus * arrayVP(i, j) - signMinus * arrayVM(i, j)) /
                           (2 * radius * gridSizePhi); // phi direction
      }
    }

    // for boundary-z
    for (Int_t i = 0; i < nRRow; i++) {
      matrixEz(i, 0) = -1 * (-0.5 * arrayV(i, 2) + 2.0 * arrayV(i, 1) - 1.5 * arrayV(i, 0)) / gridSizeZ;
      matrixEz(i, nZColumn - 1) =
        -1 *
        (1.5 * arrayV(i, nZColumn - 1) - 2.0 * arrayV(i, nZColumn - 2) + 0.5 * arrayV(i, nZColumn - 3)) /
        gridSizeZ;
    }

    for (Int_t i = 1; i < nRRow - 1; i++) {
      radius = innerRadius + i * gridSizeR;
      for (Int_t j = 0; j < nZColumn; j += nZColumn - 1) {
        matrixEr(i, j) = -1 * (arrayV(i + 1, j) - arrayV(i - 1, j)) / (2 * gridSizeR); // r direction
        matrixEPhi(i, j) = -1 * (signPlus * arrayVP(i, j) - signMinus * arrayVM(i, j)) /
                           (2 * radius * gridSizePhi); // phi direction
      }
    }

    // corner points for EPhi
    for (Int_t i = 0; i < nRRow; i += nRRow - 1) {
      radius = innerRadius + i * gridSizeR;
      for (Int_t j = 0; j < nZColumn; j += nZColumn - 1) {
        matrixEPhi(i, j) = -1 * (signPlus * arrayVP(i, j) - signMinus * arrayVM(i, j)) /
                           (2 * radius * gridSizePhi); // phi direction
      }
    }
  }
}
///
/// Local distortion and correction, calculate local distortion/correction
/// based on simplified langevin equation, see internal note ALICE-INT-2010-016.
///
/// <b> Local Distortion </b>
///
/// Local distortion is calculated based on formulation in ALICE-INT-2010-016, this function assume that
/// electric field \f$\vec{E}(r_{i},z_{j},\phi_{m})\f$ is provided.
///
/// First, we calculate integration of the Electric field in z-direction for all direction.
/// Assumption: \f$ z_{0} \f$ is location of CE (Central Electrode) and \f$ z_{nZColumn - 1} \f$ is location of End Plate.
///
/// This integration is in \f$z\f$ direction we can only use trapezoidal rule.
///
/// Let suppose we want to calculate local distortion at \f$(r_{i},z_{j},\phi_{m})\f$.
/// Assume \f$\vec{E}(r_{i},z_{j+1},\phi_{m}) \f$ and \f$\vec{E}(r_{i},z_{j},\phi_{m}) \f$ are known, see Figure \ref fig1 (a),
///
/// \anchor fig1
/// ![Local Distortion](localdist.png)
///
/// Than we can calculate definite integrations for each directions in respect of $z$  from \f$ z_{j} \f$  to \f$ z_{j + 1} \f$   as follows:
///
/// \f$  \int^{z_{j+1}}_{z_{j}} \frac{E_{r}}{E_{z}}(r_{i},z_{j},\phi_{m}) dzDist  \approx \frac{-1}{\mathrm{ezField}} \frac{h_{z}}{2.0} \left( E_{r}(r_{i},z_{j},\phi_{m}) + E_{r}(r_{i},z_{j+1},\phi_{m}) \right)\f$
///
/// \f$  \int^{z_{j+1}}_{z_{j}} \frac{E_{\phi}}{E_{z}}(r_{i},z_{j},\phi_{m}) dzDist  \approx  \frac{-1}{\mathrm{ezField}} \frac{h_{z}}{2.0} \left( E_{\phi}(r_{i},z_{j},\phi_{m}) + E_{\phi}(r_{i},z_{j+1},\phi_{m}) \right)\f$
///
/// \f$  \int^{z_{j+1}}_{z_{j}} E_{z}(r_{i},z_{j},\phi_{m}) dzDist  \approx   \frac{h_{z}}{2.0} \left( E_{z}(r_{i},z_{j},\phi_{m}) + E_{z}(r_{i},z_{j+1},\phi_{m}) \right) \f$
///
/// Code sample at \ref impllocaldist is an implementation of the local integration of electric field.
///
/// \anchor impllocaldist
/// ~~~
/// Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV-AliTPCPoissonSolver::fgkGG)/AliTPCPoissonSolver::fgkTPCZ0; // = Electric Field (V/cm) Magnitude ~ -400 V/cm;
///
/// localIntErOverEz = (gridSizeZ/2.0)*((*eR)(i,j)+(*eR)(i,j+1))/(ezField + (*eZ)(i,j)) ;
/// localIntEPhiOverEz = (gridSizeZ/2.0)*((*ePhi)(i,j)+(*ePhi)(i,j+1))/(ezField + (*eZ)(i,j)) ;
/// localIntDeltaEz = (gridSizeZ/2.0)*((*eZ)(i,j)+(*eZ)(i,j+1)) ;
/// ~~~
///
///
/// After we have local integrations for electric fields in each direction,
/// local distortion \f$\hat{\delta}(r_{i},z_{j},\phi_{m})\f$ is calculated by simplified Langevin equation (see Figure \ref1 (b) for illustration):
///
/// \f$ \hat{\delta}_{rE}(r_{i},z_{j},\phi_{m}) = c_{0} \int^{z_{j+1}}_{z_{j}} \frac{E_{r}}{E_{z}} dzDist   + c_{1}  \int^{z_{j+1}}_{z_{j}} \frac{E_{\phi}}{E_{z}} dzDist \f$
///
/// ~~~
///	(*distDrDz)(i,j)        = fC0*localIntErOverEz   + fC1*localIntEPhiOverEz;
/// ~~~
///
/// \f$ r\hat{\delta}_{\phi E}(r_{i},z_{j},\phi_{m})  = - c_{1} \int^{z_{j+1}}_{z_{j}} \frac{E_{j}}{E_{j}} dzDist  + c_{0} \int^{z_{j+1}}_{j_{j}} \frac{E_{\phi}}{E_{z}} dzDist \f$
///
/// ~~~
///	(*distDPhiRDz)(i,j) = fC0*localIntEPhiOverEz - fC1*localIntErOverEz ;
/// ~~~
///
/// \f$ \hat{\delta}_{z}(r_{i},z_{j},\phi_{m})  = \int_{z_{j}}^{z_{j+1}} \frac{v^{\prime}(E)}{v_{0}} (E - E_{0}) dzDist\f$
///
/// ~~~
/// (*distDz)(i,j) = localIntDeltaEz*-1*AliTPCPoissonSolver::fgkdvdE;
/// ~~~
///
/// Where \f$c_{0}\f$ and \f$c_{1}\f$ are constants (see the ALICE-INT-2010-016 for further details).
///
/// <b> Local correction </b>
///
/// Local correction is computed as local distortion where the electric fields are in opposite direction (see Figure \ref fig2 (a)).
///
/// \anchor fig2
/// ![Local Correction](localcorr.png)
///
/// Let suppose we want to calculate local correction at \f$(r_{i},\mathbf{z_{j+1}},\phi_{m})\f$.
/// Assume \f$\vec{E}(r_{i},z_{j+1},\phi_{m}) \f$ and \f$\vec{E}(r_{i},z_{j},\phi_{m}) \f$ are known.
///
/// Than we can calculate definite integrations for each directions in respect of \f$z\f$  from \f$ z_{j+1} \f$  to \f$ z_{j} \f$   as follows:
///
/// \f$  \int^{z_{j}}_{z_{j+1}} \frac{E_{r}}{E_{z}}(r_{i},z_{j},\phi_{m}) dzDist  \approx -1 * \frac{-1}{\mathrm{ezField}} \frac{h_{z}}{2.0} \left( E_{r}(r_{i},z_{j},\phi_{m}) + E_{r}(r_{i},z_{j+1},\phi_{m}) \right)\f$
///
/// \f$  \int^{z_{j}}_{z_{j+1}} \frac{E_{\phi}}{E_{z}}(r_{i},z_{j},\phi_{m}) dzDist  \approx  -1 *  \frac{-1}{\mathrm{ezField}} \frac{h_{z}}{2.0} \left( E_{\phi}(r_{i},z_{j},\phi_{m}) + E_{\phi}(r_{i},z_{j+1},\phi_{m}) \right)\f$
///
/// \f$  \int^{z_{j}}_{z_{j+1}} E_{z}(r_{i},z_{j},\phi_{m}) dzDist  \approx  -1 *   \frac{h_{z}}{2.0} \left( E_{z}(r_{i},z_{j},\phi_{m}) + E_{z}(r_{i},z_{j+1},\phi_{m}) \right) \f$
///
/// Local correction at \f$\hat{\delta'}(r_{i},\mathbf{z_{j+1}},\phi_{m})\f$ is calculated by simplified Langevin equation (see Figure \ref fig2 (b) for illustration):
///
/// \f$ \hat{\delta'}_{rE}(r_{i},z_{j+1},\phi_{m}) = c_{0} \int^{z_{j}}_{z_{j+1}} \frac{E_{r}}{E_{z}} dzDist   + c_{1}  \int^{z_{j-1}}_{z_{j}} \frac{E_{\phi}}{E_{z}} dzDist \f$
///
/// \f$ r\hat{\delta'}_{\phi E}(r_{i},z_{j+1},\phi_{m})  = - c_{1} \int^{z_{j}}_{z_{j+1}} \frac{E_{j}}{E_{j}} dzDist  + c_{0} \int^{z_{j-1}}_{j_{k}} \frac{E_{\phi}}{E_{z}} dzDist \f$
///
/// \f$ \hat{\delta'}_{z}(r_{i},z_{j+1},\phi_{m})  = \int_{z_{j}}^{z_{j+1}} \frac{v^{\prime}(E)}{v_{0}} (E - E_{0}) dzDist\f$
///
/// For implementation, we use the fact that
///
/// \f$ \hat{\delta'}_{rE}(r_{i},z_{j+1},\phi_{m}) = -1 * \hat{\delta}_{rE}(r_{i},z_{j},\phi_{m}) \f$
///
/// \f$ r\hat{\delta'}_{\phi E}(r_{i},z_{j+1},\phi_{m}) =  -1 *  r\hat{\delta}_{\phi E}(r_{i},z_{j},\phi_{m}) \f$
///
/// \f$ \hat{\delta'}_{z}(r_{i},z_{j+1},\phi_{m}) =  -1 *  \hat{\delta}_{z}(r_{i},z_{j},\phi_{m}) \f$
///
/// ~~~
///	(*corrDrDz)(i,j+1)      = -1* (*distDrDz)(i,j) ;
/// (*corrDPhiRDz)(i,j+1) = -1* (*distDPhiRDz)(i,j);
/// (*corrDz)(i,j+1)      = -1* (*distDz)(i,j);
/// ~~~
///
/// \param matricesEr TMatrixD**  electric field for \f$r\f$ component
///	\param matricesEPhi TMatrixD** electric field for \f$\phi\f$ component
///	\param matricesEz TMatrixD** electric field for \f$z\f$ component
///	\param matricesDistDrDz TMatrixD**  local distortion \f$\hat{\delta}_{r}\f$
///	\param matricesDistDPhiRDz TMatrixD** local distortion \f$r \hat{\delta}_{\phi}\f$
///	\param matricesDistDz TMatrixD**   local distortion \f$ \hat{\delta}_{z}\f$
///	\param matricesCorrDrDz TMatrixD** local correction \f$\hat{\delta}_{r}\f$
///	\param matricesCorrDPhiRDz TMatrixD** local correction \f$r \hat{\delta}_{\phi}\f$
///	\param matricesCorrDz TMatrixD** local correction \f$ \hat{\delta}_{z}\f$
/// \param nRRow Int_t Number of nRRow in r-direction
/// \param nZColumn Int_t Number of nZColumn in z-direction
/// \param phiSlice Int_t Number of phi slices in \f$ phi \f$ direction
///	\param gridSizeZ const Float_t grid size in z direction
/// \param ezField const Double_t ezField calculated from the invoking operation
///
/// \pre matricesEr, matricesEPhi, matrices Ez assume already been calculated
/// \post Local distortion and correction are computed according simplified Langevin equation
/// ~~~
/// matricesDistDrDz,matricesDistDPhiRDz,matricesDistDz
/// ~~~
/// and correction:
/// ~~~
/// matricesCorrDrDz,matricesCorrDPhiRDz,matricesCorrDz
/// ~~~
///
void AliTPCSpaceCharge3DCalc::LocalDistCorrDz(TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
                                              TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz,
                                              TMatrixD** matricesDistDz,
                                              TMatrixD** matricesCorrDrDz, TMatrixD** matricesCorrDPhiRDz,
                                              TMatrixD** matricesCorrDz,
                                              const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
                                              const Float_t gridSizeZ,
                                              const Double_t ezField)
{
  Float_t localIntErOverEz = 0.0;
  Float_t localIntEPhiOverEz = 0.0;
  Float_t localIntDeltaEz = 0.0;
  TMatrixD* eR;
  TMatrixD* ePhi;
  TMatrixD* eZ;
  TMatrixD* distDrDz;
  TMatrixD* distDPhiRDz;
  TMatrixD* distDz;
  TMatrixD* corrDrDz;
  TMatrixD* corrDPhiRDz;
  TMatrixD* corrDz;

  // Initialization for j == column-1 integration is 0.0
  for (Int_t m = 0; m < phiSlice; m++) {
    distDrDz = matricesDistDrDz[m];
    distDPhiRDz = matricesDistDPhiRDz[m];
    distDz = matricesDistDz[m];

    corrDrDz = matricesCorrDrDz[m];
    corrDPhiRDz = matricesCorrDPhiRDz[m];
    corrDz = matricesCorrDz[m];

    for (Int_t i = 0; i < nRRow; i++) {
      (*distDrDz)(i, nZColumn - 1) = 0.0;
      (*distDPhiRDz)(i, nZColumn - 1) = 0.0;
      (*distDz)(i, nZColumn - 1) = 0.0;

      (*corrDrDz)(i, 0) = 0.0;
      (*corrDPhiRDz)(i, 0) = 0.0;
      (*corrDz)(i, 0) = 0.0;
    }
  }

  // for this case
  // use trapezoidal rule assume no ROC displacement
  for (Int_t m = 0; m < phiSlice; m++) {
    eR = matricesEr[m];
    ePhi = matricesEPhi[m];
    eZ = matricesEz[m];
    distDrDz = matricesDistDrDz[m];
    distDPhiRDz = matricesDistDPhiRDz[m];
    distDz = matricesDistDz[m];

    corrDrDz = matricesCorrDrDz[m];
    corrDPhiRDz = matricesCorrDPhiRDz[m];
    corrDz = matricesCorrDz[m];

    for (Int_t j = 0; j < nZColumn - 1; j++) {
      for (Int_t i = 0; i < nRRow; i++) {
        localIntErOverEz = (gridSizeZ * 0.5) * ((*eR)(i, j) + (*eR)(i, j + 1)) / (ezField + (*eZ)(i, j));
        localIntEPhiOverEz = (gridSizeZ * 0.5) * ((*ePhi)(i, j) + (*ePhi)(i, j + 1)) / (ezField + (*eZ)(i, j));
        localIntDeltaEz = (gridSizeZ * 0.5) * ((*eZ)(i, j) + (*eZ)(i, j + 1));

        (*distDrDz)(i, j) = fC0 * localIntErOverEz + fC1 * localIntEPhiOverEz;
        (*distDPhiRDz)(i, j) = fC0 * localIntEPhiOverEz - fC1 * localIntErOverEz;
        (*distDz)(i, j) = localIntDeltaEz * -1 * AliTPCPoissonSolver::fgkdvdE;

        (*corrDrDz)(i, j + 1) = -1 * (*distDrDz)(i, j);
        (*corrDPhiRDz)(i, j + 1) = -1 * (*distDPhiRDz)(i, j);
        (*corrDz)(i, j + 1) = -1 * (*distDz)(i, j);
      }
    }
  }
}

/// IntegrateDistCorrDriftLineDz, integration of local distortion by following electron drift

/// See explanation at LocalDistCorrDz
///
///
/// \param matricesEr TMatrixD**  electric field for \f$r\f$ component
///	\param matricesEPhi TMatrixD** electric field for \f$\phi\f$ component
///	\param matricesEz TMatrixD** electric field for \f$z\f$ component
///	\param matricesCorrDrDz TMatrixD** local correction \f$\hat{\delta}_{r}\f$
///	\param matricesCorrDPhiRDz TMatrixD** local correction \f$r \hat{\delta}_{\phi}\f$
///	\param matricesCorrDz TMatrixD** local correction \f$ \hat{\delta}_{z}\f$
/// \param nRRow Int_t Number of nRRow in r-direction
/// \param nZColumn Int_t Number of nZColumn in z-direction
/// \param phiSlice Int_t Number of phi slices in \f$ phi \f$ direction
///	\param gridSizeZ const Float_t grid size in z direction
/// \param ezField const Double_t ezField calculate from the invoking operation
///
/// \pre matricesEr, matricesEPhi, matrices Ez are provided
/// \post Local correction are computed according simplified Langevin equation
/// ~~~
/// matricesCorrDz,matricesCorrDPhiRDz,matricesDistDz
/// ~~~
///
void AliTPCSpaceCharge3DCalc::IntegrateDistCorrDriftLineDz(
  AliTPCLookUpTable3DInterpolatorD* lookupLocalDist,
  TMatrixD** matricesGDistDrDz,
  TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz,
  AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr,
  TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
  TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz,
  TMatrixD** matricesGCorrIrregularDz,
  TMatrixD** matricesRIrregular, TMatrixD** matricesPhiIrregular, TMatrixD** matricesZIrregular,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
  const Double_t* rList, const Double_t* phiList, const Double_t* zList)
{

  Float_t drDist, dPhi, dzDist, ddR, ddRPhi, ddZ;
  Float_t radius0, phi0, z0, radius, phi, z, radiusCorrection;
  radiusCorrection = 0.0;
  radius = 0.0;
  TMatrixD* mDistDrDz;
  TMatrixD* mDistDPhiRDz;
  TMatrixD* mDistDz;
  TMatrixD* mCorrDrDz;
  TMatrixD* mCorrDPhiRDz;
  TMatrixD* mCorrDz;
  TMatrixD* mCorrIrregularDrDz;
  TMatrixD* mCorrIrregularDPhiRDz;
  TMatrixD* mCorrIrregularDz;
  TMatrixD* mRIrregular;
  TMatrixD* mPhiIrregular;
  TMatrixD* mZIrregular;
  Int_t j = nZColumn - 1;
  z0 = zList[j];

  for (Int_t m = 0; m < phiSlice; m++) {
    phi0 = phiList[m];

    mDistDrDz = matricesGDistDrDz[m];
    mDistDPhiRDz = matricesGDistDPhiRDz[m];
    mDistDz = matricesGDistDz[m];

    //
    mCorrDrDz = matricesGCorrDrDz[m];
    mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
    mCorrDz = matricesGCorrDz[m];

    mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
    mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
    mCorrIrregularDz = matricesGCorrIrregularDz[m];

    mRIrregular = matricesRIrregular[m];
    mPhiIrregular = matricesPhiIrregular[m];
    mZIrregular = matricesZIrregular[m];

    for (Int_t i = 0; i < nRRow; i++) {
      // do from j to 0
      // follow the drift
      radius0 = rList[i];
      phi = phi0;

      ///
      (*mDistDrDz)(i, j) = 0.;
      (*mDistDPhiRDz)(i, j) = 0.;
      (*mDistDz)(i, j) = 0.;

      //////////////// use irregular grid look up table for correction
      if (fCorrectionType == kIrregularInterpolator) {
        (*mCorrIrregularDrDz)(i, j) = 0.0;
        (*mCorrIrregularDPhiRDz)(i, j) = 0.0;
        (*mCorrIrregularDz)(i, j) = -0.0;

        // distorted point
        (*mRIrregular)(i, j) = radius0;
        (*mPhiIrregular)(i, j) = phi0;
        (*mZIrregular)(i, j) = z0;
      }
      ///////////////
    }
  }

  // from j one column near end cap
  for (j = nZColumn - 2; j >= 0; j--) {

    z0 = zList[j];
    for (Int_t m = 0; m < phiSlice; m++) {
      phi0 = phiList[m];

      mDistDrDz = matricesGDistDrDz[m];
      mDistDPhiRDz = matricesGDistDPhiRDz[m];
      mDistDz = matricesGDistDz[m];

      //
      mCorrDrDz = matricesGCorrDrDz[m];
      mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
      mCorrDz = matricesGCorrDz[m];

      mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
      mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
      mCorrIrregularDz = matricesGCorrIrregularDz[m];

      mRIrregular = matricesRIrregular[m];
      mPhiIrregular = matricesPhiIrregular[m];
      mZIrregular = matricesZIrregular[m];

      for (Int_t i = 0; i < nRRow; i++) {
        // do from j to 0
        // follow the drift
        radius0 = rList[i];
        phi = phi0;
        radius = radius0;

        drDist = 0.0;
        dPhi = 0.0;
        dzDist = 0.0;

        // follow the drift line from z=j --> nZColumn - 1
        for (Int_t jj = j; jj < nZColumn; jj++) {
          // interpolation the local distortion for current position
          // phi += ddRPhi / radius;
          phi = phi0 + dPhi;
          radius = radius0 + drDist;
          z = zList[jj] + dzDist;

          // regulate phi
          while (phi < 0.0) {
            phi = TMath::TwoPi() + phi;
          }
          while (phi > TMath::TwoPi()) {
            phi = phi - TMath::TwoPi();
          }

          lookupLocalDist->GetValue(radius, phi, z, ddR, ddRPhi, ddZ);

          // add local distortion
          drDist += ddR;
          dPhi += (ddRPhi / radius);
          dzDist += ddZ;
        }
        // set the global distortion after following the electron drift
        (*mDistDrDz)(i, j) = drDist;
        (*mDistDPhiRDz)(i, j) = dPhi * radius0;
        (*mDistDz)(i, j) = dzDist;
        /////////////// use irregular grid look up table for correction
        // set
        if (fCorrectionType == kIrregularInterpolator) {
          (*mCorrIrregularDrDz)(i, j) = -drDist;
          (*mCorrIrregularDPhiRDz)(i, j) = -1 * dPhi * (radius0 + drDist);
          (*mCorrIrregularDz)(i, j) = -dzDist;

          // distorted point
          (*mRIrregular)(i, j) = radius0 + drDist;
          (*mPhiIrregular)(i, j) = phi0 + dPhi;
          (*mZIrregular)(i, j) = z0 + dzDist;
        }
        ///////////////

        // put the radius to the original value
        if (fCorrectionType == kRegularInterpolator) {
          if (j == nZColumn - 2) {
            radiusCorrection = radius0;
          }

          // get global correction from j+1
          drDist = (*mCorrDrDz)(i, j + 1);
          dPhi = (*mCorrDPhiRDz)(i, j + 1) / radius0;
          dzDist = (*mCorrDz)(i, j + 1);

          radiusCorrection = radius0 + drDist;
          phi = phi0 + dPhi;
          z = zList[j + 1] + dzDist;

          while (phi < 0.0) {
            phi = TMath::TwoPi() + phi;
          }
          while (phi > TMath::TwoPi()) {
            phi = phi - TMath::TwoPi();
          }

          lookupLocalCorr->GetValue(radiusCorrection, phi, z, ddR, ddRPhi, ddZ);

          drDist += ddR;
          dzDist += ddZ;
          dPhi += ddRPhi / radiusCorrection;

          (*mCorrDrDz)(i, j) = drDist;
          (*mCorrDPhiRDz)(i, j) = dPhi * radius0;
          (*mCorrDz)(i, j) = dzDist;
        }
      }
    }
  }
}

// oudated, to be removed once changes in aliroot are pushed
/// follow the drift for exact function
///
/// \param intDrDzF
/// \param intDPhiDzF
/// \param intDzDzF
/// \param ezField
/// \param matricesGDistDrDz
/// \param matricesGDistDPhiRDz
/// \param matricesGDistDz
/// \param matricesGCorrDrDz
/// \param matricesGCorrDPhiRDz
/// \param matricesGCorrDz
/// \param matricesGCorrIrregularDrDz
/// \param matricesGCorrIrregularDPhiRDz
/// \param matricesGCorrIrregularDz
/// \param matricesRIrregular
/// \param matricesPhiIrregular
/// \param matricesZIrregular
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
/// \param rList
/// \param phiList
/// \param zList
void AliTPCSpaceCharge3DCalc::IntegrateDistCorrDriftLineDz(
  TFormula* intDrDzF, TFormula* intDPhiDzF, TFormula* intDzDzF, const Double_t ezField,
  TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz,
  TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
  TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz, TMatrixD** matricesGCorrIrregularDz, TMatrixD** matricesRIrregular,
  TMatrixD** matricesPhiIrregular, TMatrixD** matricesZIrregular, const Int_t nRRow, const Int_t nZColumn,
  const Int_t phiSlice, const Double_t* rList, const Double_t* phiList, const Double_t* zList)
{

  Float_t drDist, dRPhi, dzDist, ddR, ddRPhi, ddZ;
  Float_t radius0, phi0, z0, radius, phi, z, radiusCorrection, z1;

  Float_t localIntErOverEz = 0.0;
  Float_t localIntEPhiOverEz = 0.0;
  Float_t localIntDeltaEz = 0.0;

  radiusCorrection = 0.0;
  radius = 0.0;
  TMatrixD* mDistDrDz;
  TMatrixD* mDistDPhiRDz;
  TMatrixD* mDistDz;
  TMatrixD* mCorrDrDz;
  TMatrixD* mCorrDPhiRDz;
  TMatrixD* mCorrDz;
  TMatrixD* mCorrIrregularDrDz;
  TMatrixD* mCorrIrregularDPhiRDz;
  TMatrixD* mCorrIrregularDz;
  TMatrixD* mRIrregular;
  TMatrixD* mPhiIrregular;
  TMatrixD* mZIrregular;
  Int_t j = nZColumn - 1;
  z0 = zList[j];

  for (Int_t m = 0; m < phiSlice; m++) {
    phi0 = phiList[m];

    mDistDrDz = matricesGDistDrDz[m];
    mDistDPhiRDz = matricesGDistDPhiRDz[m];
    mDistDz = matricesGDistDz[m];

    //
    mCorrDrDz = matricesGCorrDrDz[m];
    mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
    mCorrDz = matricesGCorrDz[m];

    mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
    mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
    mCorrIrregularDz = matricesGCorrIrregularDz[m];

    mRIrregular = matricesRIrregular[m];
    mPhiIrregular = matricesPhiIrregular[m];
    mZIrregular = matricesZIrregular[m];

    for (Int_t i = 0; i < nRRow; i++) {
      // do from j to 0
      // follow the drift
      radius0 = rList[i];
      phi = phi0;
      radius = radius0;

      drDist = 0.0;
      dRPhi = 0.0;
      dzDist = 0.0;
      ddRPhi = 0.0;

      ///
      (*mDistDrDz)(i, j) = drDist;
      (*mDistDPhiRDz)(i, j) = dRPhi;
      (*mDistDz)(i, j) = dzDist;

      //////////////// use irregular grid look up table for correction
      // set
      (*mCorrIrregularDrDz)(i, j) = -drDist;
      (*mCorrIrregularDPhiRDz)(i, j) = -dRPhi;
      (*mCorrIrregularDz)(i, j) = -dzDist;

      // distorted point
      (*mRIrregular)(i, j) = radius0 + drDist;
      (*mPhiIrregular)(i, j) = phi0 + (dRPhi / radius0);
      (*mZIrregular)(i, j) = z0 + dzDist;
      ///////////////
    }
  }

  // from j one column near end cap
  for (j = nZColumn - 2; j >= 0; j--) {

    z0 = zList[j];

    for (Int_t m = 0; m < phiSlice; m++) {
      phi0 = phiList[m];

      mDistDrDz = matricesGDistDrDz[m];
      mDistDPhiRDz = matricesGDistDPhiRDz[m];
      mDistDz = matricesGDistDz[m];

      //
      mCorrDrDz = matricesGCorrDrDz[m];
      mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
      mCorrDz = matricesGCorrDz[m];

      mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
      mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
      mCorrIrregularDz = matricesGCorrIrregularDz[m];

      mRIrregular = matricesRIrregular[m];
      mPhiIrregular = matricesPhiIrregular[m];
      mZIrregular = matricesZIrregular[m];

      for (Int_t i = 0; i < nRRow; i++) {
        // do from j to 0
        // follow the drift
        radius0 = rList[i];
        phi = phi0;
        radius = radius0;

        drDist = 0.0;
        dRPhi = 0.0;
        dzDist = 0.0;
        ddRPhi = 0.0;

        // follow the drift line from z=j --> nZColumn - 1
        for (Int_t jj = j; jj < nZColumn; jj++) {
          // interpolation the local distortion for current position
          phi += ddRPhi / radius;
          radius = radius0 + drDist;
          z = zList[jj] + dzDist;
          z1 = z + (zList[j + 1] - zList[j]);

          // regulate phi
          while (phi < 0.0) {
            phi = TMath::TwoPi() + phi;
          }
          while (phi > TMath::TwoPi()) {
            phi = phi - TMath::TwoPi();
          }

          //lookupLocalDist->GetValue(radius, phi, z, ddR, ddRPhi, ddZ);
          localIntErOverEz = (intDrDzF->Eval(radius, phi, z1) - intDrDzF->Eval(radius, phi, z)) / (-1 * ezField);
          localIntEPhiOverEz = (intDPhiDzF->Eval(radius, phi, z1) - intDPhiDzF->Eval(radius, phi, z)) / (-1 * ezField);
          localIntDeltaEz = intDzDzF->Eval(radius, phi, z1) - intDzDzF->Eval(radius, phi, z);

          ddR = fC0 * localIntErOverEz + fC1 * localIntEPhiOverEz;
          ddRPhi = fC0 * localIntEPhiOverEz - fC1 * localIntErOverEz;
          ddZ = localIntDeltaEz * AliTPCPoissonSolver::fgkdvdE * AliTPCPoissonSolver::fgkdvdE; // two times?

          // add local distortion
          drDist += ddR;
          dRPhi += ddRPhi;
          dzDist += ddZ;
        }
        // set the global distortion after following the electron drift
        (*mDistDrDz)(i, j) = drDist;
        (*mDistDPhiRDz)(i, j) = dRPhi;
        (*mDistDz)(i, j) = dzDist;
        /////////////// use irregular grid look up table for correction
        // set
        (*mCorrIrregularDrDz)(i, j) = -drDist;
        (*mCorrIrregularDPhiRDz)(i, j) = -dRPhi;
        (*mCorrIrregularDz)(i, j) = -dzDist;

        // distorted point
        (*mRIrregular)(i, j) = radius0 + drDist;
        (*mPhiIrregular)(i, j) = phi0 + (dRPhi / radius0);
        (*mZIrregular)(i, j) = z0 + dzDist;
        ///////////////

        // put the radius to the original value
        if (j == nZColumn - 2) {
          radiusCorrection = radius0;
        }

        // get global correction from j+1
        drDist = (*mCorrDrDz)(i, j + 1);
        dRPhi = (*mCorrDPhiRDz)(i, j + 1);
        dzDist = (*mCorrDz)(i, j + 1);

        radiusCorrection = radius0 + drDist;
        phi = phi0 + dRPhi / radiusCorrection;
        z = zList[j + 1] + dzDist;
        z1 = z - (zList[j + 1] - zList[j]);

        while (phi < 0.0) {
          phi = TMath::TwoPi() + phi;
        }
        while (phi > TMath::TwoPi()) {
          phi = phi - TMath::TwoPi();
        }

        //lookupLocalCorr->GetValue(radiusCorrection, phi, z, ddR, ddRPhi, ddZ);
        localIntErOverEz = (intDrDzF->Eval(radiusCorrection, phi, z1) - intDrDzF->Eval(radiusCorrection, phi, z)) / (-1 * ezField);
        localIntEPhiOverEz = (intDPhiDzF->Eval(radiusCorrection, phi, z1) - intDPhiDzF->Eval(radiusCorrection, phi, z)) / (-1 * ezField);
        localIntDeltaEz = intDzDzF->Eval(radiusCorrection, phi, z1) - intDzDzF->Eval(radiusCorrection, phi, z);

        ddR = fC0 * localIntErOverEz + fC1 * localIntEPhiOverEz;
        ddRPhi = fC0 * localIntEPhiOverEz - fC1 * localIntErOverEz;
        ddZ = localIntDeltaEz * AliTPCPoissonSolver::fgkdvdE * AliTPCPoissonSolver::fgkdvdE; // two times?

        drDist += ddR;
        dzDist += ddZ;
        dRPhi += ddRPhi;

        (*mCorrDrDz)(i, j) = drDist;
        (*mCorrDPhiRDz)(i, j) = dRPhi;
        (*mCorrDz)(i, j) = dzDist;
      }
    }
  }
}

/// follow the drift for exact function
///
/// \param intDrDzF
/// \param intDPhiDzF
/// \param intDzDzF
/// \param ezField
/// \param matricesGDistDrDz
/// \param matricesGDistDPhiRDz
/// \param matricesGDistDz
/// \param matricesGCorrDrDz
/// \param matricesGCorrDPhiRDz
/// \param matricesGCorrDz
/// \param matricesGCorrIrregularDrDz
/// \param matricesGCorrIrregularDPhiRDz
/// \param matricesGCorrIrregularDz
/// \param matricesRIrregular
/// \param matricesPhiIrregular
/// \param matricesZIrregular
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
/// \param rList
/// \param phiList
/// \param zList
void AliTPCSpaceCharge3DCalc::IntegrateDistCorrDriftLineDz(
  TFormula* intDrDzF, TFormula* intDPhiDzF, TFormula* intDzDzF, TFormula* ezF, const Double_t ezField,
  TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz,
  TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz,
  TMatrixD** matricesGCorrIrregularDrDz, TMatrixD** matricesGCorrIrregularDPhiRDz, TMatrixD** matricesGCorrIrregularDz, TMatrixD** matricesRIrregular,
  TMatrixD** matricesPhiIrregular, TMatrixD** matricesZIrregular, const Int_t nRRow, const Int_t nZColumn,
  const Int_t phiSlice, const Double_t* rList, const Double_t* phiList, const Double_t* zList)
{

  Float_t drDist, dPhi, dzDist, ddR, ddRPhi, ddZ;
  Float_t radius0, phi0, z0, radius, phi, z, radiusCorrection, z1;

  Float_t localIntErOverEz = 0.0;
  Float_t localIntEPhiOverEz = 0.0;
  Float_t localIntDeltaEz = 0.0;

  radiusCorrection = 0.0;
  radius = 0.0;
  TMatrixD* mDistDrDz;
  TMatrixD* mDistDPhiRDz;
  TMatrixD* mDistDz;
  TMatrixD* mCorrDrDz;
  TMatrixD* mCorrDPhiRDz;
  TMatrixD* mCorrDz;
  TMatrixD* mCorrIrregularDrDz;
  TMatrixD* mCorrIrregularDPhiRDz;
  TMatrixD* mCorrIrregularDz;
  TMatrixD* mRIrregular;
  TMatrixD* mPhiIrregular;
  TMatrixD* mZIrregular;
  Int_t j = nZColumn - 1;
  z0 = zList[j];

  for (Int_t m = 0; m < phiSlice; m++) {
    phi0 = phiList[m];

    mDistDrDz = matricesGDistDrDz[m];
    mDistDPhiRDz = matricesGDistDPhiRDz[m];
    mDistDz = matricesGDistDz[m];

    //
    mCorrDrDz = matricesGCorrDrDz[m];
    mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
    mCorrDz = matricesGCorrDz[m];

    mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
    mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
    mCorrIrregularDz = matricesGCorrIrregularDz[m];

    mRIrregular = matricesRIrregular[m];
    mPhiIrregular = matricesPhiIrregular[m];
    mZIrregular = matricesZIrregular[m];

    for (Int_t i = 0; i < nRRow; i++) {
      // follow the drift
      radius0 = rList[i];
      phi = phi0;
      radius = radius0;
      ///
      (*mDistDrDz)(i, j) = 0.0;
      (*mDistDPhiRDz)(i, j) = 0.0;
      (*mDistDz)(i, j) = 0.0;

      //////////////// use irregular grid look up table for correction
      // set
      (*mCorrIrregularDrDz)(i, j) = 0.0;
      (*mCorrIrregularDPhiRDz)(i, j) = 0.0;
      (*mCorrIrregularDz)(i, j) = 0.0;

      // distorted point
      (*mRIrregular)(i, j) = radius0;
      (*mPhiIrregular)(i, j) = phi0;
      (*mZIrregular)(i, j) = z0;
      ///////////////
    }
  }

  // from j one column near end cap
  for (j = nZColumn - 2; j >= 0; j--) {

    z0 = zList[j];

    for (Int_t m = 0; m < phiSlice; m++) {
      phi0 = phiList[m];

      mDistDrDz = matricesGDistDrDz[m];
      mDistDPhiRDz = matricesGDistDPhiRDz[m];
      mDistDz = matricesGDistDz[m];

      //
      mCorrDrDz = matricesGCorrDrDz[m];
      mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
      mCorrDz = matricesGCorrDz[m];

      mCorrIrregularDrDz = matricesGCorrIrregularDrDz[m];
      mCorrIrregularDPhiRDz = matricesGCorrIrregularDPhiRDz[m];
      mCorrIrregularDz = matricesGCorrIrregularDz[m];

      mRIrregular = matricesRIrregular[m];
      mPhiIrregular = matricesPhiIrregular[m];
      mZIrregular = matricesZIrregular[m];

      for (Int_t i = 0; i < nRRow; i++) {
        // do from j to 0
        // follow the drift
        radius0 = rList[i];
        phi = phi0;
        radius = radius0;

        drDist = 0.0;
        dPhi = 0.0;
        dzDist = 0.0;
        ddRPhi = 0.0;

        // follow the drift line from z=j --> nZColumn - 1
        for (Int_t jj = j; jj < nZColumn; jj++) {
          // interpolation the local distortion for current position
          phi = phi0 + dPhi;
          radius = radius0 + drDist;
          z = zList[jj] + dzDist;
          z1 = z + (zList[j + 1] - zList[j]);
          // regulate phi
          while (phi < 0.0) {
            phi = TMath::TwoPi() + phi;
          }
          while (phi > TMath::TwoPi()) {
            phi = phi - TMath::TwoPi();
          }

          //lookupLocalDist->GetValue(radius, phi, z, ddR, ddRPhi, ddZ);
          localIntErOverEz = (intDrDzF->Eval(radius, phi, z1) - intDrDzF->Eval(radius, phi, z)) / (ezField + ezF->Eval(radius, phi, z));
          localIntEPhiOverEz = (intDPhiDzF->Eval(radius, phi, z1) - intDPhiDzF->Eval(radius, phi, z)) / (ezField + ezF->Eval(radius, phi, z));
          localIntDeltaEz = intDzDzF->Eval(radius, phi, z1) - intDzDzF->Eval(radius, phi, z);

          ddR = fC0 * localIntErOverEz + fC1 * localIntEPhiOverEz;
          ddRPhi = fC0 * localIntEPhiOverEz - fC1 * localIntErOverEz;
          ddZ = -1 * localIntDeltaEz * AliTPCPoissonSolver::fgkdvdE;

          drDist += ddR;
          dPhi += (ddRPhi / radius);
          dzDist += ddZ;

          // add local distortion
        }
        // set the global distortion after following the electron drift
        (*mDistDrDz)(i, j) = drDist;
        (*mDistDPhiRDz)(i, j) = dPhi * radius0;
        (*mDistDz)(i, j) = dzDist;
        /////////////// use irregular grid look up table for correction
        // set
        (*mCorrIrregularDrDz)(i, j) = -drDist;
        (*mCorrIrregularDPhiRDz)(i, j) = -dPhi * (radius0 + drDist);
        (*mCorrIrregularDz)(i, j) = -dzDist;

        // distorted point
        (*mRIrregular)(i, j) = radius0 + drDist;
        (*mPhiIrregular)(i, j) = phi0 + dPhi;
        (*mZIrregular)(i, j) = z0 + dzDist;
        ///////////////

        // put the radius to the original value
        if (j == nZColumn - 2) {
          radiusCorrection = radius0;
        }

        // get global correction from j+1
        drDist = (*mCorrDrDz)(i, j + 1);
        dzDist = (*mCorrDz)(i, j + 1);

        radiusCorrection = radius0 + drDist;
        dPhi = (*mCorrDPhiRDz)(i, j + 1) / radius0;
        //dPhi = (*mCorrDPhiRDz)(i, j + 1) /radiusCorrection;
        phi = phi0 + dPhi;
        z = zList[j + 1] + dzDist;
        z1 = z - (zList[j + 1] - zList[j]);

        while (phi < 0.0) {
          phi = TMath::TwoPi() + phi;
        }
        while (phi > TMath::TwoPi()) {
          phi = phi - TMath::TwoPi();
        }

        //lookupLocalCorr->GetValue(radiusCorrection, phi, z, ddR, ddRPhi, ddZ);
        localIntErOverEz = (intDrDzF->Eval(radiusCorrection, phi, z1) - intDrDzF->Eval(radiusCorrection, phi, z)) / (ezField + intDzDzF->Eval(radiusCorrection, phi, z));
        localIntEPhiOverEz = (intDPhiDzF->Eval(radiusCorrection, phi, z1) - intDPhiDzF->Eval(radiusCorrection, phi, z)) / (ezField + intDzDzF->Eval(radiusCorrection, phi, z));
        localIntDeltaEz = intDzDzF->Eval(radiusCorrection, phi, z1) - intDzDzF->Eval(radiusCorrection, phi, z);

        ddR = fC0 * localIntErOverEz + fC1 * localIntEPhiOverEz;
        ddRPhi = fC0 * localIntEPhiOverEz - fC1 * localIntErOverEz;
        ddZ = -1 * localIntDeltaEz * AliTPCPoissonSolver::fgkdvdE; // two times?

        drDist += ddR;
        dzDist += ddZ;
        dPhi += ddRPhi / radiusCorrection;

        (*mCorrDrDz)(i, j) = drDist;
        (*mCorrDPhiRDz)(i, j) = dPhi * radius0;
        (*mCorrDz)(i, j) = dzDist;
      }
    }
  }
}

/// See explanation at LocalDistCorrDz
///
///
/// \param matricesEr TMatrixD**  electric field for \f$r\f$ component
///	\param matricesEPhi TMatrixD** electric field for \f$\phi\f$ component
///	\param matricesEz TMatrixD** electric field for \f$z\f$ component
///	\param matricesCorrDrDz TMatrixD** local correction \f$\hat{\delta}_{r}\f$
///	\param matricesCorrDPhiRDz TMatrixD** local correction \f$r \hat{\delta}_{\phi}\f$
///	\param matricesCorrDz TMatrixD** local correction \f$ \hat{\delta}_{z}\f$
/// \param nRRow Int_t Number of nRRow in r-direction
/// \param nZColumn Int_t Number of nZColumn in z-direction
/// \param phiSlice Int_t Number of phi slices in \f$ phi \f$ direction
///	\param gridSizeZ const Float_t grid size in z direction
/// \param ezField const Double_t ezField calculate from the invoking operation
///
/// \pre matricesEr, matricesEPhi, matrices Ez are provided
/// \post Local correction are computed according simplified Langevin equation
/// ~~~
/// matricesCorrDz,matricesCorrDPhiRDz,matricesDistDz
/// ~~~
void AliTPCSpaceCharge3DCalc::IntegrateDistCorrDriftLineDzWithLookUp(AliTPCLookUpTable3DInterpolatorD* lookupLocalDist, TMatrixD** matricesGDistDrDz, TMatrixD** matricesGDistDPhiRDz, TMatrixD** matricesGDistDz, AliTPCLookUpTable3DInterpolatorD* lookupLocalCorr, TMatrixD** matricesGCorrDrDz, TMatrixD** matricesGCorrDPhiRDz, TMatrixD** matricesGCorrDz, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, Double_t* rList, Double_t* phiList, Double_t* zList)
{

  Float_t drDist, dRPhi, dzDist, ddR, ddRPhi, ddZ;
  Float_t radius0, phi0, z0, radius, phi, z, radiusCorrection;
  radiusCorrection = 0.0;
  radius = 0.0;
  TMatrixD* mDistDrDz;
  TMatrixD* mDistDPhiRDz;
  TMatrixD* mDistDz;
  TMatrixD* mCorrDrDz;
  TMatrixD* mCorrDPhiRDz;
  TMatrixD* mCorrDz;
  Int_t j = nZColumn - 1;

  // allocate look up for temporal
  AliTPCLookUpTable3DInterpolatorD* lookupGlobalDistTemp =
    new AliTPCLookUpTable3DInterpolatorD(
      nRRow, matricesGDistDrDz, rList, phiSlice, matricesGDistDPhiRDz, phiList, nZColumn, matricesGDistDz,
      zList, 2);

  z0 = zList[j];

  for (Int_t m = 0; m < phiSlice; m++) {
    phi0 = phiList[m];

    mDistDrDz = matricesGDistDrDz[m];
    mDistDPhiRDz = matricesGDistDPhiRDz[m];
    mDistDz = matricesGDistDz[m];

    //
    mCorrDrDz = matricesGCorrDrDz[m];
    mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
    mCorrDz = matricesGCorrDz[m];

    for (Int_t i = 0; i < nRRow; i++) {
      // do from j to 0
      // follow the drift
      radius0 = rList[i];
      phi = phi0;
      radius = radius0;

      drDist = 0.0;
      dRPhi = 0.0;
      dzDist = 0.0;
      ddRPhi = 0.0;

      ///
      (*mDistDrDz)(i, j) = drDist;
      (*mDistDPhiRDz)(i, j) = dRPhi;
      (*mDistDz)(i, j) = dzDist;
    }
  }

  // from j one column near end cap
  for (j = nZColumn - 2; j >= 0; j--) {

    z0 = zList[j];
    for (Int_t m = 0; m < phiSlice; m++) {
      phi0 = phiList[m];

      mDistDrDz = matricesGDistDrDz[m];
      mDistDPhiRDz = matricesGDistDPhiRDz[m];
      mDistDz = matricesGDistDz[m];

      //
      mCorrDrDz = matricesGCorrDrDz[m];
      mCorrDPhiRDz = matricesGCorrDPhiRDz[m];
      mCorrDz = matricesGCorrDz[m];

      for (Int_t i = 0; i < nRRow; i++) {

        // do from j to 0
        // follow the drift
        radius0 = rList[i];
        phi = phi0;
        radius = radius0;
        z = z0;

        lookupLocalDist->GetValue(radius, phi, z, ddR, ddRPhi, ddZ);

        phi += ddRPhi / radius;
        radius = radius0 + ddR;
        z = zList[j + 1] + ddZ;

        if (j < nZColumn - 2) {
          lookupGlobalDistTemp->GetValue(radius, phi, z, drDist, dRPhi, dzDist);
        }

        (*mDistDrDz)(i, j) = drDist + ddR;
        (*mDistDPhiRDz)(i, j) = dRPhi + ddRPhi;
        (*mDistDz)(i, j) = dzDist + ddZ;

        if (j > 0) {
          (*mDistDrDz)(i, j) = drDist + ddR;
          (*mDistDPhiRDz)(i, j) = dRPhi + ddRPhi;
          (*mDistDz)(i, j) = dzDist + ddZ;
        }
        // copy to 1D for being able to interpolate at next step

        // put the radius to the original value
        if (j == nZColumn - 2) {
          radiusCorrection = radius0;
        }

        // get global correction from j+1
        drDist = (*mCorrDrDz)(i, j + 1);
        dRPhi = (*mCorrDPhiRDz)(i, j + 1);
        dzDist = (*mCorrDz)(i, j + 1);

        radiusCorrection = radius0 + drDist;
        phi = phi0 + dRPhi / radiusCorrection;
        z = zList[j + 1] + dzDist;

        while (phi < 0.0) {
          phi = TMath::TwoPi() + phi;
        }
        while (phi > TMath::TwoPi()) {
          phi = phi - TMath::TwoPi();
        }

        lookupLocalCorr->GetValue(radiusCorrection, phi, z, ddR, ddRPhi, ddZ);

        drDist += ddR;
        dzDist += ddZ;
        dRPhi += ddRPhi;

        (*mCorrDrDz)(i, j) = drDist;
        (*mCorrDPhiRDz)(i, j) = dRPhi;
        (*mCorrDz)(i, j) = dzDist;
      }
    }

    lookupGlobalDistTemp->CopyFromMatricesToInterpolator(j);
    if (j > 0) {
      lookupGlobalDistTemp->CopyFromMatricesToInterpolator(j - 1);
    }
  }
  delete lookupGlobalDistTemp;
}

///
/// \param lookupGlobal
/// \param lookupRDz
/// \param lookupPhiRDz
/// \param lookupDz
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
/// \param rList
/// \param phiList
/// \param zList
void AliTPCSpaceCharge3DCalc::FillLookUpTable(AliTPCLookUpTable3DInterpolatorD* lookupGlobal, TMatrixD** lookupRDz,
                                              TMatrixD** lookupPhiRDz, TMatrixD** lookupDz, const Int_t nRRow,
                                              const Int_t nZColumn, const Int_t phiSlice, const Double_t* rList,
                                              const Double_t* phiList, const Double_t* zList)
{
  Double_t r, phi, z;
  TMatrixD* mR;
  TMatrixD* mPhiR;
  TMatrixD* mDz;

  /// * Interpolate basicLookup tables; once for each rod, then sum the results
  for (Int_t k = 0; k < fNPhiSlices; k++) {
    phi = fListPhi[k];

    mR = lookupRDz[k];
    mPhiR = lookupPhiRDz[k];
    mDz = lookupDz[k];
    for (Int_t j = 0; j < fNZColumns; j++) {
      z = fListZ[j]; // Symmetric solution in Z that depends only on ABS(Z)

      for (Int_t i = 0; i < fNRRows; i++) {
        r = fListR[i];

        lookupGlobal->GetValue(r, phi, z, (*mR)(i, j), (*mPhiR)(i, j), (*mDz)(i, j));
      }
    }
  }
}
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetDistortionCyl", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  GetDistortionCylAC(x, roc, dx);
}
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetDistortionCylAC(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetDistortionCylAC", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < -1e-16) || (sign == -1 && z > -1e-16)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::GetDistortionCylAC", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    fLookupIntDistA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupIntDistC->GetValue(r, phi, -1 * z, dR, dRPhi, dZ);
    dZ = -1 * dZ;
  }

  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}
/// Get Correction from irregular table
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrectionCylACIrregular(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetCorrectionCylACIrregular", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Double_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNRRows - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZColumns - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhiSlices;

  r = x[0];
  phi = x[1];
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceChargeCalc3D::GetCorrectionCylACIrregular", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  // get distortion from irregular table

  if (z > 0) {
    fLookupIntCorrIrregularA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupIntCorrIrregularC->GetValue(r, phi, -z, dR, dRPhi, dZ);
    dZ = -1 * dZ;
  }

  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}

/// Get Correction from irregular table
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrectionCylACIrregular(const Float_t x[], Short_t roc, Float_t dx[], const Int_t side)
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetCorrectionCylACIrregular", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Double_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNRRows - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZColumns - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhiSlices;

  r = x[0];
  phi = x[1];
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceChargeCalc3D::GetCorrectionCylACIrregular", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  // get distortion from irregular table

  if (side == 0) {
    fLookupIntCorrIrregularA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupIntCorrIrregularC->GetValue(r, phi, -z, dR, dRPhi, dZ);
    dZ = -1 * dZ;
  }

  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}

/// Get correction regular grid by following electron
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrectionCylAC(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetDistortionCylAC", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < -1e-16) || (sign == -1 && z > -1e-16)) { // just a consistency check
    Error("AliTPCSpaceChargeCalc3D::GetCorrectionCylAC", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    fLookupIntCorrA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupIntCorrC->GetValue(r, phi, -z, dR, dRPhi, dZ);
    dZ = -1 * dZ;
  }
  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}
void AliTPCSpaceCharge3DCalc::GetDistortion(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetDistortion", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t pCyl[3]; // a point in cylindrical coordinate
  Float_t dCyl[3]; // distortion

  pCyl[0] = TMath::Sqrt(x[0] * x[0] + x[1] * x[1]);
  pCyl[1] = TMath::ATan2(x[1], x[0]);

  // normalize phi
  while (pCyl[1] > TMath::Pi()) {
    pCyl[1] -= TMath::TwoPi();
  }
  while (pCyl[1] < -TMath::Pi()) {
    pCyl[1] += TMath::TwoPi();
  }

  pCyl[2] = x[2]; // Create temporary copy of x[2]

  GetDistortionCylAC(pCyl, roc, dCyl);

  // Calculate distorted position
  if (pCyl[0] > 0.0) {
    //pCyl[0] = pCyl[0] + fCorrectionFactor * dCyl[0];
    pCyl[1] = pCyl[1] + fCorrectionFactor * dCyl[1] / pCyl[0];
    pCyl[0] = pCyl[0] + fCorrectionFactor * dCyl[0];
  }

  dCyl[2] = fCorrectionFactor * dCyl[2];

  // distortion in x,y and z
  dx[0] = (pCyl[0] * TMath::Cos(pCyl[1]) - x[0]);
  dx[1] = (pCyl[0] * TMath::Sin(pCyl[1]) - x[1]);
  dx[2] = dCyl[2];
}
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrectionCyl(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetCorrectionCyl", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }
  if (fCorrectionType == kRegularInterpolator) {
    GetCorrectionCylAC(x, roc, dx);
  } else {
    GetCorrectionCylACIrregular(x, roc, dx);
  }
}
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrection(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetCorrection", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t pCyl[3]; // a point in cylindrical coordinate
  Float_t dCyl[3]; // distortion

  pCyl[0] = TMath::Sqrt(x[0] * x[0] + x[1] * x[1]);
  pCyl[1] = TMath::ATan2(x[1], x[0]);
  pCyl[2] = x[2]; // Create temporary copy of x[2]

  if (fCorrectionType == kRegularInterpolator) {
    while (pCyl[1] > TMath::Pi()) {
      pCyl[1] -= TMath::TwoPi();
    }
    while (pCyl[1] < -TMath::Pi()) {
      pCyl[1] += TMath::TwoPi();
    }

    GetCorrectionCylAC(pCyl, roc, dCyl);
  } else {
    GetCorrectionCylACIrregular(pCyl, roc, dCyl);
  }

  // Calculate distorted position
  if (pCyl[0] > 0.0) {
    //pCyl[0] = pCyl[0] + fCorrectionFactor * dCyl[0];
    pCyl[1] = pCyl[1] + fCorrectionFactor * dCyl[1] / pCyl[0];
    pCyl[0] = pCyl[0] + fCorrectionFactor * dCyl[0];
  }

  dCyl[2] = fCorrectionFactor * dCyl[2];

  // distortion in x,y and z
  dx[0] = (pCyl[0] * TMath::Cos(pCyl[1]) - x[0]);
  dx[1] = (pCyl[0] * TMath::Sin(pCyl[1]) - x[1]);
  dx[2] = dCyl[2];
}

///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrection(const Float_t x[], Short_t roc, Float_t dx[], const Int_t side)
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::GetCorrection", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t pCyl[3]; // a point in cylindrical coordinate
  Float_t dCyl[3]; // distortion

  pCyl[0] = TMath::Sqrt(x[0] * x[0] + x[1] * x[1]);
  pCyl[1] = TMath::ATan2(x[1], x[0]);
  pCyl[2] = x[2]; // Create temporary copy of x[2]

  if (fCorrectionType == kRegularInterpolator) {
    while (pCyl[1] > TMath::Pi()) {
      pCyl[1] -= TMath::TwoPi();
    }
    while (pCyl[1] < -TMath::Pi()) {
      pCyl[1] += TMath::TwoPi();
    }

    GetCorrectionCylAC(pCyl, roc, dCyl);
  } else {
    GetCorrectionCylACIrregular(pCyl, roc, dCyl, side);
  }

  // Calculate distorted position
  if (pCyl[0] > 0.0) {
    pCyl[0] = pCyl[0] + fCorrectionFactor * dCyl[0];
    pCyl[1] = pCyl[1] + fCorrectionFactor * dCyl[1] / pCyl[0];
  }

  dCyl[2] = fCorrectionFactor * dCyl[2];

  // distortion in x,y and z
  dx[0] = (pCyl[0] * TMath::Cos(pCyl[1]) - x[0]);
  dx[1] = (pCyl[0] * TMath::Sin(pCyl[1]) - x[1]);
  dx[2] = dCyl[2];
}

/// Use 3D space charge map as an optional input
/// The layout of the input histogram is assumed to be: (phi,r,z)
/// Density histogram is  expected to bin in  C/m^3
///
/// Standard histogram interpolation is used in order to use the density at center of bin
///
/// \param hisSpaceCharge3D
/// \param norm
void AliTPCSpaceCharge3DCalc::SetInputSpaceCharge(TH3* hisSpaceCharge3D, Double_t norm)
{
  fHistogram3DSpaceCharge = hisSpaceCharge3D;
  fInitLookUp = kFALSE;

  Info("AliTPCSpaceCharge3DCalc:SetInputSpaceCharge", "Set Input Space Charge by 3D");
  Double_t rMin = hisSpaceCharge3D->GetYaxis()->GetBinCenter(0);
  Double_t rMax = hisSpaceCharge3D->GetYaxis()->GetBinUpEdge(hisSpaceCharge3D->GetYaxis()->GetNbins());
  Double_t zMin = hisSpaceCharge3D->GetZaxis()->GetBinCenter(0);
  Double_t zMax = hisSpaceCharge3D->GetZaxis()->GetBinCenter(hisSpaceCharge3D->GetZaxis()->GetNbins());

  Double_t radius0, z0, phi0;
  TMatrixD* charge;
  for (Int_t iSide = 0; iSide < 2; iSide++) {
    for (Int_t k = 0; k < fNPhiSlices; k++) {
      if (iSide == 0) {
        charge = fMatrixChargeA[k];
      } else {
        charge = fMatrixChargeC[k];
      }

      phi0 = fListPhi[k];

      for (Int_t i = 0; i < fNRRows; i++) {
        radius0 = fListR[i];

        for (Int_t j = 0; j < fNZColumns; j++) {
          z0 = fListZ[j];

          if (radius0 > rMin && radius0 < rMax && z0 > zMin && z0 < zMax) {

            (*charge)(i, j) = norm * InterpolatePhi(hisSpaceCharge3D, phi0, radius0, z0);
          }
          //}
        } // end j
      }   // end i
    }     // end phi
  }

  fInterpolatorChargeA->SetValue(fMatrixChargeA);
  if (fInterpolationOrder > 2) {
    fInterpolatorChargeA->InitCubicSpline();
  }
  fInterpolatorChargeC->SetValue(fMatrixChargeC);
  if (fInterpolationOrder > 2) {
    fInterpolatorChargeC->InitCubicSpline();
  }
}

/// SetInputCharge
///
/// \param hisSpaceCharge3D TH3* histogram for space charge
/// \param norm Double_t norm/weight
/// \param side Int_t side = 0 => side A, side = 1 => side C
///
/// side effects: create Charge interpolator
void AliTPCSpaceCharge3DCalc::SetInputSpaceCharge(TH3* hisSpaceCharge3D, Double_t norm, Int_t side)
{
  if (side == 0) {
    fHistogram3DSpaceChargeA = hisSpaceCharge3D;
  } else {
    fHistogram3DSpaceChargeC = hisSpaceCharge3D;
  }

  Double_t rMin = hisSpaceCharge3D->GetYaxis()->GetBinCenter(0);
  Double_t rMax = hisSpaceCharge3D->GetYaxis()->GetBinUpEdge(hisSpaceCharge3D->GetYaxis()->GetNbins());
  Double_t zMin = hisSpaceCharge3D->GetZaxis()->GetBinCenter(0);
  Double_t zMax = hisSpaceCharge3D->GetZaxis()->GetBinCenter(hisSpaceCharge3D->GetZaxis()->GetNbins());
  Double_t radius0, z0, phi0;
  TMatrixD* charge;

  for (Int_t k = 0; k < fNPhiSlices; k++) {
    if (side == 0) {
      charge = fMatrixChargeA[k];
    } else {
      charge = fMatrixChargeC[k];
    }

    phi0 = fListPhi[k];
    for (Int_t i = 0; i < fNRRows; i++) {
      radius0 = fListR[i];
      for (Int_t j = 0; j < fNZColumns; j++) {
        z0 = fListZ[j];

        if (radius0 > rMin && radius0 < rMax && z0 > zMin && z0 < zMax) {
          (*charge)(i, j) = norm * InterpolatePhi(hisSpaceCharge3D, phi0, radius0, z0);
        }
      } // end j
    }   // end i
  }     // end phi

  if (side == 0) {
    fInterpolatorChargeA->SetValue(fMatrixChargeA);
    fInterpolatorChargeA->InitCubicSpline();
  } else {
    fInterpolatorChargeC->SetValue(fMatrixChargeC);
    fInterpolatorChargeC->InitCubicSpline();
  }

  fInitLookUp = kFALSE;
}

/// InterpolationPhi is only used for reading from TH3F (since it is not cylindrical)
///
/// \param r
/// \param z
/// \return
Double_t AliTPCSpaceCharge3DCalc::InterpolatePhi(TH3* h3, const Double_t phi, const Double_t r, const Double_t z)
{

  Int_t ubx = h3->GetXaxis()->FindBin(phi);
  if (phi < h3->GetXaxis()->GetBinCenter(ubx)) {
    ubx -= 1;
  }
  Int_t obx = ubx + 1;
  Int_t uby = h3->GetYaxis()->FindBin(r);
  if (r < h3->GetYaxis()->GetBinCenter(uby)) {
    uby -= 1;
  }
  Int_t oby = uby + 1;
  Int_t ubz = h3->GetZaxis()->FindBin(z);
  if (z < h3->GetZaxis()->GetBinCenter(ubz)) {
    ubz -= 1;
  }
  Int_t obz = ubz + 1;

  if (uby <= 0 || ubz <= 0 ||
      oby > h3->GetYaxis()->GetNbins() || obz > h3->GetZaxis()->GetNbins()) {
    return 0;
  }

  if (ubx <= 0) {
    ubx = h3->GetXaxis()->GetNbins();
  }

  if (obx > h3->GetXaxis()->GetNbins()) {
    obx = 1;
  }

  Double_t xw = h3->GetXaxis()->GetBinCenter(obx) - h3->GetXaxis()->GetBinCenter(ubx);
  Double_t yw = h3->GetYaxis()->GetBinCenter(oby) - h3->GetYaxis()->GetBinCenter(uby);
  Double_t zw = h3->GetZaxis()->GetBinCenter(obz) - h3->GetZaxis()->GetBinCenter(ubz);
  Double_t xd = (phi - h3->GetXaxis()->GetBinCenter(ubx)) / xw;
  Double_t yd = (r - h3->GetYaxis()->GetBinCenter(uby)) / yw;
  Double_t zd = (z - h3->GetZaxis()->GetBinCenter(ubz)) / zw;
  Double_t v[] = {h3->GetBinContent(ubx, uby, ubz), h3->GetBinContent(ubx, uby, obz),
                  h3->GetBinContent(ubx, oby, ubz), h3->GetBinContent(ubx, oby, obz),
                  h3->GetBinContent(obx, uby, ubz), h3->GetBinContent(obx, uby, obz),
                  h3->GetBinContent(obx, oby, ubz), h3->GetBinContent(obx, oby, obz)};
  Double_t i1 = v[0] * (1 - zd) + v[1] * zd;
  Double_t i2 = v[2] * (1 - zd) + v[3] * zd;
  Double_t j1 = v[4] * (1 - zd) + v[5] * zd;
  Double_t j2 = v[6] * (1 - zd) + v[7] * zd;
  Double_t w1 = i1 * (1 - yd) + i2 * yd;
  Double_t w2 = j1 * (1 - yd) + j2 * yd;
  Double_t result = w1 * (1 - xd) + w2 * xd;
  return result;
}
/// returns the (input) space charge density at a given point according
/// Note: input in [cm], output in [C/m^3/e0] !!
Float_t AliTPCSpaceCharge3DCalc::GetSpaceChargeDensity(Float_t r, Float_t phi, Float_t z)
{
  while (phi < 0) {
    phi += TMath::TwoPi();
  }
  while (phi > TMath::TwoPi()) {
    phi -= TMath::TwoPi();
  }

  const Int_t order = 1; //

  const Float_t x[] = {r, phi, z};
  Float_t sc = 0;
  if (z > -1e-16) {
    sc = GetChargeCylAC(x, 0);
  } else {
    sc = GetChargeCylAC(x, 18);
  }

  return sc;
}
/// returns the (input) space charge density at a given point according
/// Note: input in [cm], output in [C/m^3/e0] !!
Float_t AliTPCSpaceCharge3DCalc::GetPotential(Float_t r, Float_t phi, Float_t z)
{
  while (phi < 0) {
    phi += TMath::TwoPi();
  }
  while (phi > TMath::TwoPi()) {
    phi -= TMath::TwoPi();
  }

  const Int_t order = 1; //

  const Float_t x[] = {r, phi, z};
  Float_t v = 0;
  if (z > -1e-16) {
    v = GetPotentialCylAC(x, 0);
  } else {
    v = GetPotentialCylAC(x, 18);
  }

  return v;
}
///
///
/// \param matricesDistDrDz TMatrixD **  matrix of global distortion drDist (r direction)
/// \param matricesDistDPhiRDz TMatrixD **  matrix of global distortion dRPhi (phi r direction)
/// \param matricesDistDz TMatrixD **  matrix of global distortion dzDist (z direction)
/// \param rList Double_t * points of r in the grid (ascending mode)
/// \param zList Double_t * points of z in the grid (ascending mode)
/// \param phiList Double_t * points of phi in the grid (ascending mode)
/// \param nRRow Int_t number of grid in r direction
/// \param nZColumn Int_t number of grid in z direction
/// \param phiSlice Int_t number of grid in phi direction
///	\param nStep Int_t number of step to calculate local dist
///
void AliTPCSpaceCharge3DCalc::InverseGlobalToLocalDistortionGlobalInvTable(
  TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz, Double_t* rList,
  Double_t* zList, Double_t* phiList, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
  const Int_t nStep, const Bool_t useCylAC, Int_t stepR, Int_t stepZ, Int_t stepPhi, Int_t type)
{
  Double_t z, phi, r, zAfter, zPrevious, ddR, ddRPhi, ddZ, zl, drDist, dRPhi, dzDist, ddPhi, dPhi, deltaZ, r0, z0, phi0;
  Float_t x[3], dx[3], pdx[3];
  Int_t roc;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  TMatrixD* distDrDz;
  TMatrixD* distDPhiRDz;
  TMatrixD* distDz;

  // correction build up for inverse flow
  TMatrixD* corrDrDz;
  TMatrixD* corrDPhiRDz;
  TMatrixD* corrDz;
  TMatrixD* listR;
  TMatrixD* listPhi;
  TMatrixD* listZ;
  TMatrixD* matricesCorrDrDz[phiSlice];
  TMatrixD* matricesCorrDPhiRDz[phiSlice];
  TMatrixD* matricesCorrDz[phiSlice];
  TMatrixD* matricesRList[phiSlice];
  TMatrixD* matricesPhiList[phiSlice];
  TMatrixD* matricesZList[phiSlice];

  for (Int_t m = 0; m < phiSlice; m++) {
    matricesCorrDrDz[m] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDPhiRDz[m] = new TMatrixD(nRRow, nZColumn);
    matricesCorrDz[m] = new TMatrixD(nRRow, nZColumn);

    matricesRList[m] = new TMatrixD(nRRow, nZColumn);
    matricesPhiList[m] = new TMatrixD(nRRow, nZColumn);
    matricesZList[m] = new TMatrixD(nRRow, nZColumn);
  }

  AliTPCLookUpTable3DInterpolatorIrregularD* lookupInverseCorr = new AliTPCLookUpTable3DInterpolatorIrregularD(
    nRRow, matricesCorrDrDz, matricesRList, phiSlice, matricesCorrDPhiRDz,
    matricesPhiList, nZColumn, matricesCorrDz, matricesZList, 2,
    stepR, stepZ, stepPhi, type);

  lookupInverseCorr->SetKernelType(GetRBFKernelType());

  for (Int_t k = 0; k < phiSlice; k++) {
    distDrDz = matricesDistDrDz[k];
    distDPhiRDz = matricesDistDPhiRDz[k];
    distDz = matricesDistDz[k];

    listR = matricesRList[k];
    listPhi = matricesPhiList[k];
    listZ = matricesZList[k];

    for (Int_t i = 0; i < nRRow; i++) {
      (*distDrDz)(i, nZColumn - 1) = 0.0;
      (*distDPhiRDz)(i, nZColumn - 1) = 0.0;
      (*distDz)(i, nZColumn - 1) = 0.0;

      for (Int_t j = 0; j < nZColumn; j++) {
        (*listR)(i, j) = rList[i];
        (*listPhi)(i, j) = phiList[k];
        (*listZ)(i, j) = zList[j];
      }
    }
  }

  // 1) create global correction
  deltaZ = (zList[1] - zList[0]);
  Int_t iAnchor, kAnchor, zAnchor;

  for (Int_t j = nZColumn - 2; j >= 0; j--) {

    roc = 0; // FIXME
    for (Int_t k = 0; k < phiSlice; k++) {

      corrDrDz = matricesCorrDrDz[k];
      corrDPhiRDz = matricesCorrDPhiRDz[k];
      corrDz = matricesCorrDz[k];

      listR = matricesRList[k];
      listPhi = matricesPhiList[k];
      listZ = matricesZList[k];

      for (Int_t i = 0; i < nRRow; i++) {
        // get global distortion

        r = rList[i];
        phi = phiList[k];
        z = zList[j];

        drDist = 0.0;
        dzDist = 0.0;
        dRPhi = 0.0;

        x[0] = r;
        x[1] = phi;
        x[2] = z;

        if (useCylAC == kTRUE) {
          GetDistortionCylAC(x, roc, dx);
        } else {
          GetDistortionCyl(x, roc, dx);
        }

        drDist = dx[0];
        dzDist = dx[2];
        dRPhi = dx[1];

        r = rList[i];
        phi = phiList[k];
        z = zList[j];

        (*corrDrDz)(i, j + 1) = -drDist;
        (*corrDz)(i, j + 1) = -dzDist;
        (*corrDPhiRDz)(i, j + 1) = -dRPhi;

        (*listR)(i, j + 1) = r + drDist;
        (*listPhi)(i, j + 1) = phi + dRPhi / r;
        (*listZ)(i, j + 1) = z + dzDist;
      }
    }
    lookupInverseCorr->CopyFromMatricesToInterpolator(j + 1);
  }
  // 2) calculate local distortion
  for (Int_t j = nZColumn - 2; j >= 0; j--) {
    roc = 0; // FIXME
    for (Int_t k = 0; k < phiSlice; k++) {
      distDrDz = matricesDistDrDz[k];
      distDPhiRDz = matricesDistDPhiRDz[k];
      distDz = matricesDistDz[k];
      for (Int_t i = 0; i < nRRow; i++) {
        // get global distortion
        r = rList[i];
        phi = phiList[k];
        z = zList[j];
        drDist = 0.0;
        dzDist = 0.0;
        dRPhi = 0.0;

        if (j < nZColumn - 2) {
          // get global distortion of this point
          x[0] = r;
          x[1] = phi;
          x[2] = z;
          if (useCylAC == kTRUE) {
            GetDistortionCylAC(x, roc, dx);
          } else {
            GetDistortionCyl(x, roc, dx);
          }

          r0 = r + dx[0];
          z0 = zList[j + 1] + dx[2];
          phi0 = phi + (dx[1] / r);
          iAnchor = TMath::FloorNint((r0 - AliTPCPoissonSolver::fgkIFCRadius) / gridSizeR);
          kAnchor = TMath::FloorNint(phi0 / gridSizePhi);
          zAnchor = TMath::FloorNint(z0 / gridSizeZ);

          if (j > nZColumn - (GetIrregularGridSize() + 2)) {
            lookupInverseCorr->GetValue(r0, phi0, z0, drDist, dRPhi, dzDist, iAnchor, kAnchor, zAnchor,
                                        nRRow / 4 + 1, phiSlice / 4 + 1, 1, 0);
          } else {
            lookupInverseCorr->GetValue(r0, phi0, z0, drDist, dRPhi, dzDist, iAnchor, kAnchor, zAnchor,
                                        nRRow / 4 + 1, phiSlice / 4 + 1, GetIrregularGridSize(), 0);
          }

          phi0 = phi0 + ((dRPhi) / r0);
          r0 = r0 + (drDist);
          z0 += dzDist;

          x[0] = r0;
          x[1] = phi0;
          x[2] = z0;

          if (phi0 < 0) {
            phi0 = TMath::TwoPi() + phi0;
          }
          if (phi0 > TMath::TwoPi()) {
            phi0 = phi0 - TMath::TwoPi();
          }

          if (useCylAC == kTRUE) {
            GetDistortionCylAC(x, roc, pdx);
          } else {
            GetDistortionCyl(x, roc, pdx);
          }

          drDist = (dx[0] - pdx[0]);
          dzDist = (dx[2] - pdx[2]);
          dRPhi = (dx[1] - pdx[1]);

        } else if (j == (nZColumn - 2)) {

          x[0] = r;
          x[1] = phi;
          x[2] = zList[j];
          if (useCylAC == kTRUE) {
            GetDistortionCylAC(x, roc, dx);
          } else {
            GetDistortionCyl(x, roc, dx);
          }

          x[2] = zList[j + 1];
          if (useCylAC == kTRUE) {
            GetDistortionCylAC(x, roc, pdx);
          } else {
            GetDistortionCyl(x, roc, pdx);
          }
          drDist = (dx[0] - pdx[0]);
          dzDist = (dx[2] - pdx[2]);
          dRPhi = (dx[1] - pdx[1]);
        }

        (*distDrDz)(i, j) = drDist;
        (*distDz)(i, j) = dzDist;
        (*distDPhiRDz)(i, j) = dRPhi;
      }
    }
  }

  for (Int_t m = 0; m < phiSlice; m++) {
    delete matricesCorrDrDz[m];
    delete matricesCorrDPhiRDz[m];
    delete matricesCorrDz[m];
    delete matricesRList[m];
    delete matricesPhiList[m];
    delete matricesZList[m];
  }
  delete lookupInverseCorr;
}
///
/// \param matricesEr
/// \param matricesEPhi
/// \param matricesEz
/// \param matricesInvLocalIntErDz
/// \param matricesInvLocalIntEPhiDz
/// \param matricesInvLocalIntEz
/// \param matricesDistDrDz
/// \param matricesDistDPhiRDz
/// \param matricesDistDz
/// \param rList
/// \param zList
/// \param phiList
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
void AliTPCSpaceCharge3DCalc::InverseLocalDistortionToElectricField(
  TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
  TMatrixD** matricesInvLocalIntErDz, TMatrixD** matricesInvLocalIntEPhiDz,
  TMatrixD** matricesInvLocalIntEz, TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz,
  TMatrixD** matricesDistDz, Double_t* rList, Double_t* zList, Double_t* phiList,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice)
{
  // calculate integral
  Float_t localIntErOverEz, localIntEPhiOverEz, localIntDeltaEz, z2;
  Double_t r;
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;

  TMatrixD* distDrDz;
  TMatrixD* distDz;
  TMatrixD* distDPhiRDz;
  TMatrixD* tDistDz;
  TMatrixD* tDistDPhiRDz;
  TMatrixD* tDistDrDz;
  Float_t c02c12 = fC0 * fC0 + fC1 * fC1;

  // solve local integration
  for (Int_t j = 0; j < nZColumn; j++) {
    for (Int_t k = 0; k < phiSlice; k++) {
      distDrDz = matricesDistDrDz[k];
      distDz = matricesDistDz[k];
      distDPhiRDz = matricesDistDPhiRDz[k];

      tDistDrDz = matricesInvLocalIntErDz[k];
      tDistDz = matricesInvLocalIntEz[k];
      tDistDPhiRDz = matricesInvLocalIntEPhiDz[k];
      for (Int_t i = 0; i < nRRow; i++) {
        localIntErOverEz = fC0 * (*distDrDz)(i, j) - fC1 * (*distDPhiRDz)(i, j);
        localIntErOverEz = localIntErOverEz / (fC0 * fC0 + fC1 * fC1);
        localIntEPhiOverEz = ((*distDrDz)(i, j) - (fC0 * localIntErOverEz)) / fC1;
        localIntDeltaEz = -1 * (*distDz)(i, j) / AliTPCPoissonSolver::fgkdvdE; // two times?
        (*tDistDrDz)(i, j) = localIntErOverEz;
        (*tDistDPhiRDz)(i, j) = localIntEPhiOverEz;
        (*tDistDz)(i, j) = localIntDeltaEz;
      }
    }
  }
  TMatrixD* mEPhi;
  TMatrixD* mEr;
  TMatrixD* mEz;

  // use central-backward-forward difference for calculating Electric field component
  for (Int_t m = 0; m < phiSlice; m++) {
    mEPhi = matricesEPhi[m];
    mEr = matricesEr[m];
    mEz = matricesEz[m];
    distDrDz = matricesInvLocalIntErDz[m];
    distDPhiRDz = matricesInvLocalIntEPhiDz[m];
    distDz = matricesInvLocalIntEz[m];
    for (Int_t i = 0; i < nRRow; i++) {
      (*mEr)(i, 0) = ((*distDrDz)(i, 0) / gridSizeZ) * -1 * ezField;
      (*mEPhi)(i, 0) = ((*distDPhiRDz)(i, 0) / gridSizeZ) * -1 * ezField;
      (*mEz)(i, 0) = ((*distDz)(i, 0) / gridSizeZ);
      (*mEr)(i, nZColumn - 1) =
        ((-0.5 * (*distDrDz)(i, nZColumn - 3) + 1.5 * (*distDrDz)(i, nZColumn - 2)) / gridSizeZ) * ezField;
      (*mEPhi)(i, nZColumn - 1) =
        ((-0.5 * (*distDPhiRDz)(i, nZColumn - 3) + 1.5 * (*distDPhiRDz)(i, nZColumn - 2)) / gridSizeZ) *
        ezField;
      (*mEz)(i, nZColumn - 1) =
        (-0.5 * (*distDz)(i, nZColumn - 3) + 1.5 * (*distDz)(i, nZColumn - 2)) / gridSizeZ;
    }

    for (Int_t i = 0; i < nRRow; i++) {
      for (Int_t j = 1; j < nZColumn - 1; j++) {
        (*mEr)(i, j) = (((*distDrDz)(i, j) + (*distDrDz)(i, j - 1)) / (2 * gridSizeZ)) *
                       ezField; // z direction
        (*mEPhi)(i, j) = (((*distDPhiRDz)(i, j) + (*distDPhiRDz)(i, j - 1)) / (2 * gridSizeZ)) *
                         ezField;                                                 // z direction
        (*mEz)(i, j) = ((*distDz)(i, j) + (*distDz)(i, j - 1)) / (2 * gridSizeZ); // z direction
      }
    }
  }
}
/// Inverse Electric Field to Charge
/// using partial differential
///
/// \param matricesCharge
/// \param matricesEr
/// \param matricesEPhi
/// \param matricesEz
/// \param rList
/// \param zList
/// \param phiList
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
void AliTPCSpaceCharge3DCalc::InverseElectricFieldToCharge(
  TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
  Double_t* rList, Double_t* zList, Double_t* phiList, const Int_t nRRow,
  const Int_t nZColumn, const Int_t phiSlice)
{

  Float_t radius;
  Double_t drDist, dzDist, dPhi;
  Int_t mPlus, mMinus, mPlus2, mMinus2, signPlus, signMinus;
  Int_t symmetry = 0;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;

  for (Int_t m = 0; m < phiSlice; m++) {
    mPlus = m + 1;
    signPlus = 1;
    mMinus = m - 1;
    signMinus = 1;
    mPlus2 = m + 2;
    mMinus2 = m - 2;
    if (symmetry == 1) { // Reflection symmetry in phi (e.g. symmetry at sector boundaries, or half sectors, etc.)
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
      }
      if (mMinus < 0) {
        mMinus = 1;
      }
    } else if (symmetry == -1) { // Anti-symmetry in phi
      if (mPlus > phiSlice - 1) {
        mPlus = phiSlice - 2;
        signPlus = -1;
      }
      if (mMinus < 0) {
        mMinus = 1;
        signMinus = -1;
      }
    } else { // No Symmetries in phi, no boundaries, the calculations is continuous across all phi
      if (mPlus > phiSlice - 1) {
        mPlus = m + 1 - phiSlice;
      }
      if (mMinus < 0) {
        mMinus = m - 1 + phiSlice;
      }
      if (mPlus2 > phiSlice - 1) {
        mPlus2 = m + 2 - phiSlice;
      }
      if (mMinus2 < 0) {
        mMinus2 = m - 2 + phiSlice;
      }
    }

    TMatrixD& matrixCharge = *matricesCharge[m];
    TMatrixD& matrixEr = *matricesEr[m];
    TMatrixD& matrixEz = *matricesEz[m];
    TMatrixD& matrixEPhi = *matricesEPhi[m];
    TMatrixD& matrixEPhiM = *matricesEPhi[mMinus];
    TMatrixD& matrixEPhiP = *matricesEPhi[mPlus];
    TMatrixD& matrixEPhiM2 = *matricesEPhi[mMinus2];
    TMatrixD& matrixEPhiP2 = *matricesEPhi[mPlus2];

    // for non-boundary V
    for (Int_t i = 2; i < nRRow - 2; i++) {
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
      for (Int_t j = 2; j < nZColumn - 2; j++) {
        drDist = (-matrixEr(i + 2, j) + 8 * matrixEr(i + 1, j) - 8 * matrixEr(i - 1, j) + matrixEr(i - 2, j)) /
                 (12 * gridSizeR); // r direction
        dzDist = (-matrixEz(i, j + 2) + 8 * matrixEz(i, j + 1) - 8 * matrixEz(i, j - 1) + matrixEz(i, j - 2)) /
                 (12 * gridSizeZ); // r direction
        dPhi = (-matrixEPhiP2(i, j) + 8 * matrixEPhiP(i, j) - 8 * matrixEPhiM(i, j) + matrixEPhiM2(i, j)) /
               (12 * gridSizePhi); // phi

        matrixCharge(i, j) = -1 * (matrixEr(i, j) / radius + drDist + dPhi / radius + dzDist);
      }
    }

    // for boundary in r
    for (Int_t j = 2; j < nZColumn - 2; j++) {

      // r near inner radius
      // for index r[0]
      radius = AliTPCPoissonSolver::fgkIFCRadius;
      drDist = (-(11.0 / 6.0) * matrixEr(0, j) + (3.0 * matrixEr(1, j)) - (1.5 * matrixEr(2, j)) +
                ((1.0 / 3.0) * matrixEr(3, j))) /
               gridSizeR; // forward difference

      //	drDist  =  ( -(1.5)*matrixEr(0,j) + (2.0*matrixEr(1,j)) - (0.5*matrixEr(2,j)) )  / gridSizeR;

      dzDist = (-matrixEz(0, j + 2) + 8 * matrixEz(0, j + 1) - 8 * matrixEz(0, j - 1) + matrixEz(0, j - 2)) /
               (12.0 * gridSizeZ); // z direction
      dPhi = (-matrixEPhiP2(0, j) + 8 * matrixEPhiP(0, j) - 8 * matrixEPhiM(0, j) + matrixEPhiM2(0, j)) /
             (12.0 * gridSizePhi);

      matrixCharge(0, j) = -1 * (matrixEr(0, j) / radius + drDist + dPhi / radius + dzDist);

      // index use central difference 3-point center
      radius = AliTPCPoissonSolver::fgkIFCRadius + gridSizeR;
      //	drDist  =  (-matrixEr(3,j)  +6.0*matrixEr(2,j) - 3.0*matrixEr(1,j) - 2*matrixEr(0,j) ) / (6.0*gridSizeR) ; // forward difference
      drDist = (matrixEr(2, j) - matrixEr(0, j)) / (2.0 * gridSizeR);

      dzDist = (-matrixEz(1, j + 2) + 8 * matrixEz(1, j + 1) - 8 * matrixEz(1, j - 1) + matrixEz(1, j - 2)) /
               (12 * gridSizeZ); // z direction
      dPhi = (-matrixEPhiP2(1, j) + 8 * matrixEPhiP(1, j) - 8 * matrixEPhiM(1, j) + matrixEPhiM2(1, j)) /
             (12 * gridSizePhi);
      matrixCharge(1, j) = -1 * (matrixEr(1, j) / radius + drDist + dPhi / radius + dzDist);

      // index use central difference 3-point center
      radius = AliTPCPoissonSolver::fgkIFCRadius + (nRRow - 2) * gridSizeR;
      //	drDist =   (2.0 * matrixEr(nRRow - 1,j)  + 3.0*matrixEr(nRRow - 2,j) - 6.0*matrixEr(nRRow -3,j) + matrixEr(nRRow-4,j) ) / (6.0*gridSizeR) ;
      drDist = (matrixEr(nRRow - 1, j) - matrixEr(nRRow - 3, j)) / (2.0 * gridSizeR);

      dzDist = (-matrixEz(nRRow - 2, j + 2) + 8 * matrixEz(nRRow - 2, j + 1) - 8 * matrixEz(nRRow - 2, j - 1) +
                matrixEz(nRRow - 2, j - 2)) /
               (12 * gridSizeZ);
      dPhi = (-matrixEPhiP2(nRRow - 2, j) + 8 * matrixEPhiP(nRRow - 2, j) - 8 * matrixEPhiM(nRRow - 2, j) +
              matrixEPhiM2(nRRow - 2, j)) /
             (12.0 * gridSizePhi);
      matrixCharge(nRRow - 2, j) = -1 * (matrixEr(nRRow - 2, j) / radius + drDist + dPhi / radius + dzDist);

      // index r[nRRow -1] backward difference
      radius = AliTPCPoissonSolver::fgkIFCRadius + (nRRow - 1) * gridSizeR;
      //drDist =  ( 1.5*matrixEr(nRRow-1,j) - 2.0*matrixEr(nRRow-2,j) + 0.5*matrixEr(nRRow-3,j) ) / gridSizeR ; // backward difference
      drDist =
        (-(11.0 / 6.0) * matrixEr(nRRow - 1, j) + (3.0 * matrixEr(nRRow - 2, j)) -
         (1.5 * matrixEr(nRRow - 3, j)) +
         ((1.0 / 3.0) * matrixEr(nRRow - 4, j))) /
        (-1 * gridSizeR);

      //dzDist    =  ( matrixEz(nRRow-1,j+1) - matrixEz(nRRow-1,j-1) ) / (2*gridSizeZ) ; // z direction
      dzDist = (-matrixEz(nRRow - 1, j + 2) + 8 * matrixEz(nRRow - 1, j + 1) - 8 * matrixEz(nRRow - 1, j - 1) +
                matrixEz(nRRow - 1, j - 2)) /
               (12 * gridSizeZ);

      dPhi = (-matrixEPhiP2(nRRow - 1, j) + 8 * matrixEPhiP(nRRow - 1, j) - 8 * matrixEPhiM(nRRow - 1, j) +
              matrixEPhiM2(nRRow - 1, j)) /
             (12 * gridSizePhi);
      matrixCharge(nRRow - 1, j) = -1 * (matrixEr(nRRow - 1, j) / radius + drDist + dPhi / radius + dzDist);
    }

    // boundary z
    for (Int_t i = 2; i < nRRow - 2; i++) {
      // z[0]
      radius = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
      dzDist = (-(11.0 / 6.0) * matrixEz(i, 0) + (3.0 * matrixEz(i, 1)) - (1.5 * matrixEz(i, 2)) +
                ((1.0 / 3.0) * matrixEz(i, 3))) /
               (1 * gridSizeZ); // forward difference
      drDist = (-matrixEr(i + 2, 0) + 8 * matrixEr(i + 1, 0) - 8 * matrixEr(i - 1, 0) + matrixEr(i - 2, 0)) /
               (12 * gridSizeR); // z direction
      dPhi = (-matrixEPhiP2(i, 0) + 8 * matrixEPhiP(i, 0) - 8 * matrixEPhiM(i, 0) + matrixEPhiM2(i, 0)) /
             (12 * gridSizePhi);
      matrixCharge(i, 0) = -1 * (matrixEr(i, 0) / radius + drDist + dPhi / radius + dzDist);

      dzDist = (matrixEz(i, 2) - matrixEz(i, 0)) / (2.0 * gridSizeZ); // forward difference

      drDist = (-matrixEr(i + 2, 1) + 8 * matrixEr(i + 1, 1) - 8 * matrixEr(i - 1, 1) + matrixEr(i - 2, 1)) /
               (12 * gridSizeR); // z direction
      dPhi = (-matrixEPhiP2(i, 1) + 8 * matrixEPhiP(i, 1) - 8 * matrixEPhiM(i, 1) + matrixEPhiM2(i, 1)) /
             (12 * gridSizePhi);
      matrixCharge(i, 1) = -1 * (matrixEr(i, 1) / radius + drDist + dPhi / radius + dzDist);

      dzDist = (matrixEz(i, nZColumn - 1) - matrixEz(i, nZColumn - 3)) / (2.0 * gridSizeZ); // forward difference

      drDist = (-matrixEr(i + 2, nZColumn - 2) + 8 * matrixEr(i + 1, nZColumn - 2) -
                8 * matrixEr(i - 1, nZColumn - 2) +
                matrixEr(i - 2, nZColumn - 2)) /
               (12 * gridSizeR); // z direction
      dPhi = (-matrixEPhiP2(i, nZColumn - 2) + 8 * matrixEPhiP(i, nZColumn - 2) -
              8 * matrixEPhiM(i, nZColumn - 2) +
              matrixEPhiM2(i, nZColumn - 2)) /
             (12 * gridSizePhi);
      matrixCharge(i, nZColumn - 2) = -1 * (matrixEr(i, nZColumn - 2) / radius + drDist + dPhi / radius + dzDist);

      dzDist = (-(11.0 / 6.0) * matrixEz(i, nZColumn - 1) + (3.0 * matrixEz(i, nZColumn - 2)) -
                (1.5 * matrixEz(i, nZColumn - 3)) + ((1.0 / 3.0) * matrixEz(i, nZColumn - 4))) /
               (-gridSizeZ); // backward difference
      drDist = (-matrixEr(i + 2, nZColumn - 1) + 8 * matrixEr(i + 1, nZColumn - 1) -
                8 * matrixEr(i - 1, nZColumn - 1) +
                matrixEr(i - 2, nZColumn - 1)) /
               (12 * gridSizeR); // z direction
      dPhi = (-matrixEPhiP2(i, nZColumn - 1) + 8 * matrixEPhiP(i, nZColumn - 1) -
              8 * matrixEPhiM(i, nZColumn - 1) +
              matrixEPhiM2(i, nZColumn - 1)) /
             (12 * gridSizePhi);
      matrixCharge(i, nZColumn - 1) = -1 * (matrixEr(i, nZColumn - 1) / radius + drDist + dPhi / radius + dzDist);
    }
    // for corner points
    // corner points for EPhi
    radius = AliTPCPoissonSolver::fgkIFCRadius;
    drDist =
      (-0.5 * matrixEr(2, 0) + 2.0 * matrixEr(1, 0) - 1.5 * matrixEr(0, 0)) / gridSizeR; // forward difference
    dzDist =
      (-0.5 * matrixEz(0, 2) + 2.0 * matrixEz(0, 1) - 1.5 * matrixEz(0, 0)) / gridSizeZ; // forward difference
    dPhi = (-matrixEPhiP2(0, 0) + 8 * matrixEPhiP(0, 0) - 8 * matrixEPhiM(0, 0) + matrixEPhiM2(0, 0)) /
           (12 * gridSizePhi);
    matrixCharge(0, 0) = -1 * (matrixEr(0, 0) / radius + drDist + dPhi / radius + dzDist);
    drDist =
      (-0.5 * matrixEr(2, 1) + 2.0 * matrixEr(1, 1) - 1.5 * matrixEr(0, 1)) / gridSizeR; // forward difference
    dzDist = (matrixEz(0, 2) - matrixEz(0, 0)) / (2.0 * gridSizeZ);                      // forward difference
    dPhi = (-matrixEPhiP2(0, 1) + 8 * matrixEPhiP(0, 1) - 8 * matrixEPhiM(0, 1) + matrixEPhiM2(0, 1)) /
           (12 * gridSizePhi);
    matrixCharge(0, 1) = -1 * (matrixEr(0, 1) / radius + drDist + dPhi / radius + dzDist);
    drDist =
      (-0.5 * matrixEr(2, nZColumn - 2) + 2.0 * matrixEr(1, nZColumn - 2) - 1.5 * matrixEr(0, nZColumn - 2)) /
      gridSizeR; // forward difference
    dzDist = (2.0 * matrixEz(0, nZColumn - 1) + 3.0 * matrixEz(0, nZColumn - 2) - 6.0 * matrixEz(0, nZColumn - 3) +
              matrixEz(0, nZColumn - 4)) /
             (6.0 * gridSizeZ); // backward difference
    dPhi = (-matrixEPhiP2(0, nZColumn - 2) + 8 * matrixEPhiP(0, nZColumn - 2) - 8 * matrixEPhiM(0, nZColumn - 2) +
            matrixEPhiM2(0, nZColumn - 2)) /
           (12 * gridSizePhi);
    matrixCharge(0, nZColumn - 2) = -1 * (matrixEr(0, nZColumn - 2) / radius + drDist + dPhi / radius + dzDist);
    drDist =
      (-0.5 * matrixEr(2, nZColumn - 1) + 2.0 * matrixEr(1, nZColumn - 1) - 1.5 * matrixEr(0, nZColumn - 1)) /
      gridSizeR; // forward difference
    dzDist = (1.5 * matrixEz(0, nZColumn - 1) - 2.0 * matrixEz(0, nZColumn - 2) + 0.5 * matrixEz(0, nZColumn - 3)) /
             gridSizeZ; // backward difference
    dPhi = (-matrixEPhiP2(0, nZColumn - 1) + 8 * matrixEPhiP(0, nZColumn - 1) - 8 * matrixEPhiM(0, nZColumn - 1) +
            matrixEPhiM2(0, nZColumn - 1)) /
           (12 * gridSizePhi);
    matrixCharge(0, nZColumn - 1) = -1 * (matrixEr(0, nZColumn - 1) / radius + drDist + dPhi / radius + dzDist);

    radius = AliTPCPoissonSolver::fgkIFCRadius + gridSizeR;
    drDist = (-matrixEr(3, 0) + 6.0 * matrixEr(2, 0) - 3.0 * matrixEr(1, 0) - 2 * matrixEr(0, 0)) /
             (6.0 * gridSizeR); // forward difference
    dzDist =
      (-0.5 * matrixEz(1, 2) + 2.0 * matrixEz(1, 1) - 1.5 * matrixEz(1, 0)) / gridSizeZ; // forward difference
    dPhi = (-matrixEPhiP2(1, 0) + 8 * matrixEPhiP(1, 0) - 8 * matrixEPhiM(1, 0) + matrixEPhiM2(1, 0)) /
           (12 * gridSizePhi);
    matrixCharge(1, 0) = -1 * (matrixEr(1, 0) / radius + drDist + dPhi / radius + dzDist);
    drDist = (-matrixEr(3, 1) + 6.0 * matrixEr(2, 1) - 3.0 * matrixEr(1, 1) - 2 * matrixEr(0, 1)) /
             (6.0 * gridSizeR); // forward difference
    dzDist = (-matrixEz(1, 3) + 6.0 * matrixEz(1, 2) - 3.0 * matrixEz(1, 1) - 2 * matrixEz(1, 0)) /
             (6.0 * gridSizeZ); // forward difference
    dPhi = (-matrixEPhiP2(1, 1) + 8 * matrixEPhiP(1, 1) - 8 * matrixEPhiM(1, 1) + matrixEPhiM2(1, 1)) /
           (12 * gridSizePhi);
    matrixCharge(1, 1) = -1 * (matrixEr(1, 1) / radius + drDist + dPhi / radius + dzDist);
    drDist = (-matrixEr(3, nZColumn - 2) + 6.0 * matrixEr(2, nZColumn - 2) - 3.0 * matrixEr(1, nZColumn - 2) -
              2 * matrixEr(0, nZColumn - 2)) /
             (6.0 * gridSizeR); // forward difference
    dzDist = (2.0 * matrixEz(1, nZColumn - 1) + 3.0 * matrixEz(1, nZColumn - 2) - 6.0 * matrixEz(1, nZColumn - 3) +
              matrixEz(1, nZColumn - 4)) /
             (6.0 * gridSizeZ); // backward difference
    dPhi = (-matrixEPhiP2(1, nZColumn - 2) + 8 * matrixEPhiP(1, nZColumn - 2) - 8 * matrixEPhiM(1, nZColumn - 2) +
            matrixEPhiM2(1, nZColumn - 2)) /
           (12 * gridSizePhi);
    matrixCharge(1, nZColumn - 2) = -1 * (matrixEr(1, nZColumn - 2) / radius + drDist + dPhi / radius + dzDist);

    drDist = (-matrixEr(3, nZColumn - 1) + 6.0 * matrixEr(2, nZColumn - 1) - 3.0 * matrixEr(1, nZColumn - 1) -
              2 * matrixEr(0, nZColumn - 1)) /
             (6.0 * gridSizeR); // forward difference
    dzDist = (1.5 * matrixEz(1, nZColumn - 1) - 2.0 * matrixEz(1, nZColumn - 2) + 0.5 * matrixEz(1, nZColumn - 3)) /
             gridSizeZ; // backward difference
    dPhi = (-matrixEPhiP2(1, nZColumn - 1) + 8 * matrixEPhiP(1, nZColumn - 1) - 8 * matrixEPhiM(1, nZColumn - 1) +
            matrixEPhiM2(1, nZColumn - 1)) /
           (12 * gridSizePhi);
    matrixCharge(1, nZColumn - 1) = -1 * (matrixEr(1, nZColumn - 1) / radius + drDist + dPhi / radius + dzDist);

    radius = AliTPCPoissonSolver::fgkIFCRadius + (nRRow - 2) * gridSizeR;
    drDist = (2.0 * matrixEr(nRRow - 1, 0) + 3.0 * matrixEr(nRRow - 2, 0) - 6.0 * matrixEr(nRRow - 3, 0) +
              matrixEr(nRRow - 4, 0)) /
             (6.0 * gridSizeR); // backward difference
    dzDist = (-0.5 * matrixEz(nRRow - 2, 2) + 2.0 * matrixEz(nRRow - 2, 1) - 1.5 * matrixEz(nRRow - 2, 0)) /
             gridSizeZ; // forward difference
    dPhi = (-matrixEPhiP2(nRRow - 2, 0) + 8 * matrixEPhiP(nRRow - 2, 0) - 8 * matrixEPhiM(nRRow - 2, 0) +
            matrixEPhiM2(nRRow - 2, 0)) /
           (12 * gridSizePhi);

    matrixCharge(nRRow - 2, 0) = -1 * (matrixEr(nRRow - 2, 0) / radius + drDist + dPhi / radius + dzDist);
    drDist = (2.0 * matrixEr(nRRow - 1, 1) + 3.0 * matrixEr(nRRow - 2, 1) - 6.0 * matrixEr(nRRow - 3, 1) +
              matrixEr(nRRow - 4, 1)) /
             (6.0 * gridSizeR); // backward difference
    dzDist = (-matrixEz(nRRow - 2, 3) + 6.0 * matrixEz(nRRow - 2, 2) - 3.0 * matrixEz(nRRow - 2, 1) -
              2 * matrixEz(nRRow - 2, 0)) /
             (6.0 * gridSizeZ); // forward difference
    dPhi = (-matrixEPhiP2(nRRow - 2, 1) + 8 * matrixEPhiP(nRRow - 2, 1) - 8 * matrixEPhiM(nRRow - 2, 1) +
            matrixEPhiM2(nRRow - 2, 1)) /
           (12 * gridSizePhi);
    matrixCharge(nRRow - 2, 1) = -1 * (matrixEr(nRRow - 2, 1) / radius + drDist + dPhi / radius + dzDist);
    drDist = (2.0 * matrixEr(nRRow - 1, nZColumn - 2) + 3.0 * matrixEr(nRRow - 2, nZColumn - 2) -
              6.0 * matrixEr(nRRow - 3, nZColumn - 2) + matrixEr(nRRow - 4, nZColumn - 2)) /
             (6.0 * gridSizeR); // backward difference
    dzDist = (2.0 * matrixEz(nRRow - 2, nZColumn - 1) + 3.0 * matrixEz(nRRow - 2, nZColumn - 2) -
              6.0 * matrixEz(nRRow - 2, nZColumn - 3) + matrixEz(nRRow - 2, nZColumn - 4)) /
             (6.0 * gridSizeZ); // backward difference
    dPhi = (-matrixEPhiP2(nRRow - 2, nZColumn - 2) + 8 * matrixEPhiP(nRRow - 2, nZColumn - 2) -
            8 * matrixEPhiM(nRRow - 2, nZColumn - 2) + matrixEPhiM2(nRRow - 2, nZColumn - 2)) /
           (12 * gridSizePhi);
    matrixCharge(nRRow - 2, nZColumn - 2) =
      -1 * (matrixEr(nRRow - 2, nZColumn - 2) / radius + drDist + dPhi / radius + dzDist);
    drDist = (2.0 * matrixEr(nRRow - 1, nZColumn - 1) + 3.0 * matrixEr(nRRow - 2, nZColumn - 1) -
              6.0 * matrixEr(nRRow - 3, nZColumn - 1) + matrixEr(nRRow - 4, nZColumn - 1)) /
             (6.0 * gridSizeR); // backward difference
    dzDist = (1.5 * matrixEz(0, nZColumn - 1) - 2.0 * matrixEz(0, nZColumn - 2) + 0.5 * matrixEz(0, nZColumn - 3)) /
             gridSizeZ; // backward difference
    dPhi = (-matrixEPhiP2(nRRow - 2, nZColumn - 1) + 8 * matrixEPhiP(nRRow - 2, nZColumn - 1) -
            8 * matrixEPhiM(nRRow - 2, nZColumn - 1) + matrixEPhiM2(nRRow - 2, nZColumn - 1)) /
           (12 * gridSizePhi);

    matrixCharge(nRRow - 2, nZColumn - 1) =
      -1 * (matrixEr(nRRow - 2, nZColumn - 1) / radius + drDist + dPhi / radius + dzDist);
    radius = AliTPCPoissonSolver::fgkIFCRadius + (nRRow - 1) * gridSizeR;
    drDist = (1.5 * matrixEr(nRRow - 1, 0) - 2.0 * matrixEr(nRRow - 2, 0) + 0.5 * matrixEr(nRRow - 3, 0)) /
             gridSizeR; // backward difference
    dzDist = (-0.5 * matrixEz(nRRow - 1, 2) + 2.0 * matrixEz(nRRow - 1, 1) - 1.5 * matrixEz(nRRow - 1, 0)) /
             gridSizeZ; // forward difference
    dPhi = (-matrixEPhiP2(nRRow - 1, 0) + 8 * matrixEPhiP(nRRow - 1, 0) - 8 * matrixEPhiM(nRRow - 1, 0) +
            matrixEPhiM2(nRRow - 1, 0)) /
           (12 * gridSizePhi);
    matrixCharge(nRRow - 1, 0) = -1 * (matrixEr(nRRow - 1, 0) / radius + drDist + dPhi / radius + dzDist);
    drDist = (1.5 * matrixEr(nRRow - 1, 1) - 2.0 * matrixEr(nRRow - 2, 1) + 0.5 * matrixEr(nRRow - 3, 1)) /
             gridSizeR; // backward difference
    dzDist = (-matrixEz(nRRow - 1, 3) + 6.0 * matrixEz(nRRow - 1, 2) - 3.0 * matrixEz(nRRow - 1, 1) -
              2 * matrixEz(nRRow - 1, 0)) /
             (6.0 * gridSizeZ); // forward difference
    dPhi = (-matrixEPhiP2(nRRow - 1, 1) + 8 * matrixEPhiP(nRRow - 1, 1) - 8 * matrixEPhiM(nRRow - 1, 1) +
            matrixEPhiM2(nRRow - 1, 1)) /
           (12 * gridSizePhi);
    matrixCharge(nRRow - 1, 1) = -1 * (matrixEr(nRRow - 1, 1) / radius + drDist + dPhi / radius + dzDist);

    drDist = (1.5 * matrixEr(nRRow - 1, nZColumn - 2) - 2.0 * matrixEr(nRRow - 2, nZColumn - 2) +
              0.5 * matrixEr(nRRow - 3, nZColumn - 2)) /
             gridSizeR; // backward difference
    dzDist = (2.0 * matrixEz(nRRow - 1, nZColumn - 1) + 3.0 * matrixEz(nRRow - 1, nZColumn - 2) -
              6.0 * matrixEz(nRRow - 1, nZColumn - 3) + matrixEz(nRRow - 1, nZColumn - 4)) /
             (6.0 * gridSizeZ); // backward difference
    dPhi = (-matrixEPhiP2(nRRow - 1, nZColumn - 2) + 8 * matrixEPhiP(nRRow - 1, nZColumn - 2) -
            8 * matrixEPhiM(nRRow - 1, nZColumn - 2) + matrixEPhiM2(nRRow - 1, nZColumn - 2)) /
           (12 * gridSizePhi);
    matrixCharge(nRRow - 1, nZColumn - 2) =
      -1 * (matrixEr(nRRow - 1, nZColumn - 2) / radius + drDist + dPhi / radius + dzDist);

    drDist = (1.5 * matrixEr(nRRow - 1, nZColumn - 1) - 2.0 * matrixEr(nRRow - 2, nZColumn - 1) +
              0.5 * matrixEr(nRRow - 3, nZColumn - 1)) /
             gridSizeR; // backward difference
    dzDist = (1.5 * matrixEz(nRRow - 1, nZColumn - 1) - 2.0 * matrixEz(nRRow - 1, nZColumn - 2) +
              0.5 * matrixEz(nRRow - 1, nZColumn - 3)) /
             gridSizeZ; // backward difference

    dPhi = (-matrixEPhiP2(nRRow - 1, nZColumn - 1) + 8 * matrixEPhiP(nRRow - 1, nZColumn - 1) -
            8 * matrixEPhiM(nRRow - 1, nZColumn - 1) + matrixEPhiM2(nRRow - 1, nZColumn - 1)) /
           (12 * gridSizePhi);

    matrixCharge(nRRow - 1, nZColumn - 1) =
      -1 * (matrixEr(nRRow - 1, nZColumn - 1) / radius + drDist + dPhi / radius + dzDist);
  }
}
///
/// \param matricesCharge
/// \param matricesEr
/// \param matricesEPhi
/// \param matricesEz
/// \param matricesInvLocalIntErDz
/// \param matricesInvLocalIntEPhiDz
/// \param matricesInvLocalEz
/// \param matricesDistDrDz
/// \param matricesDistDPhiRDz
/// \param matricesDistDz
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
/// \param nSize
/// \param useCylAC
/// \param stepR
/// \param stepZ
/// \param stepPhi
/// \param interpType
/// \param inverseType
void AliTPCSpaceCharge3DCalc::InverseDistortionMaps(
  TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
  TMatrixD** matricesInvLocalIntErDz, TMatrixD** matricesInvLocalIntEPhiDz, TMatrixD** matricesInvLocalEz,
  TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice, const Int_t nSize,
  const Bool_t useCylAC, Int_t stepR, Int_t stepZ, Int_t stepPhi, Int_t interpType)
{
  // can inverse after lookup table for global distortion been calculated
  Double_t* rList = new Double_t[nRRow];
  Double_t* zList = new Double_t[nZColumn];
  Double_t* phiList = new Double_t[phiSlice];
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = (j * gridSizeZ);
  }
  // memory allocation
  if (fInitLookUp) {
    // 1)  get local distortion
    InverseGlobalToLocalDistortionGlobalInvTable(matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz, rList,
                                                 zList, phiList, nRRow, nZColumn, phiSlice, nSize, useCylAC,
                                                 stepR, stepZ, stepPhi, interpType);

    fLookupInverseDistA->SetLookUpR(matricesDistDrDz);
    fLookupInverseDistA->SetLookUpPhi(matricesDistDPhiRDz);
    fLookupInverseDistA->SetLookUpZ(matricesDistDz);
    fLookupInverseDistA->CopyFromMatricesToInterpolator();

    // 2)  calculate local integral
    InverseLocalDistortionToElectricField(matricesEr, matricesEPhi, matricesEz, matricesInvLocalIntErDz,
                                          matricesInvLocalIntEPhiDz, matricesInvLocalEz,
                                          matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz, rList, zList,
                                          phiList, nRRow, nZColumn, phiSlice);
    // 3)  get potential from electric field assuming zero boundaries
    InverseElectricFieldToCharge(matricesCharge, matricesEr, matricesEPhi, matricesEz, rList, zList, phiList, nRRow,
                                 nZColumn, phiSlice);
  }

  // copy charge inverse here just for side A (TODO: do for side C)
  for (Int_t k = 0; k < phiSlice; k++) {
    *(fMatrixChargeInverseA[k]) = *(matricesCharge[k]);
  }
  fInterpolatorInverseChargeA->SetValue(fMatrixChargeInverseA);
  fInterpolatorInverseChargeA->InitCubicSpline();

  delete[] zList;
  delete[] rList;
  delete[] phiList;
}
/// CalculateEField (New Version: with reorganization of modules)
/// Calculate E field based on look-up table created by Poisson Solver
/// * Differentiate V(r) and solve for E(r) using special equations for the first and last row
/// * Integrate E(r)/E(z) from point of origin to pad plane
/// * Differentiate V(r) and solve for E(phi)
/// * Integrate E(phi)/E(z) from point of origin to pad plane
/// * Differentiate V(r) and solve for E(z) using special equations for the first and last row
/// * Integrate (E(z)-Ez(ROC)) from point of origin to pad plane
///
/// \param matricesV TMatrixD** 3D matrix representing calculated potential
/// \param matricesErOverEz TMatrix** 3D matrix representing e-field at Er/Ez
/// \param matricesEPhiOverEz TMatrix** 3D matrix representing e-field at EPhi/Ez
/// \param matricesDeltaZ TMatrix** 3D matrix representing e-field at DeltaZ
/// \param nRRow Int_t number of nRRow (in R direction)
/// \param nZColumn Int_t number of nZColumn (in Z direction)
/// \param phiSlice Int_t number of (phi slices in phi direction)
/// \param symmetry Int_t symmetry?
/// \param rocDisplace rocDisplacement
///
/// \pre   Matrix matricesV is assumed had been calculated  by Poisson solver
/// \post  Results of Integration and Derivations for E-field calculation are stored in matricesErOverEz, matricesEPhiOverEz, matricesDeltaZ
///
void AliTPCSpaceCharge3DCalc::CalculateEField(
  TMatrixD** matricesV, TMatrixD** matricesErOverEz, TMatrixD** matricesEPhiOverEz,
  TMatrixD** matricesDeltaEz, const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice,
  const Int_t symmetry, Bool_t rocDisplacement)
{

  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  TMatrixD *matricesEr[phiSlice], *matricesEz[phiSlice], *matricesEPhi[phiSlice];

  //Allocate memory for electric field r,z, phi direction
  for (Int_t k = 0; k < phiSlice; k++) {
    matricesEr[k] = new TMatrixD(nRRow, nZColumn);
    matricesEz[k] = new TMatrixD(nRRow, nZColumn);
    matricesEPhi[k] = new TMatrixD(nRRow, nZColumn);
  }

  //Differentiate V(r) and solve for E(r) using special equations for the first and last row
  TStopwatch w;
  w.Start();

  ElectricField(matricesV, matricesEr, matricesEPhi, matricesEz, nRRow, nZColumn,
                phiSlice, gridSizeR, gridSizePhi, gridSizeZ, symmetry, AliTPCPoissonSolver::fgkIFCRadius);

  w.Stop();
  Info("AliTPCSpaceCharge3DCalc::InitSpaceCharge3DPoissonIntegralDz", "%s", Form("Time for calculation E-field CPU = %f s\n", w.CpuTime()));

  //Integrate E(r)/E(z) from point of origin to pad plane

  IntegrateEz(matricesErOverEz, matricesEr, nRRow, nZColumn, phiSlice, ezField);
  IntegrateEz(matricesEPhiOverEz, matricesEPhi, nRRow, nZColumn, phiSlice, ezField);
  IntegrateEz(matricesDeltaEz, matricesEz, nRRow, nZColumn, phiSlice, -1.0);

  // calculate z distortion from the integrated Delta Ez residuals
  // and include the equivalence (Volt to cm) of the ROC shift !!
  for (Int_t m = 0; m < phiSlice; m++) {
    TMatrixD& arrayV = *matricesV[m];
    TMatrixD& deltaEz = *matricesDeltaEz[m];

    for (Int_t j = 0; j < nZColumn; j++) {
      for (Int_t i = 0; i < nRRow; i++) {
        // Scale the Ez distortions with the drift velocity  -> delivers cm
        deltaEz(i, j) = deltaEz(i, j) * AliTPCPoissonSolver::fgkdvdE;
        // ROC Potential in cm equivalent
        Double_t dzROCShift = arrayV(i, nZColumn - 1) / ezField;
        if (rocDisplacement) {
          deltaEz(i, j) = deltaEz(i, j) + dzROCShift; // add the ROC mis alignment
        }
      }
    }
  }
  // clear the temporary arrays lists

  for (Int_t k = 0; k < phiSlice; k++) {
    delete matricesEr[k];
    delete matricesEz[k];
    delete matricesEPhi[k];
  }
}
///
/// Integrate at z direction Ez for electron drift calculation
///
///
/// \param matricesExOverEz TMatrixD** 3D matrix representing ExOverEz
/// \param matricesEx TMatrix** 3D matrix representing e-field at x direction
/// \param nRRow const Int_t number of nRRow  (in R direction)
/// \param nZColumn const Int_t number of nZColumn  (in Z direction)
/// \param phiSlice const Int_t number of (phiSlice in phi direction)
/// \param ezField const Double_t Electric field in z direction
///
/// \pre   matricesEx is assumed already been calculated by ElectricFieldCalculation
/// \post  Matrix matricesExOverEz is calculated by integration of matricesEx
///
void AliTPCSpaceCharge3DCalc::IntegrateEz(
  TMatrixD** matricesExOverEz, TMatrixD** matricesEx, const Int_t nRRow, const Int_t nZColumn,
  const Int_t phiSlice, const Double_t ezField)
{
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  for (Int_t m = 0; m < phiSlice; m++) {
    TMatrixD& eXoverEz = *matricesExOverEz[m];
    TMatrixD& arrayEx = *matricesEx[m];

    for (Int_t j = nZColumn - 1; j >= 0; j--) {
      for (Int_t i = 0; i < nRRow; i++) {

        /// Calculate integration from int^{0}_{j} (TODO: Split the integration)
        if (j < nZColumn - 3) {
          eXoverEz(i, j) = eXoverEz(i, j + 2) +
                           (gridSizeZ / 3.0) * (arrayEx(i, j) + 4 * arrayEx(i, j + 1) + arrayEx(i, j + 2)) /
                             (-1 * ezField);
        } else {
          if (j == nZColumn - 3) {
            eXoverEz(i, j) = (gridSizeZ / 3.0) * (arrayEx(i, nZColumn - 3) + 4 * arrayEx(i, nZColumn - 2) + arrayEx(i, nZColumn - 1)) / (-1 * ezField);
          }
          if (j == nZColumn - 2) {
            eXoverEz(i, j) =
              (gridSizeZ / 3.0) * (1.5 * arrayEx(i, nZColumn - 2) + 1.5 * arrayEx(i, nZColumn - 1)) /
              (-1 * ezField);
          }
          if (j == nZColumn - 1) {
            eXoverEz(i, j) = 0.0;
          }
        }
      }
    }
  }
}
/// GetCorrection from no-drift
///
/// \param x Float_t point origin
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetCorrectionCylNoDrift(const Float_t x[], const Short_t roc, Float_t dx[])
{
  /// Calculates the correction due the Space Charge effect within the TPC drift volume

  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::", "Lookup table was not initialized! Performing the initialization now ...");
    //    InitSpaceCharge3DDistortion();
    return;
  }

  Float_t intEr, intEPhi, intDEz;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi -= TMath::TwoPi();
  }

  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (sign == -1 && z < 0.0) {
    printf("call C side\n");
    fLookupIntENoDriftC->GetValue(r, phi, z, intEr, intEPhi, intDEz);
  } else {
    fLookupIntENoDriftA->GetValue(r, phi, z, intEr, intEPhi, intDEz);
  }

  // Calculate distorted position
  if (r > 0.0) {
    phi = phi + fCorrectionFactor * (fC0 * intEPhi - fC1 * intEr) / r;
    r = r + fCorrectionFactor * (fC0 * intEr + fC1 * intEPhi);
  }
  Double_t dzDist = intDEz * fCorrectionFactor * AliTPCPoissonSolver::fgkdvdE;

  // Calculate correction in cartesian coordinates
  dx[0] = -(r - x[0]);
  dx[1] = -(phi - x[1]);
  dx[2] = -dzDist; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}
///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetDistortionCylNoDrift(const Float_t x[], Short_t roc, Float_t dx[])
{
  /// This function delivers the distortion values dx in respect to the initial coordinates x
  /// roc represents the TPC read out chamber (offline numbering convention)

  GetCorrectionCylNoDrift(x, roc, dx);
  for (Int_t j = 0; j < 3; ++j) {
    dx[j] = -dx[j];
  }
}
/// inverse for no drift
/// inverse from global distortion to local distortion
///
/// \param matricesDistDrDz
/// \param matricesDistDPhiRDz
/// \param matricesDistDz
/// \param rList
/// \param zList
/// \param phiList
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
void AliTPCSpaceCharge3DCalc::InverseGlobalToLocalDistortionNoDrift(
  TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz,
  Double_t* rList, Double_t* zList, Double_t* phiList,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice)
{

  Double_t z, phi, r, zAfter, zPrevious, ddR, ddRPhi, ddZ, drDist, dRPhi, dzDist;
  Float_t x[3], dx[3], pdx[3], dxp1[3], dxp2[3];
  Int_t roc;
  TMatrixD* distDrDz;
  TMatrixD* distDPhiRDz;
  TMatrixD* distDz;

  for (Int_t k = 0; k < phiSlice; k++) {
    distDrDz = matricesDistDrDz[k];
    distDPhiRDz = matricesDistDPhiRDz[k];
    distDz = matricesDistDz[k];
    for (Int_t i = 0; i < nRRow; i++) {
      (*distDrDz)(i, nZColumn - 1) = 0.0;
      (*distDPhiRDz)(i, nZColumn - 1) = 0.0;
      (*distDz)(i, nZColumn - 1) = 0.0;
    }
  }

  for (Int_t j = nZColumn - 2; j >= 0; j--) {
    roc = 0; // FIXME
    for (Int_t k = 0; k < phiSlice; k++) {

      distDrDz = matricesDistDrDz[k];
      distDPhiRDz = matricesDistDPhiRDz[k];
      distDz = matricesDistDz[k];
      for (Int_t i = 0; i < nRRow; i++) {
        // get global distortion

        r = rList[i];
        phi = phiList[k];
        z = zList[j];
        zPrevious = zList[j + 1];
        //zAfter = zList[j-1];

        (*distDrDz)(i, j) = 0.0;
        (*distDPhiRDz)(i, j) = 0.0;
        (*distDz)(i, j) = 0.0;
        drDist = 0.0;
        dRPhi = 0.0;
        dzDist = 0.0;

        r = rList[i];
        phi = phiList[k];
        z = zList[j];

        x[0] = r;
        x[1] = phi;
        x[2] = z;

        GetDistortionCylNoDrift(x, roc, dx);

        //x[0] = x[0] + drDist;
        //x[1] = x[1] + dRPhi/r;
        x[2] = zPrevious;

        GetDistortionCylNoDrift(x, roc, pdx);

        (*distDrDz)(i, j) = (dx[0] - pdx[0]);
        (*distDPhiRDz)(i, j) = (dx[1] - pdx[1]) * r;
        (*distDz)(i, j) = (dx[2] - pdx[2]);
      }
    }
  }
}
///
/// \param matricesCharge
/// \param matricesEr
/// \param matricesEPhi
/// \param matricesEz
/// \param matricesInvLocalIntErDz
/// \param matricesInvLocalIntEPhiDz
/// \param matricesInvLocalEz
/// \param matricesDistDrDz
/// \param matricesDistDPhiRDz
/// \param matricesDistDz
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
void AliTPCSpaceCharge3DCalc::InverseDistortionMapsNoDrift(
  TMatrixD** matricesCharge, TMatrixD** matricesEr, TMatrixD** matricesEPhi, TMatrixD** matricesEz,
  TMatrixD** matricesInvLocalIntErDz, TMatrixD** matricesInvLocalIntEPhiDz, TMatrixD** matricesInvLocalEz,
  TMatrixD** matricesDistDrDz, TMatrixD** matricesDistDPhiRDz, TMatrixD** matricesDistDz,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice)
{
  // can inverse after lookup table for global distortion been calculated
  Double_t* rList = new Double_t[nRRow];
  Double_t* zList = new Double_t[nZColumn];
  Double_t* phiList = new Double_t[phiSlice];
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;

  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = (j * gridSizeZ);
  }
  // memory allocation
  if (fInitLookUp) {
    // 1)  get local distortion
    InverseGlobalToLocalDistortionNoDrift(matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz, rList, zList,
                                          phiList, nRRow, nZColumn, phiSlice);
    // 2)  calculate local integral
    InverseLocalDistortionToElectricField(matricesEr, matricesEPhi, matricesEz, matricesInvLocalIntErDz,
                                          matricesInvLocalIntEPhiDz, matricesInvLocalEz,
                                          matricesDistDrDz, matricesDistDPhiRDz, matricesDistDz, rList, zList,
                                          phiList, nRRow, nZColumn, phiSlice);
    // 3)  get potential from electric field assuming zero boundaries
    InverseElectricFieldToCharge(matricesCharge, matricesEr, matricesEPhi, matricesEz, rList, zList, phiList, nRRow,
                                 nZColumn, phiSlice);
  }
  delete[] zList;
  delete[] rList;
  delete[] phiList;
}
///
/// \param matricesChargeA
/// \param matricesChargeC
/// \param spaceChargeHistogram3D
/// \param nRRow
/// \param nZColumn
/// \param phiSlice
void AliTPCSpaceCharge3DCalc::GetChargeDensity(
  TMatrixD** matricesChargeA, TMatrixD** matricesChargeC, TH3* spaceChargeHistogram3D,
  const Int_t nRRow, const Int_t nZColumn, const Int_t phiSlice)
{
  Int_t phiSlicesPerSector = phiSlice / kNumSector;
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (nRRow - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (nZColumn - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / phiSlice;
  const Double_t ezField = (AliTPCPoissonSolver::fgkCathodeV - AliTPCPoissonSolver::fgkGG) / AliTPCPoissonSolver::fgkTPCZ0; // = ALICE Electric Field (V/cm) Magnitude ~ -400 V/cm;
  // local variables
  Float_t radius0, phi0, z0;
  // list of point as used in the poisson relaxation and the interpolation (for interpolation)
  Double_t rList[nRRow], zList[nZColumn], phiList[phiSlice];
  for (Int_t k = 0; k < phiSlice; k++) {
    phiList[k] = gridSizePhi * k;
  }
  for (Int_t i = 0; i < nRRow; i++) {
    rList[i] = AliTPCPoissonSolver::fgkIFCRadius + i * gridSizeR;
  }
  for (Int_t j = 0; j < nZColumn; j++) {
    zList[j] = j * gridSizeZ;
  }

  TMatrixD* mCharge;
  for (Int_t side = 0; side < 2; side++) {
    for (Int_t k = 0; k < phiSlice; k++) {
      if (side == 0) {
        mCharge = matricesChargeA[k];
      } else {
        mCharge = matricesChargeC[k];
      }

      phi0 = phiList[k];
      for (Int_t i = 0; i < nRRow; i++) {
        radius0 = rList[i];
        for (Int_t j = 0; j < nZColumn; j++) {
          z0 = zList[j];
          if (side == 1) {
            z0 = -TMath::Abs(zList[j]);
          }
          if (spaceChargeHistogram3D != nullptr) {
            (*mCharge)(i, j) = InterpolatePhi(spaceChargeHistogram3D, phi0, radius0, z0);
            //InterpolatePhi(spaceChargeHistogram3D,phi0,radius0,z0);
          }
        }
      }
    }
  }
}
///
/// \param x
/// \param roc
/// \return
Double_t AliTPCSpaceCharge3DCalc::GetChargeCylAC(const Float_t x[], Short_t roc)
{
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    return fInterpolatorChargeA->GetValue(r, phi, z);
  } else {
    return fInterpolatorChargeC->GetValue(r, phi, -z);
  }
}
///
/// \param x
/// \param roc
/// \return
Double_t AliTPCSpaceCharge3DCalc::GetPotentialCylAC(const Float_t x[], Short_t roc)
{
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    return fInterpolatorPotentialA->GetValue(r, phi, z);
  } else {
    return fInterpolatorPotentialC->GetValue(r, phi, -z);
  }
}
/// chargeInverse
///
/// \param x
/// \param roc
/// \return
Double_t AliTPCSpaceCharge3DCalc::GetInverseChargeCylAC(const Float_t x[], Short_t roc)
{
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < 0) || (sign == -1 && z > 0)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-6) {
    return fInterpolatorInverseChargeA->GetValue(r, phi, z);
  } else {
    return fInterpolatorInverseChargeC->GetValue(r, phi, z);
  }
}

///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetLocalDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[])
{
  Float_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < -1e-16) || (sign == -1 && z > -1e-16)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    fLookupDistA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupDistC->GetValue(r, phi, -z, dR, dRPhi, dZ);
    dZ = -1 * dZ;
  }

  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}

/// Get Electric field from look up table
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetElectricFieldCyl(const Float_t x[], Short_t roc, Double_t dx[])
{
  Double_t eR, ePhi, eZ;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < -1e-16) || (sign == -1 && z > -1e-16)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    fLookupElectricFieldA->GetValue(r, phi, z, eR, ePhi, eZ);
  } else {
    fLookupElectricFieldC->GetValue(r, phi, -z, eR, ePhi, eZ);
    eZ = -1 * eZ;
  }

  dx[0] = eR;
  dx[1] = ePhi;
  dx[2] = eZ;
}

///
/// \param x
/// \param roc
/// \param dx
void AliTPCSpaceCharge3DCalc::GetInverseLocalDistortionCyl(const Float_t x[], Short_t roc, Float_t dx[])
{
  if (!fInitLookUp) {
    Info("AliTPCSpaceCharge3DCalc::", "Lookup table was not initialized! Performing the initialization now ...");
    InitSpaceCharge3DPoissonIntegralDz(129, 129, 144, 100, 1e-8);
  }

  Float_t dR, dRPhi, dZ;
  Double_t r, phi, z;
  Int_t sign;

  r = x[0];
  phi = x[1];
  if (phi < 0) {
    phi += TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  if (phi > TMath::TwoPi()) {
    phi = phi - TMath::TwoPi(); // Table uses phi from 0 to 2*Pi
  }
  z = x[2]; // Create temporary copy of x[2]

  if ((roc % 36) < 18) {
    sign = 1; // (TPC A side)
  } else {
    sign = -1; // (TPC C side)
  }

  if (sign == 1 && z < AliTPCPoissonSolver::fgkZOffSet) {
    z = AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if (sign == -1 && z > -AliTPCPoissonSolver::fgkZOffSet) {
    z = -AliTPCPoissonSolver::fgkZOffSet; // Protect against discontinuity at CE
  }
  if ((sign == 1 && z < -1e-16) || (sign == -1 && z > -1e-16)) { // just a consistency check
    Error("AliTPCSpaceCharge3DCalc::", "ROC number does not correspond to z coordinate! Calculation of distortions is most likely wrong!");
  }

  if (z > -1e-16) {
    fLookupInverseDistA->GetValue(r, phi, z, dR, dRPhi, dZ);
  } else {
    fLookupInverseDistC->GetValue(r, phi, -z, dR, dRPhi, dZ);
  }

  dx[0] = fCorrectionFactor * dR;
  dx[1] = fCorrectionFactor * dRPhi;
  dx[2] = fCorrectionFactor *
          dZ; // z distortion - (scaled with drift velocity dependency on the Ez field and the overall scaling factor)
}
/// Function for setting Potential Boundary Values and Charge distribution input TFormula
///
/// \param vTestFunction
/// \param rhoTestFunction
///
void AliTPCSpaceCharge3DCalc::SetPotentialBoundaryAndChargeFormula(TFormula* vTestFunction, TFormula* rhoTestFunction)
{
  /**** allocate memory for charge ***/
  // we allocate pointer to TMatrixD array to picture 3D (slices), this representation should be revised
  // since it is easier for GPU implementation to run for 1D memory
  const Float_t gridSizeR = (AliTPCPoissonSolver::fgkOFCRadius - AliTPCPoissonSolver::fgkIFCRadius) / (fNRRows - 1);
  const Float_t gridSizeZ = AliTPCPoissonSolver::fgkTPCZ0 / (fNZColumns - 1);
  const Float_t gridSizePhi = TMath::TwoPi() / fNPhiSlices;

  fFormulaPotentialV = vTestFunction;
  fFormulaChargeRho = rhoTestFunction;

  // grid size for one side
  TMatrixD* chargeA;
  TMatrixD* chargeC;
  Double_t radius0, z0, phi0, z0neg;
  Int_t indexB = 0;
  for (Int_t k = 0; k < fNPhiSlices; k++) {
    chargeA = fMatrixChargeA[k];
    chargeC = fMatrixChargeC[k];

    phi0 = k * gridSizePhi;

    /// Fill the non-boundary values
    for (Int_t i = 0; i < fNRRows; i++) {
      radius0 = AliTPCPoissonSolver::fgkIFCRadius + (i * gridSizeR);
      for (Int_t j = 0; j < fNZColumns; j++) {
        z0 = j * gridSizeZ;
        z0neg = -z0;

        (*chargeA)(i, j) = -1.0 * rhoTestFunction->Eval(radius0, phi0, z0);
        (*chargeC)(i, j) = -1.0 * rhoTestFunction->Eval(radius0, phi0, z0neg);

        if ((i == 0) || (i == fNRRows - 1) || (j == 0) || (j == fNZColumns - 1)) {
          fListPotentialBoundaryA[indexB] = vTestFunction->Eval(radius0, phi0, z0);
          fListPotentialBoundaryC[indexB] = vTestFunction->Eval(radius0, phi0, z0neg);
          indexB++;
        }

      } // end j
    }   // end i
  }     // end phi

  fInterpolatorChargeA->SetValue(fMatrixChargeA);
  fInterpolatorChargeA->InitCubicSpline();
  fInterpolatorChargeC->SetValue(fMatrixChargeC);
  fInterpolatorChargeC->InitCubicSpline();
}

/// Set interpolation
void AliTPCSpaceCharge3DCalc::SetInterpolationOrder(Int_t order)
{
  fInterpolationOrder = order;

  fInterpolatorChargeA->SetOrder(fInterpolationOrder);
  fInterpolatorChargeC->SetOrder(fInterpolationOrder);
  fInterpolatorPotentialA->SetOrder(fInterpolationOrder);
  fInterpolatorPotentialC->SetOrder(fInterpolationOrder);
  fInterpolatorInverseChargeA->SetOrder(fInterpolationOrder);
  fInterpolatorInverseChargeC->SetOrder(fInterpolationOrder);

  fLookupDistA->SetOrder(fInterpolationOrder);

  fLookupDistC->SetOrder(fInterpolationOrder);

  fLookupInverseDistA->SetOrder(fInterpolationOrder);

  fLookupInverseDistC->SetOrder(fInterpolationOrder);

  fLookupElectricFieldA->SetOrder(fInterpolationOrder);
  fLookupElectricFieldC->SetOrder(fInterpolationOrder);
  fLookupIntDistA->SetOrder(fInterpolationOrder);
  fLookupIntDistC->SetOrder(fInterpolationOrder);
  fLookupIntCorrA->SetOrder(fInterpolationOrder);
  fLookupIntCorrC->SetOrder(fInterpolationOrder);
  fLookupIntENoDriftA->SetOrder(fInterpolationOrder);
  fLookupIntENoDriftC->SetOrder(fInterpolationOrder);
  fLookupIntCorrIrregularA->SetOrder(fInterpolationOrder);

  fLookupIntCorrIrregularC->SetOrder(fInterpolationOrder);
}
