// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCLookUpTable3DInterpolatorD.cxx
/// \brief Wrap up look-up table for correction/distortion integral or derivative (electric field)
///        assume 3 components: r-component, phi-component and z-component
///
/// \author Rifki Sadikin <rifki.sadikin@cern.ch>, Indonesian Institute of Sciences
/// \date Mar 4, 2015

#include "AliTPCLookUpTable3DInterpolatorD.h"

/// \cond CLASSIMP3
ClassImp(AliTPCLookUpTable3DInterpolatorD);
/// \endcond

/// constructor
AliTPCLookUpTable3DInterpolatorD::AliTPCLookUpTable3DInterpolatorD()
{
  fOrder = 1;
  fIsAllocatingLookUp = kFALSE;
}

/// constructor
///
/// \param nRRow Int_t size of grid in R direction
/// \param rMin  Double_t minimal value of R
/// \param rMax  Double_t maximal value of R
/// \param nPhiSlice Int_t size of grid Phi direction
/// \param phiMin Double_t minimal value of Phi
/// \param phiMax Double_t maximal value of Phi
/// \param nZColumn Int_t size of grid Z direction
/// \param zMin Double_t minimal value of Z
/// \param zMax Double_t maximal value of Z

/**
   AliTPCLookUpTable3DInterpolatorD::AliTPCLookUpTable3DInterpolatorD(Int_t nRRow, Double_t rMin, Double_t rMax,
                                                                   Int_t nPhiSlice,
                                                                   Double_t phiMin, Double_t phiMax, Int_t nZColumn,
                                                                   Double_t zMin, Double_t zMax) {
   fOrder = 1;
   fIsAllocatingLookUp = kTRUE;

   fNR = nRRow;
   fNPhi = nPhiSlice;
   fNZ = nZColumn;

   fLookUpR = new TMatrixD *[fNPhi];
   fLookUpPhi = new TMatrixD *[fNPhi];
   fLookUpZ = new TMatrixD *[fNPhi];

   for (Int_t m = 0; m < fNPhi; m++) {
    fLookUpR[m] = new TMatrixD(fNR, fNZ);
    fLookUpPhi[m] = new TMatrixD(fNR, fNZ);
    fLookUpZ[m] = new TMatrixD(fNR, fNZ);
   }

   fRList = new Double_t[fNR];
   fPhiList = new Double_t[fNPhi];
   fZList = new Double_t[fNZ];

   Double_t dR = (rMax - rMin) / fNR;
   Double_t dPhi = (phiMax - phiMin) / fNPhi;
   Double_t dZ = (zMax - zMin) / fNPhi;

   for (Int_t m = 0; m < fNPhi; m++) fPhiList[m] = phiMin + dPhi * m;
   for (Int_t m = 0; m < fNR; m++) fRList[m] = rMin + dR * m;
   for (Int_t m = 0; m < fNZ; m++) fZList[m] = zMin + dZ * m;
   }
 **/

/// Constructor
///
/// \param nRRow Int_t size of grid in R direction
/// \param matricesRValue TMatrixD** values of component R
/// \param rList Double_t* list of position R
/// \param nPhiSlice Int_t size of grid in Phi direction
/// \param matricesPhiValue TMatrixD** values of component Phi
/// \param phiList Double_t* list of position Phi
/// \param nZColumn Int_t size of grid in Z direction
/// \param matricesZValue TMatrixD** values of component Z
/// \param zList Double_t* list of position Z
/// \param order Int_t order of interpolation
AliTPCLookUpTable3DInterpolatorD::AliTPCLookUpTable3DInterpolatorD(
  Int_t nRRow, TMatrixD** matricesRValue, Double_t* rList,
  Int_t nPhiSlice, TMatrixD** matricesPhiValue, Double_t* phiList,
  Int_t nZColumn, TMatrixD** matricesZValue, Double_t* zList, Int_t order)
{
  fIsAllocatingLookUp = kFALSE;

  SetNR(nRRow);
  SetLookUpR(matricesRValue);
  SetRList(rList);
  SetNPhi(nPhiSlice);
  SetLookUpPhi(matricesPhiValue);
  SetPhiList(phiList);
  SetNZ(nZColumn);
  SetLookUpZ(matricesZValue);
  SetZList(zList);

  fInterpolatorR = new AliTPC3DCylindricalInterpolator();
  fInterpolatorZ = new AliTPC3DCylindricalInterpolator();
  fInterpolatorPhi = new AliTPC3DCylindricalInterpolator();

  SetOrder(order);
  fInterpolatorR->SetNR(nRRow);
  fInterpolatorR->SetNZ(nZColumn);
  fInterpolatorR->SetNPhi(nPhiSlice);
  fInterpolatorR->SetNGridPoints();
  fInterpolatorR->SetRList(rList);
  fInterpolatorR->SetZList(zList);
  fInterpolatorR->SetPhiList(phiList);
  fInterpolatorR->SetOrder(order);

  fInterpolatorZ->SetNR(nRRow);
  fInterpolatorZ->SetNZ(nZColumn);
  fInterpolatorZ->SetNPhi(nPhiSlice);
  fInterpolatorZ->SetNGridPoints();
  fInterpolatorZ->SetRList(rList);
  fInterpolatorZ->SetZList(zList);
  fInterpolatorZ->SetPhiList(phiList);
  fInterpolatorZ->SetOrder(order);

  fInterpolatorPhi->SetNR(nRRow);
  fInterpolatorPhi->SetNZ(nZColumn);
  fInterpolatorPhi->SetNPhi(nPhiSlice);
  fInterpolatorPhi->SetNGridPoints();
  fInterpolatorPhi->SetRList(rList);
  fInterpolatorPhi->SetZList(zList);
  fInterpolatorPhi->SetPhiList(phiList);
  fInterpolatorPhi->SetOrder(order);
}

/// destructor
AliTPCLookUpTable3DInterpolatorD::~AliTPCLookUpTable3DInterpolatorD()
{
  delete fInterpolatorR;
  delete fInterpolatorZ;
  delete fInterpolatorPhi;
}

/// copy from matrices to 1D array for interpolation algorithm
void AliTPCLookUpTable3DInterpolatorD::CopyFromMatricesToInterpolator()
{
  fInterpolatorR->SetValue(fLookUpR);
  fInterpolatorZ->SetValue(fLookUpZ);
  fInterpolatorPhi->SetValue(fLookUpPhi);

  if (fOrder > 2) {
    fInterpolatorR->InitCubicSpline();
    fInterpolatorZ->InitCubicSpline();
    fInterpolatorPhi->InitCubicSpline();
  }
}

/// copy from matrices to 1D array for interpolation algorithm
void AliTPCLookUpTable3DInterpolatorD::CopyFromMatricesToInterpolator(Int_t iZ)
{
  fInterpolatorR->SetValue(fLookUpR, iZ);
  fInterpolatorZ->SetValue(fLookUpZ, iZ);
  fInterpolatorPhi->SetValue(fLookUpPhi, iZ);

  // no implementation for cubic spline interpolation
}

/// get value of 3-components at a P(r,phi,z)
///
/// \param r Double_t r position
/// \param phi Double_t phi position
/// \param z Double_t z position
/// \param rValue Double_t value of r-component
/// \param phiValue Double_t value of phi-component
/// \param zValue Double_t value of z-component
void AliTPCLookUpTable3DInterpolatorD::GetValue(
  Double_t r, Double_t phi, Double_t z,
  Double_t& rValue, Double_t& phiValue, Double_t& zValue) const
{
  rValue = fInterpolatorR->GetValue(r, phi, z);
  phiValue = fInterpolatorPhi->GetValue(r, phi, z);
  zValue = fInterpolatorZ->GetValue(r, phi, z);
}

/// get value for return value is a Float_t
///
/// \param r Double_t r position
/// \param phi Double_t phi position
/// \param z Double_t z position
/// \param rValue Float_t value of r-component
/// \param phiValue Float_t value of phi-component
/// \param zValue Float_t value of z-component
void AliTPCLookUpTable3DInterpolatorD::GetValue(
  Double_t r, Double_t phi, Double_t z,
  Float_t& rValue, Float_t& phiValue, Float_t& zValue) const
{
  rValue = fInterpolatorR->GetValue(r, phi, z);
  phiValue = fInterpolatorPhi->GetValue(r, phi, z);
  zValue = fInterpolatorZ->GetValue(r, phi, z);
}

// Set Order of interpolation
//
void AliTPCLookUpTable3DInterpolatorD::SetOrder(Int_t order)
{
  fOrder = order;
  fInterpolatorR->SetOrder(order);
  fInterpolatorZ->SetOrder(order);
  fInterpolatorPhi->SetOrder(order);
}
