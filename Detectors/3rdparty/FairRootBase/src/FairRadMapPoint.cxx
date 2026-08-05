/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairRadMapPoint source file              -----
// -------------------------------------------------------------------------

#include "FairRadMapPoint.h"

FairRadMapPoint::FairRadMapPoint()
    : FairMCPoint()
    , fPdg(0)
    , fA(0)
    , fZmat(0)
    , fDensity(0)
    , fVolMass(0)
    , fStep(0)
    , fDose(0)
    , fDoseSL(0)
    , fXOut(0)
    , fYOut(0)
    , fZOut(0)
    , fPxOut(0)
    , fPyOut(0)
    , fPzOut(0)
{}

FairRadMapPoint::FairRadMapPoint(Int_t trackID,
                                 Int_t detID,
                                 TVector3 pos,
                                 TVector3 mom,
                                 Double_t tof,
                                 Double_t length,
                                 Double_t eLoss,
                                 TVector3 posOut,
                                 TVector3 momOut,
                                 Float_t A,
                                 Float_t Z,
                                 Float_t Density,
                                 Double_t VolMass,
                                 Double_t Step,
                                 Double_t Dose,
                                 Double_t DoseSL,
                                 Int_t Pdg)
    : FairMCPoint(trackID, detID, pos, mom, tof, length, eLoss)
    , fPdg(Pdg)
    , fA(A)
    , fZmat(Z)
    , fDensity(Density)
    , fVolMass(VolMass)
    , fStep(Step)
    , fDose(Dose)
    , fDoseSL(DoseSL)
    , fXOut(posOut.X())
    , fYOut(posOut.Y())
    , fZOut(posOut.Z())
    , fPxOut(momOut.Px())
    , fPyOut(momOut.Py())
    , fPzOut(momOut.Pz())

{}

FairRadMapPoint::~FairRadMapPoint() {}

void FairRadMapPoint::Print(const Option_t*) const
{ /**to be implimented*/
}

ClassImp(FairRadMapPoint);
