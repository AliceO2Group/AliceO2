/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// ********************************************* //
// ***        D. Kresan   2004-Sep-14        *** //
// ***        D.Kresan@gsi.de                *** //
// ********************************************* //

#include "FairTrajFilter.h"

#include "FairRootManager.h"   // for FairRootManager

#include <TClonesArray.h>   // for TClonesArray
#include <TError.h>         // for Fatal
#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TMath.h>       // for Pi, TwoPi, Log
#include <TMathBase.h>   // for Abs
#include <TParticle.h>   // for TParticle
#include <iostream>      // for operator<<, basic_ostream, etc

using namespace std;

ClassImp(FairTrajFilter);

TMCThreadLocal FairTrajFilter* FairTrajFilter::fgInstance = nullptr;

FairTrajFilter* FairTrajFilter::Instance() { return fgInstance; }

FairTrajFilter::FairTrajFilter()
    : fVxMin(-2000.)
    , fVxMax(2000.)
    , fVyMin(-2000.)
    , fVyMax(2000.)
    , fVzMin(-2000.)
    , fVzMax(2000.)
    , fPMin(0.)
    , fPMax(1e10)
    , fThetaMin(0.)
    , fThetaMax(TMath::Pi())
    , fPhiMin(0.)
    , fPhiMax(TMath::TwoPi())
    , fPxMin(-1e10)
    , fPxMax(1e10)
    , fPyMin(-1e10)
    , fPyMax(1e10)
    , fPzMin(-1e10)
    , fPzMax(1e10)
    , fPtMin(0.)
    , fPtMax(1e10)
    , fRapidityMin(-1e10)
    , fRapidityMax(1e10)
    , fKinCutType(0)
    ,   // polar system by default
    fEtotMin(0.)
    , fEtotMax(1e10)
    , fStorePrim(kTRUE)
    , fStoreSec(kTRUE)
    , fStepSizeMin(0.1)
    ,   // 1mm by default
    fTrackCollection(new TClonesArray("TGeoTrack"))
    , fCurrentTrk(nullptr)
{
    if (nullptr != fgInstance) {
        Fatal("FairTrajFilter", "Singleton class already exists.");
        return;
    }

    fgInstance = this;
}

FairTrajFilter::~FairTrajFilter() { fgInstance = nullptr; }

void FairTrajFilter::Init(TString brName, TString folderName)
{
    FairRootManager::Instance()->Register(brName.Data(), folderName.Data(), fTrackCollection, kTRUE);
}

void FairTrajFilter::Reset() { fTrackCollection->Delete(); }

Bool_t FairTrajFilter::IsAccepted(const TParticle* p) const
{
    if (nullptr == p) {
        return kFALSE;
    }

    // Apply vertex cut
    if ((p->Vx() < fVxMin) || (p->Vx() > fVxMax) || (p->Vy() < fVyMin) || (p->Vy() > fVyMax) || (p->Vz() < fVzMin)
        || (p->Vz() > fVzMax)) {
        return kFALSE;
    }

    // Apply cut on kinematics
    if (0 == fKinCutType) {
        if ((p->P() < fPMin) || (p->P() > fPMax) || (p->Theta() < fThetaMin) || (p->Theta() > fThetaMax)
            || (p->Phi() < fPhiMin) || (p->Phi() > fPhiMax)) {
            return kFALSE;
        }
    } else if (1 == fKinCutType) {
        if ((p->Px() < fPxMin) || (p->Px() > fPxMax) || (p->Py() < fPyMin) || (p->Py() > fPyMax) || (p->Pz() < fPzMin)
            || (p->Pz() > fPzMax)) {
            return kFALSE;
        }
    } else {
        Double_t rapidity = 0.5 * TMath::Log((p->Energy() + p->Pz()) / (p->Energy() - p->Pz()));
        if ((p->Pt() < fPtMin) || (p->Pt() > fPtMax) || (rapidity < fRapidityMin) || (rapidity > fRapidityMax)) {
            return kFALSE;
        }
    }

    // Apply energy cut
    if ((p->Energy() < fEtotMin) || (p->Energy() > fEtotMax)) {
        return kFALSE;
    }

    // Apply generation cut
    if (-1 == p->GetFirstMother()) {
        if (kFALSE == fStorePrim) {
            return kFALSE;
        }
    } else {
        if (kFALSE == fStoreSec) {
            return kFALSE;
        }
    }

    return kTRUE;
}

void FairTrajFilter::SetVertexCut(Double_t vxMin,
                                  Double_t vyMin,
                                  Double_t vzMin,
                                  Double_t vxMax,
                                  Double_t vyMax,
                                  Double_t vzMax)
{
    TGeoBBox* cave = static_cast<TGeoBBox*>(gGeoManager->GetTopVolume()->GetShape());
    cave->ComputeBBox();
    Double_t caveX = 2. * cave->GetDX();
    Double_t caveY = 2. * cave->GetDY();
    Double_t caveZ = 2. * cave->GetDZ();
    if ((vxMax < vxMin) || (vyMax < vyMin) || (vzMax < vzMin) || (TMath::Abs(vxMin) > caveX)
        || (TMath::Abs(vxMax) > caveX) || (TMath::Abs(vyMin) > caveY) || (TMath::Abs(vyMax) > caveY)
        || (TMath::Abs(vzMin) > caveZ) || (TMath::Abs(vzMax) > caveZ)) {
        cout << "-E- FairTrajFilter::SetVertexCut() : invalid region, ignoring." << endl;
        return;
    }
    fVxMin = vxMin;
    fVxMax = vxMax;
    fVyMin = vyMin;
    fVyMax = vyMax;
    fVzMin = vzMin;
    fVzMax = vzMax;
}

void FairTrajFilter::SetMomentumCutP(Double_t pMin,
                                     Double_t thetaMin,
                                     Double_t phiMin,
                                     Double_t pMax,
                                     Double_t thetaMax,
                                     Double_t phiMax)
{
    if ((pMax < pMin) || (thetaMax < thetaMin) || (phiMax < phiMin) || (pMin < 0.) || (pMax < 0.) || (thetaMin < 0.)
        || (thetaMax > TMath::Pi()) || (phiMin < 0.) || (phiMax > TMath::TwoPi())) {
        cout << "-E- FairTrajFilter::SetMomentumCutP() : invalid region, ignoring." << endl;
        return;
    }
    fPMin = pMin;
    fPMax = pMax;
    fThetaMin = thetaMin;
    fThetaMax = thetaMax;
    fPhiMin = phiMin;
    fPhiMax = phiMax;
    fKinCutType = 0;
}

void FairTrajFilter::SetMomentumCutD(Double_t pxMin,
                                     Double_t pyMin,
                                     Double_t pzMin,
                                     Double_t pxMax,
                                     Double_t pyMax,
                                     Double_t pzMax)
{
    if ((pxMax < pxMin) || (pyMax < pyMin) || (pzMax < pzMin)) {
        cout << "-E- FairTrajFilter::SetMomentumCutD() : invalid region, ignoring." << endl;
        return;
    }
    fPxMin = pxMin;
    fPxMax = pxMax;
    fPyMin = pyMin;
    fPyMax = pyMax;
    fPzMin = pzMin;
    fPzMax = pzMax;
    fKinCutType = 1;
}

void FairTrajFilter::SetPtRapidityCut(Double_t ptMin, Double_t ptMax, Double_t rapidityMin, Double_t rapidityMax)
{
    if ((ptMax < ptMin) || (rapidityMax < rapidityMin) || (ptMin < 0.) || (ptMax < 0.)) {
        cout << "-E- FairTrajFilter::SetPtRapidityCut() : invalid region, ignoring." << endl;
        return;
    }
    fPtMin = ptMin;
    fPtMax = ptMax;
    fRapidityMin = rapidityMin;
    fRapidityMax = rapidityMax;
    fKinCutType = 2;
}

void FairTrajFilter::SetEnergyCut(Double_t etotMin, Double_t etotMax)
{
    if ((etotMax < etotMin) || (etotMin < 0.) || (etotMax < 0.)) {
        cout << "-E- FairTrajFilter::SetEnergyCut() : invalid region, ignoring." << endl;
        return;
    }
    fEtotMin = etotMin;
    fEtotMax = etotMax;
}

void FairTrajFilter::SetStepSizeCut(Double_t stepSizeMin)
{
    if (stepSizeMin < 0.) {
        cout << "-E- FairTrajFilter::SetStepSizeCut() : invalid value, ignoring." << endl;
        return;
    }
    fStepSizeMin = stepSizeMin;
}

void FairTrajFilter::GetVertexCut(Double_t& vxMin,
                                  Double_t& vyMin,
                                  Double_t& vzMin,
                                  Double_t& vxMax,
                                  Double_t& vyMax,
                                  Double_t& vzMax) const
{
    vxMin = fVxMin;
    vxMax = fVxMax;
    vyMin = fVyMin;
    vyMax = fVyMax;
    vzMin = fVzMin;
    vzMax = fVzMax;
}

void FairTrajFilter::GetMomentumCutP(Double_t& pMin,
                                     Double_t& thetaMin,
                                     Double_t& phiMin,
                                     Double_t& pMax,
                                     Double_t& thetaMax,
                                     Double_t& phiMax) const
{
    pMin = fPMin;
    pMax = fPMax;
    thetaMin = fThetaMin;
    thetaMax = fThetaMax;
    phiMin = fPhiMin;
    phiMax = fPhiMax;
}

void FairTrajFilter::GetMomentumCutD(Double_t& pxMin,
                                     Double_t& pyMin,
                                     Double_t& pzMin,
                                     Double_t& pxMax,
                                     Double_t& pyMax,
                                     Double_t& pzMax) const
{
    pxMin = fPxMin;
    pxMax = fPxMax;
    pyMin = fPyMin;
    pyMax = fPyMax;
    pzMin = fPzMin;
    pzMax = fPzMax;
}

void FairTrajFilter::GetPtRapidityCut(Double_t& ptMin,
                                      Double_t& ptMax,
                                      Double_t& rapidityMin,
                                      Double_t& rapidityMax) const
{
    ptMin = fPtMin;
    ptMax = fPtMax;
    rapidityMin = fRapidityMin;
    rapidityMax = fRapidityMax;
}

void FairTrajFilter::GetEnergyCut(Double_t& etotMin, Double_t& etotMax) const
{
    etotMin = fEtotMin;
    etotMax = fEtotMax;
}

TGeoTrack* FairTrajFilter::AddTrack(Int_t trackId, Int_t pdgCode)
{

    //  cout << "FairTrajFilter::AddTrack" << endl;
    TClonesArray& clref = *fTrackCollection;
    Int_t tsize = clref.GetEntriesFast();
    fCurrentTrk = new (clref[tsize]) TGeoTrack(trackId, pdgCode);
    return fCurrentTrk;
}

TGeoTrack* FairTrajFilter::AddTrack(TParticle* p)
{
    Int_t trackId = 0;
    //  cout << "FairTrajFilter::AddTrack" << endl;
    if (fCurrentTrk) {
        trackId = fCurrentTrk->GetId();
    }
    Int_t pdgCode = p->GetPdgCode();
    TClonesArray& clref = *fTrackCollection;
    Int_t tsize = clref.GetEntriesFast();
    fCurrentTrk = new (clref[tsize]) TGeoTrack(++trackId, pdgCode, 0, p);
    return fCurrentTrk;
}

TGeoTrack* FairTrajFilter::CheckAddTrack(Int_t trackId, TParticle* p)
{
    TClonesArray& clref = *fTrackCollection;
    Int_t tsize = clref.GetEntriesFast();
    for (Int_t itr = tsize - 1; itr >= 0; --itr) {
        TGeoTrack* tempTrack = dynamic_cast<TGeoTrack*>(clref.At(itr));
        if (tempTrack->GetId() == trackId) {
            fCurrentTrk = tempTrack;
            return fCurrentTrk;
        }
    }
    // track with given trackId not found, creating new one
    Int_t pdgCode = p->GetPdgCode();
    fCurrentTrk = new (clref[tsize]) TGeoTrack(trackId, pdgCode, 0, p);
    return fCurrentTrk;
}

TGeoTrack* FairTrajFilter::GetTrack(Int_t trackId) { return static_cast<TGeoTrack*>(fTrackCollection->At(trackId)); }
