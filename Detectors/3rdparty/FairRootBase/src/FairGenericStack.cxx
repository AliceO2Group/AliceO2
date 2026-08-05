/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                       FairGenericStack source file                    -----
// -----             Created 10/08/04  by D. Bertini                   -----
// -------------------------------------------------------------------------
#include "FairGenericStack.h"

#include "FairLogger.h"   // for FairLogger

#include <TGeoManager.h>
#include <TLorentzVector.h>
#include <TMCProcess.h>
#include <TRefArray.h>
#include <TString.h>
#include <TVirtualMC.h>
#include <cstring>   // strcmp

FairGenericStack::FairGenericStack()
    : TVirtualMCStack()
    , fLogger(FairLogger::GetLogger())
    , fDetList(0)
    , fDetIter(0)
    , fVerbose(1)
    , fFSTrackMap()
    , fFSTrackIter()
    , fFSMovedIndex(-2)
    , fFSFirstSecondary(-2)
    , fFSNofSecondaries(0)
{}

// -----   Constructor with estimated array dimension   --------------------
FairGenericStack::FairGenericStack(Int_t)
    : TVirtualMCStack()
    , fLogger(FairLogger::GetLogger())
    , fDetList(0)
    , fDetIter(0)
    , fVerbose(1)
    , fFSTrackMap()
    , fFSTrackIter()
    , fFSMovedIndex(-2)
    , fFSFirstSecondary(-2)
    , fFSNofSecondaries(0)
{}

FairGenericStack::~FairGenericStack() { delete fDetIter; }

FairGenericStack::FairGenericStack(const FairGenericStack& rhs)
    : TVirtualMCStack(rhs)
    , fLogger(FairLogger::GetLogger())
    , fDetList(rhs.fDetList)
    , fDetIter(0)
    , fVerbose(rhs.fVerbose)
{}

FairGenericStack& FairGenericStack::operator=(const FairGenericStack& rhs)
{
    // check assignment to self
    if (this == &rhs)
        return *this;

    // base class assignment
    TVirtualMCStack::operator=(rhs);

    // assignment oiperator
    fLogger = 0;
    fDetList = 0;
    fDetIter = 0;
    fVerbose = rhs.fVerbose;

    return *this;
}

void FairGenericStack::SetDetArrayList(TRefArray* detArray)
{
    fDetList = detArray;
    if (fDetList != 0) {
        fDetIter = fDetList->MakeIterator();
    }
}

FairGenericStack* FairGenericStack::CloneStack() const
{
    Fatal("CloneStack", "Has to be overriden in multi-threading applications.");
    return 0;
}

void FairGenericStack::FastSimMoveParticleTo(Double_t xx,
                                             Double_t yy,
                                             Double_t zz,
                                             Double_t tt,
                                             Double_t px,
                                             Double_t py,
                                             Double_t pz,
                                             Double_t en)
{
    TLorentzVector curPos;
    TVirtualMC::GetMC()->TrackPosition(curPos);
    LOG(debug) << "track is in " << curPos.X() << "," << curPos.Y() << "," << curPos.Z() << ", moving to " << xx << ","
               << yy << "," << zz;
    TString curVolName(TVirtualMC::GetMC()->CurrentVolName());
    TString targetVolName(gGeoManager->FindNode(xx, yy, zz)->GetVolume()->GetName());
    if (curVolName.EqualTo(targetVolName)) {
        LOG(fatal) << "FairStack::FastSimMoveParticleTo(" << xx << "," << yy << "," << zz << ": " << curVolName << " = "
                   << targetVolName << ") crashes the simulation.";
    } else {
        LOG(debug) << "gMC says track is in \"" << curVolName << "\" moving particle to \"" << targetVolName << "\".";
    }

    Int_t tobedone = 1;
    Int_t parent = 0;   // do not store it as mother
    Int_t pdg = TVirtualMC::GetMC()->TrackPid();
    Double_t polx = 0.;
    Double_t poly = 0.;
    Double_t polz = 0.;
    TMCProcess proc = kPPrimary;   // not important, the track will not be saved
    Int_t ntr = 0;                 // Track number; to be filled by the stack
    Int_t status = 0;              // not important, the track will not be saved
    Double_t weight = 0.;

    PushTrack(tobedone, parent, pdg, px, py, pz, en, xx, yy, zz, tt, polx, poly, polz, proc, ntr, weight, status, -1);
    fFSMovedIndex = GetListOfParticles()->GetEntries() - 1;
    Int_t trackID = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
    fFSTrackIter = fFSTrackMap.find(trackID);   // check if this track is not already created by FastSimulation
    if (fFSTrackIter != fFSTrackMap.end())      // indeed the track has been created by the FastSimulation mechanism
        trackID = fFSTrackIter->second;         // use the ID of the original track
    fFSTrackMap[ntr] = trackID;

    LOG(debug) << "FairStack::FastSimMoveParticleTo() created track number " << ntr << " to replace track number "
               << trackID;

    if (strcmp(TVirtualMC::GetMC()->GetName(), "TGeant3TGeo") == 0) {
        TVirtualMC::GetMC()->StopTrack();
    }
}

void FairGenericStack::FastSimStopParticle()
{
    if (TVirtualMC::GetMC()->IsTrackStop())
        LOG(fatal) << "FairStack::FastSimStopParticle() tries to stop particle that was probably moved!";
    fFSMovedIndex = -1;
    TVirtualMC::GetMC()->StopTrack();
}

void FairGenericStack::FastSimPushSecondary(Int_t parentID,
                                            Int_t pdgCode,
                                            Double_t xx,
                                            Double_t yy,
                                            Double_t zz,
                                            Double_t tt,
                                            Double_t px,
                                            Double_t py,
                                            Double_t pz,
                                            Double_t en,
                                            Double_t polx,
                                            Double_t poly,
                                            Double_t polz,
                                            TMCProcess proc,
                                            Double_t weight,
                                            Int_t is)
{
    Int_t tobedone = 1;

    Int_t ntr = 0;   // Track number; to be filled by the stack
    PushTrack(tobedone, parentID, pdgCode, px, py, pz, en, xx, yy, zz, tt, polx, poly, polz, proc, ntr, weight, is, -1);
    if (fFSNofSecondaries == 0)
        fFSFirstSecondary = GetListOfParticles()->GetEntries() - 1;
    fFSNofSecondaries++;
}

ClassImp(FairGenericStack);
