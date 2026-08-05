/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// ------------------------------------------------------------------------
// -----                  FairMCEventHeader source file                -----
// -----                                                              -----
// -----    Created 08/12/05                      V. Friese           -----
// ------------------------------------------------------------------------

#include "FairMCEventHeader.h"

#include "FairRootManager.h"

FairMCEventHeader::FairMCEventHeader()
    : TNamed("MCEvent", "MC")
    , fRunId(0)
    , fEventId(-1)
    , fX(0.)
    , fY(0.)
    , fZ(0.)
    , fT(0.)
    , fB(0.)
    , fNPrim(0)
    , fIsSet(kFALSE)
    , fRotX(0.)
    , fRotY(0.)
    , fRotZ(0.)
{}

FairMCEventHeader::FairMCEventHeader(UInt_t runId)
    : TNamed("MCEvent", "MC")
    , fRunId(runId)
    , fEventId(-1)
    , fX(0.)
    , fY(0.)
    , fZ(0.)
    , fT(0.)
    , fB(0.)
    , fNPrim(0)
    , fIsSet(kFALSE)
    , fRotX(0.)
    , fRotY(0.)
    , fRotZ(0.)
{}

FairMCEventHeader::FairMCEventHeader(Int_t iEvent,
                                     Double_t x,
                                     Double_t y,
                                     Double_t z,
                                     Double_t t,
                                     Double_t b,
                                     Int_t nPrim)
    : TNamed("MCEvent", "MC")
    , fRunId(0)
    , fEventId(iEvent)
    , fX(x)
    , fY(y)
    , fZ(z)
    , fT(t)
    , fB(b)
    , fNPrim(nPrim)
    , fIsSet(kFALSE)
    , fRotX(0.)
    , fRotY(0.)
    , fRotZ(0.)
{}

FairMCEventHeader::~FairMCEventHeader() {}

void FairMCEventHeader::Reset()
{
    fEventId = -1;
    fNPrim = 0;
    fX = fY = fZ = fT = fB = 0.;
    fIsSet = kFALSE;
    fRotX = fRotY = fRotZ = 0.;
}

void FairMCEventHeader::Register()
{
    // Dot at the end of the name is needed for splitting!!
    FairRootManager::Instance()->Register("MCEventHeader.", "Event", this, kTRUE);
}

ClassImp(FairMCEventHeader);
