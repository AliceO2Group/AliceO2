/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#include "FairMCPoint.h"

#include "FairLogger.h"

FairMCPoint::FairMCPoint()
    : FairMultiLinkedData_Interface()
    , fTrackID(-1)
    , fEventId(0)
    , fPx(0.)
    , fPy(0.)
    , fPz(0.)
    , fTime(0.)
    , fLength(0.)
    , fELoss(0.)
    , fDetectorID(-1)
    , fX(0)
    , fY(0)
    , fZ(0)
{}

FairMCPoint::FairMCPoint(Int_t trackID,
                         Int_t detID,
                         TVector3 pos,
                         TVector3 mom,
                         Double_t tof,
                         Double_t length,
                         Double_t eLoss,
                         UInt_t EventId)
    : FairMultiLinkedData_Interface()
    , fTrackID(trackID)
    , fEventId(EventId)
    , fPx(mom.Px())
    , fPy(mom.Py())
    , fPz(mom.Pz())
    , fTime(tof)
    , fLength(length)
    , fELoss(eLoss)
    , fDetectorID(detID)
    , fX(pos.X())
    , fY(pos.Y())
    , fZ(pos.Z())
{}

FairMCPoint::~FairMCPoint() {}

void FairMCPoint::Print(const Option_t*) const
{
    LOG(debug) << "FairMC point for track " << fTrackID << " in detector " << fDetectorID;
    LOG(debug) << "Position (" << fX << ", " << fY << ", " << fZ << ") cm";
    LOG(debug) << "    Momentum (" << fPx << ", " << fPy << ", " << fPz << ") GeV";
    LOG(debug) << "    Time " << fTime << " ns,  Length " << fLength << " cm,  Energy loss " << fELoss * 1.0e06
               << " keV";
}

ClassImp(FairMCPoint);
