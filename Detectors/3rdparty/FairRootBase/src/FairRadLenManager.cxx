/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairRadLenManager source file                   -----
// -----                  Created 14/01/08  by M. Al-Turany            -----
// -------------------------------------------------------------------------

#include "FairRadLenManager.h"

#include "FairRadLenPoint.h"   // for FairRadLenPoint
#include "FairRootManager.h"   // for FairRootManager

#include <TClonesArray.h>      // for TClonesArray
#include <TVector3.h>          // for TVector3
#include <TVirtualMC.h>        // for TVirtualMC
#include <TVirtualMCStack.h>   // for TVirtualMCStack

using namespace std;

ClassImp(FairRadLenManager);

FairRadLenManager* FairRadLenManager::fgInstance = nullptr;

FairRadLenManager* FairRadLenManager::Instance()
{
    /**singelton instance*/
    return fgInstance;
}

FairRadLenManager::FairRadLenManager()
    : fPointCollection(new TClonesArray("FairRadLenPoint"))
    , fTrackID(0)
    , fVolumeID(0)
    , fPosIn(TLorentzVector(0, 0, 0, 0))
    , fPosOut(TLorentzVector(0, 0, 0, 0))
    , fMomIn(TLorentzVector(0, 0, 0, 0))
    , fMomOut(TLorentzVector(0, 0, 0, 0))
    , fTime(0)
    , fLength(0)
    , fELoss(0)
    , fA(0)
    , fZmat(0)
    , fDensity(0)
    , fRadl(0)
    , fAbsl(0)
{
    /** radiation length default ctor */
    if (nullptr == fgInstance) {
        fgInstance = this;
        //    fPointCollection=new TClonesArray("FairRadLenPoint");
    }
}

FairRadLenManager::~FairRadLenManager()
{
    /** radiation length default dtor */
    fgInstance = nullptr;
    fPointCollection->Delete();
    delete fPointCollection;
}

void FairRadLenManager::Init()
{
    /**create the branch for output */
    FairRootManager::Instance()->Register("RadLen", "RadLenPoint", fPointCollection, kTRUE);
}

void FairRadLenManager::Reset()
{
    /**We have to free the momeory, Clear() is faster but not enough! */
    fPointCollection->Delete();
}

void FairRadLenManager::AddPoint(Int_t& ModuleId)
{
    /**Add a point to the collection*/
    if (TVirtualMC::GetMC()->IsTrackEntering()) {
        fELoss = 0.;
        Int_t copyNo;
        fVolumeID = TVirtualMC::GetMC()->CurrentVolID(copyNo);
        fTime = TVirtualMC::GetMC()->TrackTime() * 1.0e09;
        fLength = TVirtualMC::GetMC()->TrackLength();
        TVirtualMC::GetMC()->TrackPosition(fPosIn);
        TVirtualMC::GetMC()->TrackMomentum(fMomIn);
        //    Int_t MatId=  TVirtualMC::GetMC()->CurrentMaterial(fA, fZmat, fDensity, fRadl, fAbsl);
        TVirtualMC::GetMC()->CurrentMaterial(fA, fZmat, fDensity, fRadl, fAbsl);
    }
    /** Sum energy loss for all steps in the active volume */
    fELoss += TVirtualMC::GetMC()->Edep();
    /**  Create a point at exit of the volume */
    if (TVirtualMC::GetMC()->IsTrackExiting() || TVirtualMC::GetMC()->IsTrackStop()
        || TVirtualMC::GetMC()->IsTrackDisappeared()) {
        //    FairRadLenPoint* p=0;
        fTrackID = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
        TVirtualMC::GetMC()->TrackPosition(fPosOut);
        TVirtualMC::GetMC()->TrackMomentum(fMomOut);
        TClonesArray& clref = *fPointCollection;
        Int_t tsize = clref.GetEntriesFast();
        //    p=new(clref[tsize]) FairRadLenPoint(fTrackID, ModuleId,
        new (clref[tsize]) FairRadLenPoint(fTrackID,
                                           ModuleId,
                                           TVector3(fPosIn.X(), fPosIn.Y(), fPosIn.Z()),
                                           TVector3(fMomIn.X(), fMomIn.Y(), fMomIn.Z()),
                                           fTime,
                                           fLength,
                                           fELoss,
                                           TVector3(fPosOut.X(), fPosOut.Y(), fPosOut.Z()),
                                           TVector3(fMomOut.X(), fMomOut.Y(), fMomOut.Z()),
                                           fA,
                                           fZmat,
                                           fDensity,
                                           fRadl);
    }
}
