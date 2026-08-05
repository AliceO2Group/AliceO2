/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                    FairDetector source file                   -----
// -----            Created 06/01/04  by M. Al-Turany/ D. Bertini      -----
// -------------------------------------------------------------------------

#include "FairDetector.h"

#include "FairGeoNode.h"   // for FairGeoNode
#include "FairLogger.h"    // for FairLogger, MESSAGE_ORIGIN
#include "FairModule.h"    // for FairModule::svList, etc
#include "FairRootManager.h"
#include "FairVolume.h"   // for FairVolume

#include <TFolder.h>       // for TFolder
#include <TGeoManager.h>   // for gGeoManager
#include <TList.h>         // for TList
#include <TObject.h>       // for TObject
#include <TROOT.h>         // for TROOT, gROOT
#include <TRefArray.h>     // for TRefArray
#include <TString.h>       // for TString
#include <TVirtualMC.h>    // for TVirtualMC

FairDetector::FairDetector(const char* Name, Bool_t Active, Int_t DetId)
    : FairModule(Name, "FAIR Detector", Active)
    , fDetId(DetId)
    , fLogger(FairLogger::GetLogger())
{
    flGeoPar = new TList();
    TString lname(GetName());
    lname += "GeoPar";
    flGeoPar->SetName(lname.Data());
    fGeoSaved = kFALSE;
}

FairDetector::FairDetector(const FairDetector& rhs)
    : FairModule(rhs)
    , fDetId(rhs.fDetId)
    , fLogger(rhs.fLogger)
{}

FairDetector::~FairDetector() { delete flGeoPar; }

FairDetector& FairDetector::operator=(const FairDetector& rhs)
{
    // check assignment to self
    if (this == &rhs)
        return *this;

    // base class assignment
    FairModule::operator=(rhs);

    // assignment operator
    fDetId = rhs.fDetId;
    fLogger = rhs.fLogger;

    return *this;
}

FairDetector::FairDetector()
    : fDetId(0)
    , fLogger(FairLogger::GetLogger())
{}

// -------------------------------------------------------------------------

void FairDetector::DefineSensitiveVolumes()
{
    LOG(info) << "FairDetector::DefineSensitiveVolumes";
    TObjArray* volumes = gGeoManager->GetListOfVolumes();
    TIter next(volumes);
    TGeoVolume* volume;
    while ((volume = static_cast<TGeoVolume*>(next()))) {
        if (IsSensitive(volume->GetName())) {
            LOG(debug) << "Sensitive Volume " << volume->GetName();
            AddSensitiveVolume(volume);
        }
    }
}

// -------------------------------------------------------------------------

void FairDetector::Initialize()
{
    // Registers hits collection in Root manager;
    // sets sensitive volumes.
    // ---

    // Define sensitive volumes if in MT
    if (gMC->IsMT()) {
        std::cout << "Define sensitive volume " << std::endl;
        DefineSensitiveVolumes();
    }

    Int_t NoOfEntries = svList->GetEntries();
    Int_t fMCid;
    FairGeoNode* fN;
    TString cutName;
    TString copysign = "#";
    for (Int_t i = 0; i < NoOfEntries; i++) {
        FairVolume* aVol = static_cast<FairVolume*>(svList->At(i));
        cutName = aVol->GetName();
        Ssiz_t pos = cutName.Index(copysign, 1);
        if (pos > 1) {
            cutName.Resize(pos);
        }
        if (aVol->getModId() == GetModId()) {
            fMCid = TVirtualMC::GetMC()->VolId(cutName.Data());
            aVol->setMCid(fMCid);
            fN = aVol->getGeoNode();
            if (fN) {
                fN->setMCid(fMCid);
            }
        }
    }

    // Initialize cached pointer to MC (on master in sequential mode)
    fMC = TVirtualMC::GetMC();
}

void FairDetector::SaveGeoParams()
{
    if (!fGeoSaved) {
        LOG(info) << "Detector: " << GetName() << " Geometry parameters saved ... ";
        TFolder* mf = dynamic_cast<TFolder*>(gROOT->FindObjectAny(FairRootManager::GetFolderName()));
        TFolder* stsf = nullptr;
        if (mf) {
            stsf = dynamic_cast<TFolder*>(mf->FindObjectAny(GetName()));
        }
        if (stsf) {
            TFolder* newf = stsf->AddFolder("Parameters", "Detector parameters", nullptr);
            newf->Add(flGeoPar);
        }
        fGeoSaved = kTRUE;
    }
}

ClassImp(FairDetector);
