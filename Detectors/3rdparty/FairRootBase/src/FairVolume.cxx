/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----                      FairVolume source file                   -----
// -----          Created 12/01/04  by M. Al-Turany                    -----
// -------------------------------------------------------------------------

#include "FairVolume.h"

FairVolume::FairVolume()
    : TNamed()
    ,
    //    fName(""),          /**Volume Name in MC*/
    fRealName("")
    , /**Volume Name in ASCII file*/
    fVolumeId(-1)
    , /**Volume Id in GeoManager*/
    fModId(-1)
    , /**Module Id in which this volume exist*/
    fMCid(-1)
    , /**Volume Id in MC*/
    fCopyNo(-1)
    , /**Volume Copy No*/
    fMotherId(-1)
    , /**Mother Volume Id*/
    fMotherCopyNo(-1)
    , /**Mother Volume Copy No*/
    fDetector(0)
    , fModule(nullptr)
    ,              /**The Module (detector) which will proccess the hits for this volume*/
    fNode(nullptr) /**Node corre*/
{}

FairVolume::FairVolume(TString name, Int_t id, Int_t ModId, FairModule* fMod)
    : TNamed(name, name)
    ,
    //   fName(name),
    fRealName("")
    , /**Volume Name in ASCII file*/
    fVolumeId(id)
    , /**Volume Id in GeoManager*/
    fModId(ModId)
    , /**Module Id in which this volume exist*/
    fMCid(-1)
    , /**Volume Id in MC*/
    fCopyNo(-1)
    , /**Volume Copy No*/
    fMotherId(0)
    , /**Mother Volume Id*/
    fMotherCopyNo(0)
    , /**Mother Volume Copy No*/
    fDetector(0)
    , fModule(fMod)
    ,              /**The Module (detector) which will proccess the hits for this volume*/
    fNode(nullptr) /**Node corre*/
{
    if (fModule && fModule->InheritsFrom("FairDetector")) {
        fDetector = dynamic_cast<FairDetector*>(fModule);
    }
}

FairVolume::~FairVolume() {}

ClassImp(FairVolume);
