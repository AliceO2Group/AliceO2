/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIR_VOLUME_H
#define FAIR_VOLUME_H

#include "FairDetector.h"
#include "FairModule.h"

#include <Rtypes.h>    // for Int_t, FairVolume::Class, etc
#include <TNamed.h>    // for TNamed
#include <TString.h>   // for TString
class FairGeoNode;

/**
 * This Object is only used for internal book keeping!
 * @author M. Al-Turany, D. Bertini
 * @version 0.1
 * @since 12.01.04
 */

class FairVolume : public TNamed
{
  public:
    FairVolume();
    FairVolume(TString name, Int_t id = 0, Int_t detid = 0, FairModule* fMod = 0);
    virtual ~FairVolume();

    //    const char*  GetName() { return fName.Data();}
    //    TString getName() { return fName;}

    void setRealName(TString name) { fRealName = name; }
    const char* getRealName() { return fRealName.Data(); }
    Int_t getVolumeId() { return fVolumeId; }
    Int_t getModId() { return fModId; }
    void setModId(Int_t id) { fModId = id; }
    void setCopyNo(Int_t id) { fCopyNo = id; }
    void setVolumeId(Int_t id) { fVolumeId = id; }
    void setGeoNode(FairGeoNode* d) { fNode = d; }
    void setMotherId(Int_t fM) { fMotherId = fM; }
    void setMotherCopyNo(Int_t CopyNo) { fMotherCopyNo = CopyNo; }

    FairModule* GetModule() { return fModule; }
    FairDetector* GetDetector() { return fDetector; }
    void SetModule(FairModule* mod)
    {
        fModule = mod;
        if (mod->InheritsFrom("FairDetector")) {
            fDetector = dynamic_cast<FairDetector*>(mod);
        }
    }

    Int_t getMCid() { return fMCid; }
    Int_t getCopyNo() { return fCopyNo; }
    void setMCid(Int_t id) { fMCid = id; }
    FairGeoNode* getGeoNode() { return fNode; }
    Int_t getMotherId() { return fMotherId; }
    Int_t getMotherCopyNo() { return fMotherCopyNo; }

  private:
    FairVolume(const FairVolume&);
    FairVolume& operator=(const FairVolume&);

    //    TString fName;   /**Volume Name in MC*/
    TString fRealName;       /**Volume Name in ASCII file*/
    Int_t fVolumeId;         /**Volume Id in GeoManager*/
    Int_t fModId;            /**Module Id in which this volume exist*/
    Int_t fMCid;             /**Volume Id in MC*/
    Int_t fCopyNo;           /**Volume Copy No*/
    Int_t fMotherId;         /**Mother Volume Id*/
    Int_t fMotherCopyNo;     /**Mother Volume Copy No*/
    FairDetector* fDetector; /** The Detector which will proccess the hits for this volume*/
    FairModule* fModule;     /**The Module in which the volume is */
    FairGeoNode* fNode;      /**Node corresponding to this volume*/

    ClassDef(FairVolume, 2);
};

#endif
