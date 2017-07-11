// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TClonesArray.h"

#include "FairVolume.h"
#include "FairRootManager.h"

#include "EMCALBase/Hit.h"
#include "EMCALSimulation/Detector.h"

using namespace o2::EMCAL;

ClassImp(Detector)

Detector::Detector(const char* Name, Bool_t Active):
o2::Base::Detector(Name, Active),
mPointCollection(new TClonesArray("o2::EMCAL::Hit"))
{
}

void Detector::Initialize(){
}

Bool_t Detector::ProcessHits( FairVolume* v) {
  return true;
}

Hit *Detector::AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy,
              Int_t detID, const Point3D<float> &pos, const Vector3D<float> &mom, Double_t time, Double_t eLoss){
  
  TClonesArray& refCollection = *mPointCollection;
  Int_t size = refCollection.GetEntriesFast();
  return new(refCollection[size]) Hit(shunt, primary, trackID, parentID, detID, initialEnergy, pos, mom, time, eLoss);

}

void Detector::Register(){
  FairRootManager::Instance()->Register("EMCALHit", "EMCAL", mPointCollection, kTRUE);
}

TClonesArray* Detector::GetCollection(Int_t iColl) const {
  if(iColl == 0) return mPointCollection;
  return nullptr;
}

void Detector::Reset() {
}


void Detector::CreateMaterials() {

}

