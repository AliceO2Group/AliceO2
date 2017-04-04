#include "TClonesArray.h"

#include "FairVolume.h"
#include "FairRootManager.h"

#include "EMCALBase/Point.h"
#include "EMCALSimulation/Detector.h"

using namespace AliceO2::EMCAL;

ClassImp(Detector)

Detector::Detector(const char* Name, Bool_t Active):
AliceO2::Base::Detector(Name, Active),
mPointCollection(new TClonesArray("AliceO2::EMCAL::Point"))
{
}

Detector::~Detector()
{
}

void Detector::Initialize(){
}

Bool_t Detector::ProcessHits( FairVolume* v) {
  return true;
}

Point *Detector::AddHit(Int_t shunt, Int_t trackID, Int_t parentID, Int_t primary, Double_t initialEnergy,
              Int_t detID, Double_t *pos, Double_t *mom, Double_t time, Double_t eLoss){
  
  TClonesArray& refCollection = *mPointCollection;
  Int_t size = refCollection.GetEntriesFast();
  return new(refCollection[size]) Point(shunt, primary, trackID, parentID, detID, initialEnergy, pos, mom,
                                time, eLoss);

}

void Detector::Register(){
  FairRootManager::Instance()->Register("EMCALPoint", "EMCAL", mPointCollection, kTRUE);
}

TClonesArray* Detector::GetCollection(Int_t iColl) const {
  if(iColl == 0) return mPointCollection;
  return nullptr;
}

void Detector::Reset() {
}


void Detector::CreateMaterials() {

}

