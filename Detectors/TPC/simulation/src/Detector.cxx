#include "Detector.h"
#include "Point.h"              // for Point

#include "DataFormats/simulation/include/DetectorList.h"  // for DetectorId::kAliTpc
#include "DataFormats/simulation/include/Stack.h"         // for Stack

#include "FairRootManager.h"    // for FairRootManager
#include "FairVolume.h"         // for FairVolume

#include "TClonesArray.h"       // for TClonesArray
#include "TVirtualMC.h"         // for TVirtualMC, gMC
#include "TVirtualMCStack.h"    // for TVirtualMCStack

#include <stddef.h>             // for NULL

#include "FairGeoVolume.h"
#include "FairGeoNode.h"
#include "FairGeoLoader.h"
#include "FairGeoInterface.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairLogger.h"

#include "Data/DetectorList.h"
#include "Data/Stack.h"

#include "TSystem.h"
#include "TClonesArray.h"
#include "TVirtualMC.h"

// geo stuff
#include "TGeoManager.h"
#include "TGeoGlobalMagField.h"
#include "TGeoVolume.h"
#include "TGeoPcon.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoPgon.h"
#include "TGeoTrd1.h"
#include "TGeoCompositeShape.h"
#include "TGeoPara.h"
#include "TGeoPhysicalNode.h"
#include "TGeoHalfSpace.h"
#include "TGeoArb8.h"
#include "TGeoMatrix.h"


// dirty stuff
#include "AliTPCParam.h"
#include "Manager.h"
#include "Condition.h"

#include <iostream>
using std::cout;
using std::endl;
using std::ios_base;

using namespace AliceO2::TPC;

Detector::Detector()
  : AliceO2::Base::Detector("TPC", kTRUE, kAliTpc),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mPosition(),
    mMomentum(),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mPointCollection(new TClonesArray("AliceO2::TPC::Point")),
    mSens(0),
    mParam(0x0)
{
}

Detector::Detector(const char* name, Bool_t active)
  : AliceO2::Base::Detector(name, active, kAliTpc),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mPosition(),
    mMomentum(),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mPointCollection(new TClonesArray("AliceO2::TPC::Point")),
    mParam(0x0)
{
  //TODO: Change this at some point
  AliceO2::CDB::Condition* tpcParametersCondition = AliceO2::CDB::Manager::Instance()->getObject("TPC/Calib/Parameters");
  if (tpcParametersCondition) {
    mParam = dynamic_cast<AliTPCParam*>(tpcParametersCondition->getObject());
  }
  if (!mParam) {
    LOG(ERROR) << "Could not load TPC Parameters" << FairLogger::endl;
  }

}

Detector::~Detector()
{
  if (mPointCollection) {
    mPointCollection->Delete();
    delete mPointCollection;
  }
}

void Detector::Initialize()
{
    AliceO2::Base::Detector::Initialize();
//     LOG(INFO) << "Initialize" << FairLogger::endl;
}

Bool_t  Detector::ProcessHits(FairVolume* vol)
{
  /** This method is called from the MC stepping */
//   LOG(INFO) << "TPC::ProcessHits" << FairLogger::endl;
  //Set parameters at entrance of volume. Reset ELoss.
  if ( TVirtualMC::GetMC()->IsTrackEntering() ) {
    mEnergyLoss  = 0.;
    mTime   = TVirtualMC::GetMC()->TrackTime() * 1.0e09;
    mLength = TVirtualMC::GetMC()->TrackLength();
    TVirtualMC::GetMC()->TrackPosition(mPosition);
    TVirtualMC::GetMC()->TrackMomentum(mMomentum);
  }

  // Sum energy loss for all steps in the active volume
  mEnergyLoss += TVirtualMC::GetMC()->Edep();

  // Create DetectorPoint at exit of active volume
  if ( TVirtualMC::GetMC()->IsTrackExiting()    ||
       TVirtualMC::GetMC()->IsTrackStop()       ||
       TVirtualMC::GetMC()->IsTrackDisappeared()   ) {
    mTrackNumberID  = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
    mVolumeID = vol->getMCid();
    if (mEnergyLoss == 0. ) { return kFALSE; }
    AddHit(mTrackNumberID, mVolumeID, TVector3(mPosition.X(),  mPosition.Y(),  mPosition.Z()),
           TVector3(mMomentum.Px(), mMomentum.Py(), mMomentum.Pz()), mTime, mLength,
           mEnergyLoss);
//     LOG(INFO) << "TPC::AddHit" << FairLogger::endl
//     << "   -- " << mTrackNumberID <<","  << mVolumeID
//     << ", Pos: (" << mPosition.X() << ", "  << mPosition.Y() <<", "<<  mPosition.Z() << ") "
//     << ", Mom: (" << mMomentum.Px() << ", " << mMomentum.Py() << ", "  <<  mMomentum.Pz() << ") "
//     << " Time: "<<  mTime <<", Len: " << mLength << ", Eloss: " <<
//     mEnergyLoss << FairLogger::endl;

    // Increment number of Detector det points in TParticle
    AliceO2::Data::Stack* stack = (AliceO2::Data::Stack*)TVirtualMC::GetMC()->GetStack();
    stack->AddPoint(kAliTpc);

  }

  return kTRUE;
}


// Bool_t  Detector::ProcessHits(FairVolume* vol)
// {
//   //
//   // Called for every step in the Time Projection Chamber
//   //
//
//   //
//   // parameters used for the energy loss calculations
//   //
//   const Float_t prim = 14.35; // number of primary collisions per 1 cm
//   const Float_t poti = 20.77e-9; // first ionization potential for Ne/CO2
//   const Float_t wIon = 35.97e-9; // energy for the ion-electron pair creation
//   const Float_t kScalewIonG4 = 0.85; // scale factor to tune kwIon for Geant4
//   const Float_t kFanoFactorG4 = 0.7; // parameter for smearing the number of ionizations (nel) using Geant4
//   const Int_t   kMaxDistRef =15;     // maximal difference between 2 stored references
//   // Float_t prim = fTPCParam->GetNprim();
//   // Float_t poti = fTPCParam->GetFpot();
//   // Float_t wIon = fTPCParam->GetWmean();
//
//   const Float_t kbig = 1.e10;
//
//   Int_t id,copy;
//   Float_t hits[5];
//   Int_t vol[2];
//   TLorentzVector p;
//
//   vol[1]=0; // preset row number to 0
//   //
//   if (!fPrimaryIonisation) TVirtualMC::GetMC()->SetMaxStep(kbig);
//
//   if(!TVirtualMC::GetMC()->IsTrackAlive()) return; // particle has disappeared
//
//   Float_t charge = TVirtualMC::GetMC()->TrackCharge();
//
//   if(TMath::Abs(charge)<=0.) return; // take only charged particles
//
//   // check the sensitive volume
//
//   id = TVirtualMC::GetMC()->CurrentVolID(copy); // vol ID and copy number (starts from 1!)
//   if(id != fIDrift && id != fIdSens) return; // not in the sensitive folume
//
//   if ( fPrimaryIonisation && id == fIDrift ) {
//     Double_t rnd = TVirtualMC::GetMC()->GetRandom()->Rndm();
//     TVirtualMC::GetMC()->SetMaxStep(0.2+(2.*rnd-1.)*0.05);  // 2 mm +- rndm*0.5mm step
//   }
//
//   //if ( fPrimaryIonisation && id == fIDrift && TVirtualMC::GetMC()->IsTrackEntering()) {
//   //  TVirtualMC::GetMC()->SetMaxStep(0.2);  // 2 mm
//   //}
//
//   TVirtualMC::GetMC()->TrackPosition(p);
//   Double_t r = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
//   //
//
//   //
//   Double_t angle = TMath::ACos(p[0]/r);
//   angle = (p[1]<0.) ? TMath::TwoPi()-angle : angle;
//   //
//   // angular segment, it is not a real sector number...
//   //
//   Int_t sector=TMath::Nint((angle-fTPCParam->GetInnerAngleShift())/
//   fTPCParam->GetInnerAngle());
//   // rotate to segment "0"
//   Float_t cos,sin;
//   fTPCParam->AdjustCosSin(sector,cos,sin);
//   Float_t x1=p[0]*cos + p[1]*sin;
//   // check if within sector's limits
//   if((x1>=fTPCParam->GetInnerRadiusLow()&&x1<=fTPCParam->GetInnerRadiusUp())
//     ||(x1>=fTPCParam->GetOuterRadiusLow()&&x1<=fTPCParam->GetOuterRadiusUp())){
//     // calculate real sector number...
//     if (x1>fTPCParam->GetOuterRadiusLow()){
//       sector = TMath::Nint((angle-fTPCParam->GetOuterAngleShift())/
//       fTPCParam->GetOuterAngle())+fTPCParam->GetNInnerSector();
//       if (p[2]<0)         sector+=(fTPCParam->GetNOuterSector()>>1);
//     } else {
//       if (p[2]<0) sector+=(fTPCParam->GetNInnerSector()>>1);
//     }
//     //
//     // here I have a sector number
//     //
//
//     vol[0]=sector;
//
//     static Double_t lastReferenceR=0;
//     if (TMath::Abs(lastReferenceR-r)>kMaxDistRef){
//       AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kTPC);
//       lastReferenceR = r;
//     }
//
//     // check if change of sector
//     if(sector != fSecOld){
//       fSecOld=sector;
//       // add track reference
//       AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kTPC);
//     }
//     // track is in the sensitive strip
//     if(id == fIdSens){
//       // track is entering the strip
//       if (TVirtualMC::GetMC()->IsTrackEntering()){
//         Int_t totrows = fTPCParam->GetNRowLow()+fTPCParam->GetNRowUp();
//         vol[1] = (copy<=totrows) ? copy-1 : copy-1-totrows;
//         // row numbers are autonomous for lower and upper sectors
//         if(vol[0] > fTPCParam->GetNInnerSector()) {
//           vol[1] -= fTPCParam->GetNRowLow();
//         }
//         //
//         if(vol[0]<fTPCParam->GetNInnerSector()&&vol[1] == 0){
//
//           // lower sector, row 0, because Jouri wants to have this
//
//           TVirtualMC::GetMC()->TrackMomentum(p);
//           hits[0]=p[0];
//           hits[1]=p[1];
//           hits[2]=p[2];
//           hits[3]=0.; // this hit has no energy loss
//           // Get also the track time for pileup simulation
//           hits[4]=TVirtualMC::GetMC()->TrackTime();
//
//           AddHit(gAlice->GetMCApp()->GetCurrentTrackNumber(), vol,hits);
//         }
//         //
//
//         TVirtualMC::GetMC()->TrackPosition(p);
//         hits[0]=p[0];
//         hits[1]=p[1];
//         hits[2]=p[2];
//         hits[3]=0.; // this hit has no energy loss
//         // Get also the track time for pileup simulation
//         hits[4]=TVirtualMC::GetMC()->TrackTime();
//
//         AddHit(gAlice->GetMCApp()->GetCurrentTrackNumber(), vol,hits);
//
//       }
//       else return;
//     }
//     //-----------------------------------------------------------------
//     //  charged particle is in the sensitive drift volume
//     //-----------------------------------------------------------------
//     if(TVirtualMC::GetMC()->TrackStep() > 0) {
//       Int_t nel=0;
//       if (!fPrimaryIonisation) {
//         nel = (Int_t)(((TVirtualMC::GetMC()->Edep())-poti)/wIon) + 1;
//       } else {
//
//         /*
//          *      static Double_t deForNextStep = 0.;
//          *      // Geant4 (the meaning of Edep as in Geant3) - wrong
//          *      //nel = (Int_t)(((TVirtualMC::GetMC()->Edep())-poti)/wIon) + 1;
//          *
//          *      // Geant4 (the meaning of Edep as in Geant3) - NEW
//          *      Double_t eAvailable = TVirtualMC::GetMC()->Edep() + deForNextStep;
//          *      nel = (Int_t)(eAvailable/wIon);
//          *      deForNextStep = eAvailable - nel*wIon;
//          */
//
//         //new Geant4-approach
//         Double_t meanIon = TVirtualMC::GetMC()->Edep()/(wIon*kScalewIonG4);
//         nel = (Int_t) ( kFanoFactorG4*AliMathBase::Gamma(meanIon/kFanoFactorG4)); // smear nel using gamma distr w mean = meanIon and variance = meanIon/kFanoFactorG4
//       }
//       nel=TMath::Min(nel,300); // 300 electrons corresponds to 10 keV
//       //
//       TVirtualMC::GetMC()->TrackPosition(p);
//       hits[0]=p[0];
//       hits[1]=p[1];
//       hits[2]=p[2];
//       hits[3]=(Float_t)nel;
//
//       // Add this hit
//
//       //    if (fHitType&&2){
//       if(fHitType){
//         TVirtualMC::GetMC()->TrackMomentum(p);
//         Float_t momentum = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]);
//         Float_t precision =   (momentum>0.1) ? 0.002 :0.01;
//         fTrackHits->SetHitPrecision(precision);
//       }
//
//       // Get also the track time for pileup simulation
//       hits[4]=TVirtualMC::GetMC()->TrackTime();
//
//       AddHit(gAlice->GetMCApp()->GetCurrentTrackNumber(), vol,hits);
//       if (fDebugStreamer){
//         // You can dump here what you need
//         // function  CreateDebugStremer() to be called in the Config.C  macro
//         // if you want to enable it
//         // By default debug streaemer is OFF
//         Float_t edep = TVirtualMC::GetMC()->Edep();
//         Float_t tstep = TVirtualMC::GetMC()->TrackStep();
//         Int_t pid=TVirtualMC::GetMC()->TrackPid();
//         (*fDebugStreamer)<<"hit"<<
//         "x="<<hits[0]<<  // hit position
//         "y="<<hits[1]<<
//         "z="<<hits[2]<<
//         "nel="<<hits[3]<<  // number of electorns
//         "tof="<<hits[4]<<  // hit TOF
//         "edep="<<edep<<    // energy deposit
//         "pid="<<pid<<      // pid
//         "step="<<tstep<<
//         "p.="<<&p<<
//         "\n";
//       }
//
//     } // step>0
//   } //within sector's limits
//   // Stemax calculation for the next step
//
//   Float_t pp;
//   TLorentzVector mom;
//   // below is valid only for Geant3 (fPromaryIonisation not set)
//   if(!fPrimaryIonisation){
//     TVirtualMC::GetMC()->TrackMomentum(mom);
//     Float_t ptot=mom.Rho();
//     Float_t betaGamma = ptot/TVirtualMC::GetMC()->TrackMass();
//
//     //Int_t pid=TVirtualMC::GetMC()->TrackPid();
//     // if((pid==kElectron || pid==kPositron) && ptot > 0.002)
//     //       {
//     //         pp = prim*1.58; // electrons above 20 MeV/c are on the plateau!
//     //       }
//     //     else
//     //       {
//
//     betaGamma = TMath::Max(betaGamma,(Float_t)7.e-3); // protection against too small bg
//     TVectorD *bbpar = fTPCParam->GetBetheBlochParametersMC(); //get parametrization from OCDB
//     pp=prim*AliMathBase::BetheBlochAleph(betaGamma,(*bbpar)(0),(*bbpar)(1),(*bbpar)(2),(*bbpar)(3),(*bbpar)(4));
//     //     }
//
//     Double_t rnd = TVirtualMC::GetMC()->GetRandom()->Rndm();
//
//     TVirtualMC::GetMC()->SetMaxStep(-TMath::Log(rnd)/pp);
//   }
//
// }

void Detector::EndOfEvent()
{

  mPointCollection->Clear();

}



void Detector::Register()
{

  /** This will create a branch in the output tree called
      DetectorPoint, setting the last parameter to kFALSE means:
      this collection will not be written to the file, it will exist
      only during the simulation.
  */

  FairRootManager::Instance()->Register("TPCPoint", "TPC",mPointCollection, kTRUE);

}


TClonesArray* Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0) { return mPointCollection; }
  else { return NULL; }
}

void Detector::Reset()
{
  mPointCollection->Clear();
}

void Detector::ConstructGeometry()
{
  // Create the detector materials
  createMaterials();

}

Point* Detector::AddHit(Int_t trackID, Int_t detID,
                                      TVector3 pos, TVector3 mom,
                                      Double_t time, Double_t length,
                                      Double_t eLoss)
{
  TClonesArray& clref = *mPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new(clref[size]) Point(trackID, detID, pos, mom,
         time, length, eLoss);
}

ClassImp(AliceO2::TPC::Detector)
