// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.cxx
/// \brief Implementation of the Detector class
/// \author antonio.uras@cern.ch, bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#include "ITSMFTSimulation/Hit.h"

#include "MFTBase/Geometry.h"
#include "MFTBase/GeometryTGeo.h"

#include "MFTSimulation/Detector.h"

#include "SimulationDataFormat/Stack.h"
#include "Field/MagneticField.h"

#include "TVirtualMC.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TClonesArray.h"
#include "TGeoManager.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "FairVolume.h"

using o2::ITSMFT::Hit;
using namespace o2::MFT;

ClassImp(o2::MFT::Detector)

//_____________________________________________________________________________
Detector::Detector()
: o2::Base::DetImpl<Detector>("MFT", kTRUE),
  mVersion(1),
  mDensitySupportOverSi(0.036),
  mHits(new std::vector<o2::ITSMFT::Hit>),
  mTrackData()
{

}

//_____________________________________________________________________________
Detector::Detector(const Detector& src)
  : o2::Base::DetImpl<Detector>(src),
    mVersion(src.mVersion),
    mDensitySupportOverSi(src.mDensitySupportOverSi),
    mHits(new std::vector<o2::ITSMFT::Hit>),
    mTrackData()
{
  
}

//_____________________________________________________________________________
Detector &Detector::operator=(const Detector &src)
{

  if (this == &src) {
    return *this;
  }

  // base class assignment
  Base::Detector::operator=(src);

  mVersion = src.mVersion;
  mDensitySupportOverSi = src.mDensitySupportOverSi;
  mHits = nullptr;
  mTrackData.mHitStarted = src.mTrackData.mHitStarted;
  mTrackData.mTrkStatusStart = src.mTrackData.mTrkStatusStart;
  mTrackData.mPositionStart = src.mTrackData.mPositionStart;
  mTrackData.mMomentumStart = src.mTrackData.mMomentumStart;
  mTrackData.mEnergyLoss = src.mTrackData.mEnergyLoss;
  return *this;
}

//_____________________________________________________________________________
Detector::~Detector()
{

  if (mHits) {
    delete mHits;
  }
  
}

//_____________________________________________________________________________
void Detector::Initialize()
{

  mGeometryTGeo = GeometryTGeo::Instance();

  defineSensitiveVolumes();

  FairDetector::Initialize();

}

//_____________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping

  // do not track neutral particles
  if (!(TVirtualMC::GetMC()->TrackCharge())) {
    return kFALSE;
  }

  Geometry *mftGeo = Geometry::instance();

  Int_t copy;
  // Check if hit is into a MFT sensor volume
  if(TVirtualMC::GetMC()->CurrentVolID(copy) != mftGeo->getSensorVolumeID() ) return kFALSE;

  // Get The Sensor Unique ID
  Int_t sensorID = -1, ladderID = -1, diskID = -1, halfID = -1, level = 0;
  TVirtualMC::GetMC()->CurrentVolOffID(++level,sensorID);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,ladderID);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,diskID);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,halfID);
  
  Int_t sensorIndex = mGeometryTGeo->getSensorIndex(halfID,diskID,ladderID,sensorID);

  //LOG(INFO) << "Found hit into half = " << halfID << "; disk = " << diskID << "; ladder = " << ladderID << "; sensor = " << sensorID << FairLogger::endl;

  bool startHit=false, stopHit=false;
  unsigned char status = 0;
  if (TVirtualMC::GetMC()->IsTrackEntering()) { status |= Hit::kTrackEntering; }
  if (TVirtualMC::GetMC()->IsTrackInside())   { status |= Hit::kTrackInside; }
  if (TVirtualMC::GetMC()->IsTrackExiting())  { status |= Hit::kTrackExiting; }
  if (TVirtualMC::GetMC()->IsTrackOut())      { status |= Hit::kTrackOut; }
  if (TVirtualMC::GetMC()->IsTrackStop())     { status |= Hit::kTrackStopped; }
  if (TVirtualMC::GetMC()->IsTrackAlive())    { status |= Hit::kTrackAlive; }

  // track is entering or created in the volume
  if ( (status & Hit::kTrackEntering) || (status & Hit::kTrackInside && !mTrackData.mHitStarted) ) {
    startHit = true;
  }
  else if ( (status & (Hit::kTrackExiting|Hit::kTrackOut|Hit::kTrackStopped)) ) {
    stopHit = true;
  }

  // increment energy loss at all steps except entrance
  if (!startHit) mTrackData.mEnergyLoss += TVirtualMC::GetMC()->Edep();
  if (!(startHit|stopHit)) return kFALSE; // do noting

  if (startHit) {

    mTrackData.mEnergyLoss = 0.;
    TVirtualMC::GetMC()->TrackMomentum(mTrackData.mMomentumStart);
    TVirtualMC::GetMC()->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = status;
    mTrackData.mHitStarted = true;

  }

  if (stopHit) {

    TLorentzVector positionStop;
    TVirtualMC::GetMC()->TrackPosition(positionStop);

    Int_t trackID  = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
    //Int_t detID = vol->getMCid();

    Hit *p = addHit(trackID,
                    sensorIndex,
                    mTrackData.mPositionStart.Vect(),
                    positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(),
                    mTrackData.mMomentumStart.E(),
                    positionStop.T(),
                    mTrackData.mEnergyLoss, 
                    mTrackData.mTrkStatusStart,
                    status);
    
    o2::Data::Stack *stack = (o2::Data::Stack *) TVirtualMC::GetMC()->GetStack();
    stack->addHit(GetDetId());
    
  }
  
  return kTRUE;

}

//_____________________________________________________________________________
Hit* Detector::addHit(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 endPos, TVector3 startMom, Double_t startE, Double_t endTime, Double_t eLoss, unsigned char startStatus, unsigned char endStatus)
{

  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);

  return &(mHits->back());

}

//_____________________________________________________________________________
void Detector::createMaterials()
{
  
  // data from PDG booklet 2002                 density [gr/cm^3]     rad len [cm]           abs len [cm]
  Float_t   aSi = 28.085 ,    zSi   = 14. ,     dSi      =  2.329 ,   radSi   =  21.82/dSi , absSi   = 108.4/dSi  ;    // Silicon
  Float_t   aCarb = 12.01 ,   zCarb =  6. ,     dCarb    =  2.265 ,   radCarb =  18.8 ,      absCarb = 49.9       ;    // Carbon
  Float_t   aAlu = 26.98 ,    zAlu  = 13. ,     dAlu     =  2.70  ,   radAlu  =  8.897 ,     absAlu  = 39.70      ;    // Aluminum
  Float_t   aBe = 9.012182 ,  zBe   = 4. ,      dBe      =  1.85 ,    radBe   =  65.19/dBe , absBe   = 77.8/dBe  ;     // Beryllium
  Float_t   aCu = 63.546 ,    zCu  = 29.  ,     dCu      =  8.96  ,   radCu   =  1.436 ,     absCu   = 15.32      ;    // Copper

  // Air mixture
  const Int_t nAir = 4;
  Float_t   aAir[nAir] = {12, 14, 16, 36} ,  zAir[nAir] = {6, 7, 8, 18} ,   wAir[nAir]={0.000124, 0.755267, 0.231781, 0.012827} , dAir=0.00120479, dAirVacuum=0.00120479e-4;

  // Water mixture
  const Int_t nWater = 2;
  Float_t   aWater[nWater] = {1.00794, 15.9994} ,  zWater[nWater] = {1, 8} ,   wWater[nWater] = {0.111894, 0.888106} , dWater=1.;
  
  // SiO2 mixture
  const Int_t nSiO2 = 2;
  Float_t   aSiO2[nSiO2] = {15.9994, 28.0855} ,   zSiO2[nSiO2] = {8., 14.} ,   wSiO2[nSiO2] = {0.532565, 0.467435} , dSiO2 = 2.20;
  
  // Inox mixture
  const Int_t nInox = 9;
  Float_t   aInox[nInox] = {12.0107, 54.9380, 28.0855, 30.9738, 32.0660, 58.6928, 51.9961, 95.9400, 55.8450} ;
  Float_t   zInox[nInox] = { 6,      25,      14,      15,      16,      28,      24,      42,      26     } ;
  Float_t   wInox[nInox] = {0.0003,  0.02,    0.01,    0.00045, 0.0003,  0.12,    0.17,    0.025,   0.65395} ;
  Float_t   dInox = 8.03;
  
  // Kapton polyimide film (from SPD AliITSv11.cxx)  and http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=179
  Float_t aKapton[4]={1.00794,12.0107, 14.010,15.9994};
  Float_t zKapton[4]={1.,6.,7.,8.};
  Float_t wKapton[4]={0.026362,0.69113,0.07327,0.209235};
  Float_t dKapton   = 1.42;
  
  //--- EPOXY  --- C18 H19 O3 from ITS AliITSv11.cxx
  Float_t aEpoxy[3] = {15.9994, 1.00794, 12.0107} ;
  Float_t zEpoxy[3] = {     8.,      1.,      6.} ;
  Float_t wEpoxy[3] = {     3.,     19.,     18.} ;
  Float_t dEpoxy = 1.23; //  1.8 very high value from ITS! ou 1.23 from eccobond 45 lv datasheet

  //--- Silicone SE4445 Dow Corning  
  // Si, Al, O, C, H
  Float_t aSE4445[5] = {28.0855, 26.981538, 15.9994, 12.0107, 1.00794} ;
  Float_t zSE4445[5] = {    14.,       13.,      8.,      6.,      1.} ;
  Float_t wSE4445[5] = {  5.531,    45.222,  43.351,   4.717,   1.172} ;
  Float_t dSE4445 = 2.36; //from LBNL file, priv. comm.
  
  //--- CARBON FIBER CM55J --- from ITS AliITSv11.cxx
  Float_t aCM55J[4]={12.0107,14.0067,15.9994,1.00794};
  Float_t zCM55J[4]={6.,7.,8.,1.};
  Float_t wCM55J[4]={0.908508078,0.010387573,0.055957585,0.025146765};
  //Float_t dCM55J = 1.33; // from J.M. Buhour infos
  Float_t dCM55J = 1.548;// increase by 16.4% to account that water pipes are outside the rohacell plate

  // Rohacell mixture
  const Int_t nRohacell = 3;
  Float_t aRohacell[nRohacell] = {1.00794, 12.0107, 15.9994};
  Float_t zRohacell[nRohacell] = {1., 6., 8.};
  Float_t wRohacell[nRohacell] = {0.0858, 0.5964, 0.3178};
  //Float_t dRohacell = 0.032;  // 0.032 g/cm3 rohacell 31, 0.075 g/cm3 rohacell 71;
  Float_t dRohacell = 0.032/(1-0.3134);  // the density is increased since the water pipes are not inside the rohacell in the code ==> thickness decreased by 31.34%
  
  // Polyimide pipe mixture
  const Int_t nPolyimide = 4;
  Float_t aPolyimide[nPolyimide] = {1.00794, 12.0107, 14.0067, 15.9994};
  Float_t zPolyimide[nPolyimide] = {1, 6, 7, 8};
  Float_t wPolyimide[nPolyimide] = {0.00942, 0.56089, 0.13082, 0.29887};
  Float_t dPolyimide = 1.4;   

  // PEEK mixture (Polyether Ether Ketone)
  const Int_t nPEEK = 3;
  Float_t   aPEEK[nPEEK] = {1.00794, 12.0107, 15.9994} ;
  Float_t   zPEEK[nPEEK] = {1,       6,        8} ;
  Float_t   wPEEK[nPEEK] = {0.06713, 0.40001,  0.53285} ;
  Float_t   dPEEK = 1.32;
  
  // (Printed Circuit Board), material type FR4
  const Int_t nFR4 = 5;
  Float_t   aFR4[nFR4] = {1.00794,    12.0107, 15.9994, 28.0855,   79.904} ;
  Float_t   zFR4[nFR4] = {1,          6,       8,       14,   35} ;
  Float_t   wFR4[nFR4] = {0.0684428,  0.278042,0.405633, 0.180774,    0.0671091} ;
  Float_t   dFR4 = 1.7; //Density FR4= 1.7 Cu=8.96


  //======================== From ITS code ===================================
  //X7R capacitors - updated from F.Tosello's web page - M.S. 18 Oct 10
  // 58.6928 --> innner electrodes (mainly Ni)
  // 63.5460 --> terminaisons (Cu) 
  // 118.710 --> terminaisons (Sn)
  // 137.327 Ba, 47.867 Ti, 15.9994 O  (mainly BaTiO3)
  Float_t aX7R[6]={137.327,47.867,15.9994,58.6928,63.5460,118.710};
  Float_t zX7R[6]={56.,22.,8.,28.,29.,50.};
  Float_t wX7R[6]={0.524732,0.176736,0.179282,0.079750,0.019750,0.019750};
  Float_t dX7R = 6.07914;
  
  //X7R weld, i.e. Sn 60% Pb 40% (from F.Tosello's web page - M.S. 15 Oct 10)
  
  Float_t aX7Rweld[2]={118.71 , 207.20};
  Float_t zX7Rweld[2]={ 50.   ,  82.  };
  Float_t wX7Rweld[2]={  0.60 ,   0.40};
  Float_t dX7Rweld   = 8.52358;
  //==========================================================================
  
  Int_t   matId  = 0;                        // tmp material id number
  Int_t   unsens = 0, sens=1;                // sensitive or unsensitive medium
  Int_t   itgfld = 3;                        // type of field intergration 0 no field -1 user in guswim 1 Runge Kutta 2 helix 3 const field along z
  Float_t maxfld = 5.;                       // max field value
  
  Float_t tmaxfd = -10.0;                    // max deflection angle due to magnetic field in one step
  Float_t stemax =  0.001;                   // max step allowed [cm]
  Float_t deemax = -0.2;                     // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil  =  0.001;                   // tracking precision [cm]
  Float_t stmin  = -0.001;                   // minimum step due to continuous processes [cm] (negative value: choose it automatically)
  
  Float_t tmaxfdSi =  0.1;                   // max deflection angle due to magnetic field in one step
  Float_t stemaxSi =  5.0e-4;                // maximum step allowed [cm]
  Float_t deemaxSi =  0.1;                   // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsilSi  =  0.5e-4;                // tracking precision [cm]
  Float_t stminSi  = -0.001;                 // minimum step due to continuous processes [cm] (negative value: choose it automatically)
  
  Int_t fieldType;
  Float_t maxField;
  o2::Base::Detector::initFieldTrackingParams(fieldType, maxField);
  
  LOG(INFO) << "Detector::createMaterials >>>>> fieldType " << fieldType << " maxField " << maxField << "\n"; 

  o2::Base::Detector::Mixture(++matId, "Air$", aAir, zAir, dAir, nAir, wAir);
  o2::Base::Detector::Medium(Air,     "Air$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  o2::Base::Detector::Mixture(++matId, "Vacuum$", aAir, zAir, dAirVacuum, nAir, wAir);
  o2::Base::Detector::Medium(Vacuum,  "Vacuum$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  o2::Base::Detector::Material(++matId, "Si$", aSi, zSi, dSi, radSi, absSi);
  o2::Base::Detector::Medium(Si,       "Si$", matId, sens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
  o2::Base::Detector::Material(++matId, "Readout$", aSi, zSi, dSi, radSi, absSi);
  o2::Base::Detector::Medium(Readout,  "Readout$", matId, unsens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
  o2::Base::Detector::Material(++matId, "Support$", aSi, zSi, dSi*mDensitySupportOverSi, radSi/mDensitySupportOverSi, absSi/mDensitySupportOverSi);
  o2::Base::Detector::Medium(Support,  "Support$", matId, unsens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
  Double_t maxBending       = 0;     // Max Angle
  Double_t maxStepSize      = 0.001; // Max step size
  Double_t maxEnergyLoss    = 1;     // Max Delta E
  Double_t precision        = 0.001; // Precision
  Double_t minStepSize      = 0.001; // Minimum step size

  // Carbon
  aCarb                = 12.011;
  zCarb                = 6.;
  dCarb          = 2.265;
  radCarb  = 18.8;
  absCarb = 999;
  maxBending       = 10;
  maxStepSize      = .01;
  precision        = .003;
  minStepSize      = .003;
  o2::Base::Detector::Material(++matId, "Carbon$", aCarb, zCarb, dCarb, radCarb, absCarb);
  o2::Base::Detector::Medium(Carbon, "Carbon$", matId,0,fieldType,maxField,maxBending,maxStepSize,maxEnergyLoss,precision,minStepSize);

  o2::Base::Detector::Material(++matId, "Be$", aBe, zBe, dBe, radBe, absBe );
  o2::Base::Detector::Medium(Be,   "Be$", matId, unsens, fieldType,  maxField, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Material(++matId, "Alu$", aAlu, zAlu, dAlu, radAlu, absAlu);
  o2::Base::Detector::Medium(Alu,      "Alu$", matId, unsens, fieldType,  maxField, tmaxfd, stemax, deemax, epsil, stmin);
    
  o2::Base::Detector::Mixture(++matId, "Water$", aWater, zWater, dWater, nWater, wWater);
  o2::Base::Detector::Medium(Water,   "Water$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "SiO2$", aSiO2, zSiO2, dSiO2, nSiO2, wSiO2);
  o2::Base::Detector::Medium(SiO2,    "SiO2$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "Inox$", aInox, zInox, dInox, nInox, wInox);
  o2::Base::Detector::Medium(Inox,    "Inox$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "Kapton$", aKapton, zKapton, dKapton, 4, wKapton);
  o2::Base::Detector::Medium(Kapton,"Kapton$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "Epoxy$", aEpoxy, zEpoxy, dEpoxy, -3, wEpoxy);
  o2::Base::Detector::Medium(Epoxy,"Epoxy$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "SE4445$", aSE4445, zSE4445, dSE4445, -5, wSE4445);
  o2::Base::Detector::Medium(SE4445,"SE4445$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId,"CarbonFiber$",aCM55J,zCM55J,dCM55J,4,wCM55J);
  o2::Base::Detector::Medium(CarbonEpoxy,"CarbonFiber$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId,  "Rohacell", aRohacell, zRohacell, dRohacell, nRohacell, wRohacell);
  o2::Base::Detector::Medium(Rohacell, "Rohacell", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId,  "Polyimide", aPolyimide, zPolyimide, dPolyimide, nPolyimide, wPolyimide);
  o2::Base::Detector::Medium(Polyimide, "Polyimide", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
        
  o2::Base::Detector::Mixture(++matId, "PEEK$", aPEEK, zPEEK, dPEEK, nPEEK, wPEEK);
  o2::Base::Detector::Medium(PEEK,    "PEEK$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Mixture(++matId, "FR4$", aFR4, zFR4, dFR4, nFR4, wFR4);
  o2::Base::Detector::Medium(FR4,    "FR4$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  o2::Base::Detector::Material(++matId, "Cu$", aCu, zCu, dCu, radCu, absCu);
  o2::Base::Detector::Medium(Cu,       "Cu$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
 
  o2::Base::Detector::Mixture(++matId, "X7Rcapacitors$",aX7R,zX7R,dX7R,6,wX7R);
  o2::Base::Detector::Medium(X7R,     "X7Rcapacitors$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  o2::Base::Detector::Mixture(++matId, "X7Rweld$",aX7Rweld,zX7Rweld,dX7Rweld,2,wX7Rweld);
  o2::Base::Detector::Medium(X7Rw,    "X7Rweld$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  // Carbon fleece from AliITSSUv2.cxx
  o2::Base::Detector::Material(++matId,"CarbonFleece$",12.0107,6,0.4,radCarb,absCarb);          // 999,999);  why 999???
  o2::Base::Detector::Medium(CarbonFleece,  "CarbonFleece$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  LOG(INFO) << "Detector::createMaterials -----> matId = " << matId << "\n";

}

//_____________________________________________________________________________
void Detector::createGeometry()
{

  Geometry* mftGeom = Geometry::instance();
  mftGeom->build();

}

//_____________________________________________________________________________
void Detector::ConstructGeometry()
{

  createMaterials();
  createGeometry();

}

//_____________________________________________________________________________
void Detector::defineSensitiveVolumes()
{

  TGeoVolume* vol;
  Geometry* mftGeom = Geometry::instance();

  vol = gGeoManager->GetVolume("MFTSensor");
  if (!vol) {
    LOG(FATAL) << "can't find volume MFTSensor" << FairLogger::endl;
  } else {
    AddSensitiveVolume(vol);
    if(!mftGeom->getSensorVolumeID()) {
      mftGeom->setSensorVolumeID(vol->GetNumber());
    } else if (mftGeom->getSensorVolumeID() != vol->GetNumber()) {
      LOG(FATAL) << "CreateSensors: different Sensor volume ID !!!!" << FairLogger::endl;
    }
  }

}

//_____________________________________________________________________________
void Detector::EndOfEvent()
{

  Reset();

}

//_____________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }

}

//_____________________________________________________________________________
void Detector::Reset()
{

  mHits->clear();

}
