/// \file Detector.cxx
/// \brief Implementation of the Detector class
/// \author antonio.uras@cern.ch, bogdan.vulpescu@cern.ch 
/// \date 01/08/2016

#include "MFTSimulation/Geometry.h"
#include "MFTSimulation/GeometryTGeo.h"
#include "MFTSimulation/Detector.h"
#include "MFTSimulation/Point.h"
#include "MFTSimulation/HalfSegmentation.h"
#include "MFTSimulation/HalfDiskSegmentation.h"
#include "MFTSimulation/LadderSegmentation.h"

#include "SimulationDataFormat/DetectorList.h"
#include "SimulationDataFormat/Stack.h"

#include "TVirtualMC.h"
#include "TGeoGlobalMagField.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TClonesArray.h"
#include "TGeoManager.h"

#include "FairLogger.h"
#include "FairRootManager.h"
#include "FairVolume.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::Detector)

//_____________________________________________________________________________
Detector::Detector()
: AliceO2::Base::Detector("MFT", kTRUE, kAliMft),
  fVersion(1),
  fGeometryTGeo(0),
  fDensitySupportOverSi(0.036),
  fPoints(new TClonesArray("AliceO2::MFT::Point"))
{

}

//_____________________________________________________________________________
Detector::Detector(const Detector& src)
  : AliceO2::Base::Detector(src),
    fVersion(src.fVersion),
    fGeometryTGeo(src.fGeometryTGeo),
    fDensitySupportOverSi(src.fDensitySupportOverSi),
    fPoints(0)
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

  fVersion = src.fVersion;
  fGeometryTGeo = src.fGeometryTGeo;
  fDensitySupportOverSi = src.fDensitySupportOverSi;
  fPoints = 0;

}

//_____________________________________________________________________________
Detector::~Detector()
{

  delete fGeometryTGeo;

  if (fPoints) {
    fPoints->Delete();
    delete fPoints;
  }
  
}

//_____________________________________________________________________________
void Detector::Initialize()
{

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

  Geometry *mftGeo = Geometry::Instance();
  Segmentation * seg = mftGeo->GetSegmentation();
  if (!seg) Fatal("ProcessHits","No segmentation available",0,0);

  Int_t copy;
  // Check if hit is into a MFT sensor volume
  if(TVirtualMC::GetMC()->CurrentVolID(copy) != mftGeo->GetSensorVolumeID() ) return kFALSE;

  // Get The Sensor Unique ID
  Int_t chipId = -1, ladderId = -1, diskId = -1, halfId = -1, level = 0;
  TVirtualMC::GetMC()->CurrentVolOffID(++level,chipId);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,ladderId);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,diskId);
  TVirtualMC::GetMC()->CurrentVolOffID(++level,halfId);

  Int_t detElemID = mftGeo->GetObjectID(Geometry::kSensorType,halfId,diskId,ladderId,chipId);

  LOG(DEBUG1) << "Found hit into half = " << halfId << "; disk = " << diskId << "; ladder = " << ladderId << "; chip = " << chipId << FairLogger::endl;

  if (TVirtualMC::GetMC()->IsTrackExiting()) {
    //AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kMFT);
  }
  
  static TLorentzVector position, momentum;
  
  Int_t  status = 0;
  
  // Track status
  if (TVirtualMC::GetMC()->IsTrackInside())      status += 0x1<<0;
  if (TVirtualMC::GetMC()->IsTrackEntering())    status += 0x1<<1;
  if (TVirtualMC::GetMC()->IsTrackExiting())     status += 0x1<<2;
  if (TVirtualMC::GetMC()->IsTrackOut())         status += 0x1<<3;
  if (TVirtualMC::GetMC()->IsTrackDisappeared()) status += 0x1<<4;
  if (TVirtualMC::GetMC()->IsTrackStop())        status += 0x1<<5;
  if (TVirtualMC::GetMC()->IsTrackAlive())       status += 0x1<<6;
  
  // ---------- Fill hit structure
  /*
  hit.SetDetElemID(detElemID);
  hit.SetPlane(diskId);
  hit.SetTrack(gAlice->GetMCApp()->GetCurrentTrackNumber());
  
  TVirtualMC::GetMC()->TrackPosition(position);
  TVirtualMC::GetMC()->TrackMomentum(momentum);
  
  LOG(DEBUG1) << TVirtualMC::GetMC()->CurrentVolName() << " Hit " << fNHits << " (x = " << position.X() << " , y = " << position.Y() << " , z = " << position.Z() << ") belongs to track " << gAlice->GetMCApp()->GetCurrentTrackNumber() << FairLogger::endl;
  
  hit.SetPosition(position);
  hit.SetTOF(TVirtualMC::GetMC()->TrackTime());
  hit.SetMomentum(momentum);
  hit.SetStatus(status);
  hit.SetEloss(TVirtualMC::GetMC()->Edep());
  //hit.SetShunt(GetIshunt());
  //if (TVirtualMC::GetMC()->IsTrackEntering()) {
  //  hit.SetStartPosition(position);
  //  hit.SetStartTime(TVirtualMC::GetMC()->TrackTime());
  //  hit.SetStartStatus(status);
  //  return; // don't save entering hit.
  //}
  
  // Fill hit structure with this new hit.
  new ((*fHits)[fNhits++]) AliMFTHit(hit);
  
  // Save old position... for next hit.
  //   hit.SetStartPosition(position);
  //   hit.SetStartTime(TVirtualMC::GetMC()->TrackTime());
  //   hit.SetStartStatus(status);
  */
  if (!TVirtualMC::GetMC()->IsTrackExiting()) return kFALSE;

  Int_t trackID  = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
  Int_t detID = vol->getMCid();
  TVirtualMC::GetMC()->TrackPosition(position);
  TVirtualMC::GetMC()->TrackMomentum(momentum);
  Double32_t time = TVirtualMC::GetMC()->TrackTime();
  Double32_t length = TVirtualMC::GetMC()->TrackLength();
  Double32_t eLoss = TVirtualMC::GetMC()->Edep();

  AddHit(trackID, detID, TVector3(position.X(),position.Y(),position.Z()), TVector3(momentum.X(),momentum.Y(),momentum.Z()), time, length, eLoss);
  /*
  printf("%5d %3d %10.4f %10.4f %10.4f %6.2f %6.2f %6.2f status ",trackID,detID,position.X(),position.Y(),position.Z(),momentum.X(),momentum.Y(),momentum.Z());
  for (Int_t i = 7; i >= 0; i--) printf("%1d",((status >> i) & 0x1));
  printf("\n");
  */
  // Increment number of Detector det points in TParticle
  AliceO2::Data::Stack *stack = (AliceO2::Data::Stack *) TVirtualMC::GetMC()->GetStack();
  stack->AddPoint(kAliMft);

  return kTRUE;

}

//_____________________________________________________________________________
Point* Detector::AddHit(Int_t trackID, Int_t detID, TVector3 pos, TVector3 mom, Double_t time, Double_t length, Double_t eLoss)
{

  TClonesArray &clref = *fPoints;
  Int_t size = clref.GetEntriesFast();

  return new(clref[size]) Point(trackID, detID, pos, mom, time, length, eLoss);

}

//_____________________________________________________________________________
void Detector::CreateMaterials()
{
  
  // data from PDG booklet 2002                 
  // density [gr/cm^3], rad len [cm], abs len [cm]
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
  Float_t dCM55J = 1.33; // new value for MFT, from J.M. Buhour infos

  // Rohacell mixture
  const Int_t nRohacell = 3;
  Float_t aRohacell[nRohacell] = {1.00794, 12.0107, 15.9994};
  Float_t zRohacell[nRohacell] = {1., 6., 8.};
  Float_t wRohacell[nRohacell] = {0.0858, 0.5964, 0.3178};
  Float_t dRohacell = 0.032;  // 0.032 g/cm3 rohacell 31, 0.075 g/cm3 rohacell 71;
  
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
  // X7R capacitors - updated from F.Tosello's web page - M.S. 18 Oct 10
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
  Int_t   itgfld = 3;			     // type of field intergration 0 no field -1 user in guswim 1 Runge Kutta 2 helix 3 const field along z
  Float_t maxfld = 5.; 		             // max field value
  
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
  
  //Int_t    fieldType        = ((AliceO2::Field::MagneticField*)TGeoGlobalMagField::Instance()->GetField())->Integral();     // Field type
  //Double_t maxField         = ((AliceO2::Field::MagneticField*)TGeoGlobalMagField::Instance()->GetField())->Max();     // Field max.
  
  Int_t fieldType = 2;
  Float_t maxField = 10.0;

  AliceO2::Base::Detector::Mixture(kAir,"Air$", aAir, zAir, dAir, nAir, wAir);
  AliceO2::Base::Detector::Medium(kAir,    "Air$", kAir, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(kVacuum, "Vacuum$", aAir, zAir, dAirVacuum, nAir, wAir);
  AliceO2::Base::Detector::Medium(kVacuum,  "Vacuum$", kVacuum, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  AliceO2::Base::Detector::Material(++matId, "Si$", aSi, zSi, dSi, radSi, absSi);
  AliceO2::Base::Detector::Medium(kSi,       "Si$", matId, sens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
  AliceO2::Base::Detector::Material(++matId, "Readout$", aSi, zSi, dSi, radSi, absSi);
  AliceO2::Base::Detector::Medium(kReadout,  "Readout$", matId, unsens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
  AliceO2::Base::Detector::Material(++matId, "Support$", aSi, zSi, dSi*fDensitySupportOverSi, radSi/fDensitySupportOverSi, absSi/fDensitySupportOverSi);
  AliceO2::Base::Detector::Medium(kSupport,  "Support$", matId, unsens, fieldType, maxField, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  
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
  AliceO2::Base::Detector::Material(matId, "Carbon$", aCarb, zCarb, dCarb, radCarb, absCarb);
  AliceO2::Base::Detector::Medium(kCarbon, "Carbon$", matId,0,fieldType,maxField,maxBending,maxStepSize,maxEnergyLoss,precision,minStepSize);

  AliceO2::Base::Detector::Material(++matId, "Be$", aBe, zBe, dBe, radBe, absBe );
  AliceO2::Base::Detector::Medium(kBe,   "Be$", matId, unsens, fieldType,  maxField, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Material(++matId, "Alu$", aAlu, zAlu, dAlu, radAlu, absAlu);
  AliceO2::Base::Detector::Medium(kAlu,      "Alu$", matId, unsens, fieldType,  maxField, tmaxfd, stemax, deemax, epsil, stmin);
    
  AliceO2::Base::Detector::Mixture(++matId, "Water$", aWater, zWater, dWater, nWater, wWater);
  AliceO2::Base::Detector::Medium(kWater,   "Water$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "SiO2$", aSiO2, zSiO2, dSiO2, nSiO2, wSiO2);
  AliceO2::Base::Detector::Medium(kSiO2,    "SiO2$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "Inox$", aInox, zInox, dInox, nInox, wInox);
  AliceO2::Base::Detector::Medium(kInox,    "Inox$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "Kapton$", aKapton, zKapton, dKapton, 4, wKapton);
  AliceO2::Base::Detector::Medium(kKapton,"Kapton$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "Epoxy$", aEpoxy, zEpoxy, dEpoxy, -3, wEpoxy);
  AliceO2::Base::Detector::Medium(kEpoxy,"Epoxy$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "SE4445$", aSE4445, zSE4445, dSE4445, -5, wSE4445);
  AliceO2::Base::Detector::Medium(kSE4445,"SE4445$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId,"CarbonFiber$",aCM55J,zCM55J,dCM55J,4,wCM55J);
  AliceO2::Base::Detector::Medium(kCarbonEpoxy,"CarbonFiber$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId,  "Rohacell", aRohacell, zRohacell, dRohacell, nRohacell, wRohacell);
  AliceO2::Base::Detector::Medium(kRohacell, "Rohacell", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId,  "Polyimide", aPolyimide, zPolyimide, dPolyimide, nPolyimide, wPolyimide);
  AliceO2::Base::Detector::Medium(kPolyimide, "Polyimide", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
	
  AliceO2::Base::Detector::Mixture(++matId, "PEEK$", aPEEK, zPEEK, dPEEK, nPEEK, wPEEK);
  AliceO2::Base::Detector::Medium(kPEEK,    "PEEK$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Mixture(++matId, "FR4$", aFR4, zFR4, dFR4, nFR4, wFR4);
  AliceO2::Base::Detector::Medium(kFR4,    "FR4$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
  
  AliceO2::Base::Detector::Material(++matId, "Cu$", aCu, zCu, dCu, radCu, absCu);
  AliceO2::Base::Detector::Medium(kCu,       "Cu$", matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);
 
  AliceO2::Base::Detector::Mixture(++matId, "X7Rcapacitors$",aX7R,zX7R,dX7R,6,wX7R);
  AliceO2::Base::Detector::Medium(kX7R,     "X7Rcapacitors$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  AliceO2::Base::Detector::Mixture(++matId, "X7Rweld$",aX7Rweld,zX7Rweld,dX7Rweld,2,wX7Rweld);
  AliceO2::Base::Detector::Medium(kX7Rw,    "X7Rweld$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

  // Carbon fleece from AliITSSUv2.cxx
  AliceO2::Base::Detector::Material(++matId,"CarbonFleece$",12.0107,6,0.4,radCarb,absCarb);          // 999,999);  why 999???
  AliceO2::Base::Detector::Medium(kCarbonFleece,  "CarbonFleece$",matId, unsens, itgfld, maxfld, tmaxfd, stemax, deemax, epsil, stmin);

}

//_____________________________________________________________________________
void Detector::CreateGeometry()
{

  Geometry *mftGeom = Geometry::Instance();
  mftGeom->Build();
  fGeometryTGeo = new GeometryTGeo();

}

//_____________________________________________________________________________
void Detector::ConstructGeometry()
{

  CreateMaterials();
  CreateGeometry();
  DefineSensitiveVolumes();

}

//_____________________________________________________________________________
void Detector::DefineSensitiveVolumes()
{

  TGeoVolume* vol;
  /*
  Geometry *mftGeo = Geometry::Instance();
  Segmentation *seg = mftGeo->GetSegmentation();

  Int_t nSensVol = 0;
  for (Int_t iHalf = 0; iHalf < 2; iHalf++) {
    HalfSegmentation * halfSeg = seg->GetHalf(iHalf);
    for (Int_t iDisk = 0; iDisk < halfSeg->GetNHalfDisks(); iDisk++) {
      HalfDiskSegmentation* halfDiskSeg = halfSeg->GetHalfDisk(iDisk);
      for (Int_t iLadder = 0; iLadder < halfDiskSeg->GetNLadders(); iLadder++) {
	LadderSegmentation* ladderSeg = halfDiskSeg->GetLadder(iLadder);
	TString volumeName = Form("MFT_S_%d_%d_%d",
	  mftGeo->GetHalfID(ladderSeg->GetUniqueID()),
	  mftGeo->GetHalfDiskID(ladderSeg->GetUniqueID()),
	  mftGeo->GetLadderID(ladderSeg->GetUniqueID()));
	vol = gGeoManager->GetVolume(volumeName.Data());
	LOG(INFO) << "Add sensitive volume: " << volumeName.Data() << FairLogger::endl;
	AddSensitiveVolume(vol);
	nSensVol++;
      }
    }
  }
  Info("DefineSensitiveVolumes",Form("%d volumes addes.\n",nSensVol),0,0);
  */
  vol = gGeoManager->GetVolume("MFTSensor");
  AddSensitiveVolume(vol);

}

//_____________________________________________________________________________
void Detector::EndOfEvent()
{

  if (fPoints) { 
    fPoints->Clear(); 
  }

}

//_____________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Point, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->Register("MFTPoints", "MFT", fPoints, kTRUE);
  }

}
//_____________________________________________________________________________
TClonesArray *Detector::GetCollection(Int_t iColl) const
{

  if (iColl == 0) {
    return fPoints;
  } else {
    return NULL;
  }

}

//_____________________________________________________________________________
void Detector::Reset()
{

  fPoints->Clear();

}
