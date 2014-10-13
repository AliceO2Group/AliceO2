#include "Detector.h"

#include "AliDetectorList.h"
#include "Point.h"
#include "AliStack.h"

#include "FairVolume.h"
#include "FairGeoVolume.h"
#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairGeoLoader.h"
#include "FairGeoInterface.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"

#include "UpgradeV1Layer.h"
#include "UpgradeGeometryTGeo.h"

#include "TClonesArray.h"
#include "TGeoManager.h"
#include "TGeoTube.h"
#include "TGeoVolume.h"
#include "TVirtualMC.h"

#include <iostream>
#include <Riostream.h>

using std::cout;
using std::endl;

using namespace AliceO2::ITS;

AliceO2::ITS::Detector::Detector()
  : O2Detector("ITS", kTRUE, kAliIts),
    fIdSens(0),
    fTrackID(-1),
    fVolumeID(-1),
    fStartPos(),
    fPos(),
    fMom(),
    fStartTime(-1.),
    fTime(-1.),
    fLength(-1.),
    fELoss(-1),
    fShunt(),
    fO2itsPointCollection(new TClonesArray("AliceO2::ITS::Point")),
    fGeoHandler(new GeometryHandler()),
    fMisalignPar(NULL),
    fNrOfDetectors(-1),
    fShiftX(),
    fShiftY(),
    fShiftZ(),
    fRotX(),
    fRotY(),
    fRotZ(),
    fModifyGeometry(kFALSE),
    fNWrapVol(0),
    fWrapRMin(0),
    fWrapRMax(0),
    fWrapZSpan(0),
  	fLay2WrapV(0),
    fLayTurbo(0),
    fLayPhi0(0),
    fLayRadii(0),
    fLayZLength(0),
    fStavPerLay(0),
    fUnitPerStave(0),
    fStaveThick(0),
    fStaveWidth(0),
    fStaveTilt(0),
    fDetThick(0),
    fChipTypeID(0),
    fBuildLevel(0),
    fUpGeom(0),
    fStaveModelIB(kIBModel0),
    fStaveModelOB(kOBModel0)
{
}

AliceO2::ITS::Detector::Detector(const char* name, Bool_t active, const Int_t nlay)
  : O2Detector(name, active, kAliIts),
    fIdSens(0),
    fTrackID(-1),
    fVolumeID(-1),
    fStartPos(),
    fPos(),
    fMom(),
    fStartTime(-1.),
    fTime(-1.),
    fLength(-1.),
    fELoss(-1),
    fShunt(),
    fO2itsPointCollection(new TClonesArray("AliceO2::ITS::Point")),
    fGeoHandler(new GeometryHandler()),
    fMisalignPar(NULL),
    fNrOfDetectors(-1),
    fShiftX(),
    fShiftY(),
    fShiftZ(),
    fRotX(),
    fRotY(),
    fRotZ(),
    fModifyGeometry(kFALSE),
    fNWrapVol(0),
    fWrapRMin(0),
    fWrapRMax(0),
    fWrapZSpan(0),
  	fLay2WrapV(0),
    fLayTurbo(0),
    fLayPhi0(0),
    fLayRadii(0),
    fLayZLength(0),
    fStavPerLay(0),
    fUnitPerStave(0),
    fStaveThick(0),
    fStaveWidth(0),
    fStaveTilt(0),
    fDetThick(0),
    fChipTypeID(0),
    fBuildLevel(0),
    fUpGeom(0),
    fNLayers(nlay),
    fStaveModelIB(kIBModel0),
    fStaveModelOB(kOBModel0)
{
  fLayerName = new TString[fNLayers];
  
  for (Int_t j=0; j<fNLayers; j++)
    fLayerName[j].Form("%s%d", UpgradeGeometryTGeo::GetITSSensorPattern(),j); // See UpgradeV1Layer

  fLayTurbo     = new Bool_t[fNLayers];
  fLayPhi0      = new Double_t[fNLayers];
  fLayRadii     = new Double_t[fNLayers];
  fLayZLength   = new Double_t[fNLayers];
  fStavPerLay   = new Int_t[fNLayers];
  fUnitPerStave = new Int_t[fNLayers];
  fStaveThick   = new Double_t[fNLayers];
  fStaveWidth   = new Double_t[fNLayers];
  fStaveTilt    = new Double_t[fNLayers];
  fDetThick     = new Double_t[fNLayers];
  fChipTypeID   = new UInt_t[fNLayers];
  fBuildLevel   = new Int_t[fNLayers];

  fUpGeom = new UpgradeV1Layer*[fNLayers];
  
  if (fNLayers > 0) { // if not, we'll Fatal-ize in CreateGeometry
    for (Int_t j=0; j<fNLayers; j++) {
      fLayPhi0[j]      = 0;
      fLayRadii[j]     = 0.;
      fLayZLength[j]   = 0.;
      fStavPerLay[j]   = 0;
      fUnitPerStave[j] = 0;
      fStaveWidth[j]   = 0.;
      fDetThick[j]     = 0.;
      fChipTypeID[j]   = 0;
      fBuildLevel[j]   = 0;
      fUpGeom[j]       = 0;
    }
  }
}

AliceO2::ITS::Detector::~Detector()
{
  delete [] fLayTurbo;
  delete [] fLayPhi0;
  delete [] fLayRadii;
  delete [] fLayZLength;
  delete [] fStavPerLay;
  delete [] fUnitPerStave;
  delete [] fStaveThick;
  delete [] fStaveWidth;
  delete [] fStaveTilt;
  delete [] fDetThick;
  delete [] fChipTypeID;
  delete [] fBuildLevel;
  delete [] fUpGeom;
  delete [] fWrapRMin;
  delete [] fWrapRMax;
  delete [] fWrapZSpan;
  delete [] fLay2WrapV;
  
  if (fO2itsPointCollection) {
    fO2itsPointCollection->Delete();
    delete fO2itsPointCollection;
  }
  
  delete[] fIdSens;
}

AliceO2::ITS::Detector& AliceO2::ITS::Detector::operator=(const AliceO2::ITS::Detector &h){
    // The standard = operator
    // Inputs:
    //   Detector   &h the sourse of this copy
    // Outputs:
    //   none.
    // Return:
    //  A copy of the sourse hit h

    if(this == &h) return *this;
    this->fStatus  = h.fStatus;
    this->fModule  = h.fModule;
    this->fPx      = h.fPx;
    this->fPy      = h.fPy;
    this->fPz      = h.fPz;
    this->fDestep  = h.fDestep;
    this->fTof     = h.fTof;
    this->fStatus0 = h.fStatus0;
    this->fx0      = h.fx0;
    this->fy0      = h.fy0;
    this->fz0      = h.fz0;
    this->ft0      = h.ft0;
    return *this;
}

void AliceO2::ITS::Detector::Initialize()
{
  if (!fIdSens) {
    fIdSens = new Int_t[fNLayers];
  }
  
  for (int i=0;i<fNLayers;i++) {
    fIdSens[i] = gMC ? gMC->VolId(fLayerName[i]) : 0;
  }
  
  fGeomTGeo = new UpgradeGeometryTGeo(kTRUE);
  
  FairDetector::Initialize();

//  FairRuntimeDb* rtdb= FairRun::Instance()->GetRuntimeDb();
//  O2itsGeoPar* par=(O2itsGeoPar*)(rtdb->getContainer("O2itsGeoPar"));
}

void AliceO2::ITS::Detector::InitParContainers()
{
  LOG(INFO)<< "Initialize aliitsdet misallign parameters"<<FairLogger::endl;
  fNrOfDetectors=fMisalignPar->GetNrOfDetectors();
  fShiftX=fMisalignPar->GetShiftX();
  fShiftY=fMisalignPar->GetShiftY();
  fShiftZ=fMisalignPar->GetShiftZ();
  fRotX=fMisalignPar->GetRotX();
  fRotY=fMisalignPar->GetRotY();
  fRotZ=fMisalignPar->GetRotZ();
}

void AliceO2::ITS::Detector::SetParContainers()
{
  LOG(INFO)<< "Set tutdet misallign parameters"<<FairLogger::endl;
  // Get Base Container
  FairRun* sim = FairRun::Instance();
  LOG_IF(FATAL, !sim) << "No run object"<<FairLogger::endl;
  FairRuntimeDb* rtdb=sim->GetRuntimeDb();
  LOG_IF(FATAL, !rtdb) << "No runtime database"<<FairLogger::endl;

  fMisalignPar = (MisalignmentParameter*)
                 (rtdb->getContainer("MisallignmentParameter"));

}

Bool_t AliceO2::ITS::Detector::ProcessHits(FairVolume* vol)
{
  /** This method is called from the MC stepping */
  if(!(gMC->TrackCharge())) {
    return kFALSE;
  }
  
  //FIXME: Is copy actually needed?
  Int_t copy = vol->getCopyNo();
  Int_t id = vol->getMCid();
  Int_t lay = 0;
  Int_t cpn0, cpn1, mod;
    
  //FIXME: Determine the layer number. Is this information available directly from the FairVolume?
  while ((lay<fNLayers) && id!=fIdSens[lay]) {
    ++lay;
  }

  //FIXME: Is it needed to keep a track reference when the outer ITS volume is encountered?
  /* if(gMC->IsTrackExiting()) {
    AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kITS);
  } // if Outer ITS mother Volume */
  
  // Retrieve the indices with the volume path
  copy = 1;
  gMC->CurrentVolOffID(1, cpn1);
  gMC->CurrentVolOffID(2, cpn0);
    
  mod = fGeomTGeo->GetChipIndex(lay, cpn0, cpn1);
 
  // Record information on the points
  fELoss = gMC->Edep();
  fTime = gMC->TrackTime();
  fTrackID = gMC->GetStack()->GetCurrentTrackNumber();
  fVolumeID = vol->getMCid();
  
  //FIXME: Set a temporary value to fShunt for now, determine its use at a later stage
  fShunt = 0;
  
  gMC->TrackPosition(fPos);
  gMC->TrackMomentum(fMom);
  
  //fLength = gMC->TrackLength();
 
  if(gMC->IsTrackEntering()){
    fStartPos = fPos;
    fStartTime = fTime;
    return kFALSE; // don't save entering hit.
  }

  // Create Point on every step of the active volume
  AddHit(fTrackID, fVolumeID, TVector3(fStartPos.X(),  fStartPos.Y(),  fStartPos.Z()),
          TVector3(fPos.X(),  fPos.Y(),  fPos.Z()), TVector3(fMom.Px(), fMom.Py(), fMom.Pz()),
          fStartTime, fTime, fLength, fELoss, fShunt);

  // Increment number of Detector det points in TParticle
  AliStack* stack = (AliStack*) gMC->GetStack();
  stack->AddPoint(kAliIts);
  
  // Save old position for the next hit.
  fStartPos = fPos;
  fStartTime = fTime;

  return kTRUE;
}

void AliceO2::ITS::Detector::CreateMaterials()
{
  //Int_t   ifield = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Integ();
  //Float_t fieldm = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Max();
  //FIXME: values taken from the AliMagF constructor. These must (?) be provided by the run_sim macro instead
  Int_t   ifield = 2;
  Float_t fieldm = 10.;

  Float_t tmaxfd = 0.1; // 1.0; // Degree
  Float_t stemax = 1.0; // cm
  Float_t deemax = 0.1; // 30.0; // Fraction of particle's energy 0<deemax<=1
  Float_t epsil  = 1.0E-4; // 1.0; // cm
  Float_t stmin  = 0.0; // cm "Default value used"

  Float_t tmaxfdSi = 0.1; // .10000E+01; // Degree
  Float_t stemaxSi = 0.0075; //  .10000E+01; // cm
  Float_t deemaxSi = 0.1; // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilSi  = 1.0E-4;// .10000E+01;
  Float_t stminSi  = 0.0; // cm "Default value used"

  Float_t tmaxfdAir = 0.1; // .10000E+01; // Degree
  Float_t stemaxAir = .10000E+01; // cm
  Float_t deemaxAir = 0.1; // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilAir  = 1.0E-4;// .10000E+01;
  Float_t stminAir  = 0.0; // cm "Default value used"

  // AIR
  Float_t aAir[4]={12.0107,14.0067,15.9994,39.948};
  Float_t zAir[4]={6.,7.,8.,18.};
  Float_t wAir[4]={0.000124,0.755267,0.231781,0.012827};
  Float_t dAir = 1.20479E-3;

  // Water
  Float_t aWater[2]={1.00794,15.9994};
  Float_t zWater[2]={1.,8.};
  Float_t wWater[2]={0.111894,0.888106};
  Float_t dWater   = 1.0;

  // Kapton
  Float_t aKapton[4]={1.00794,12.0107, 14.010,15.9994};
  Float_t zKapton[4]={1.,6.,7.,8.};
  Float_t wKapton[4]={0.026362,0.69113,0.07327,0.209235};
  Float_t dKapton   = 1.42;
 
  AliMixture(1,"AIR$",aAir,zAir,dAir,4,wAir);
  AliMedium(1, "AIR$",1,0,ifield,fieldm,tmaxfdAir,stemaxAir,deemaxAir,epsilAir,stminAir);

  AliMixture(2,"WATER$",aWater,zWater,dWater,2,wWater);
  AliMedium(2, "WATER$",2,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);

  AliMaterial(3,"SI$",0.28086E+02,0.14000E+02,0.23300E+01,0.93600E+01,0.99900E+03);
  AliMedium(3,  "SI$",3,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);

  AliMaterial(4,"BERILLIUM$",9.01, 4., 1.848, 35.3, 36.7);// From AliPIPEv3
  AliMedium(4,  "BERILLIUM$",4,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);

  AliMaterial(5,"COPPER$",0.63546E+02,0.29000E+02,0.89600E+01,0.14300E+01,0.99900E+03);
  AliMedium(5,  "COPPER$",5,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);
    
  // needed for STAVE , Carbon, kapton, Epoxy, flexcable

  //AliMaterial(6,"CARBON$",12.0107,6,2.210,999,999);
  AliMaterial(6,"CARBON$",12.0107,6,2.210/1.3,999,999);
  AliMedium(6,  "CARBON$",6,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);

  AliMixture(7,"KAPTON(POLYCH2)$", aKapton, zKapton, dKapton, 4, wKapton);
  AliMedium(7, "KAPTON(POLYCH2)$",7,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);

  // values below modified as compared to source AliITSv11 !

  // All types of carbon
  // Unidirectional prepreg
  AliMaterial(8,"K13D2U2k$",12.0107,6,1.643,999,999);
  AliMedium(8,  "K13D2U2k$",8,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);
  //Impregnated thread
  AliMaterial(9,"M60J3K$",12.0107,6,2.21,999,999);
  AliMedium(9,  "M60J3K$",9,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);
  //Impregnated thread
  AliMaterial(10,"M55J6K$",12.0107,6,1.63,999,999);
  AliMedium(10,  "M55J6K$",10,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);
  // Fabric(0/90)
  AliMaterial(11,"T300$",12.0107,6,1.725,999,999);
  AliMedium(11,  "T300$",11,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);
  //AMEC Thermasol
  AliMaterial(12,"FGS003$",12.0107,6,1.6,999,999);
  AliMedium(12,  "FGS003$",12,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);
  // Carbon fleece
  AliMaterial(13,"CarbonFleece$",12.0107,6,0.4,999,999);
  AliMedium(13,  "CarbonFleece$",13,0,ifield,fieldm,tmaxfdSi,stemaxSi,deemaxSi,epsilSi,stminSi);

  // Flex cable
  Float_t aFCm[5]={12.0107,1.00794,14.0067,15.9994,26.981538};
  Float_t zFCm[5]={6.,1.,7.,8.,13.};
  Float_t wFCm[5]={0.520088819984,0.01983871336,0.0551367996,0.157399667056, 0.247536};
  //Float_t dFCm = 1.6087;  // original
  //Float_t dFCm = 2.55;   // conform with STAR
  Float_t dFCm = 2.595;   // conform with Corrado

  AliMixture(14,"FLEXCABLE$",aFCm,zFCm,dFCm,5,wFCm);
  AliMedium(14, "FLEXCABLE$",14,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);

  //AliMaterial(7,"GLUE$",0.12011E+02,0.60000E+01,0.1930E+01/2.015,999,999); // original
  AliMaterial(15,"GLUE$",12.011,6,1.93/2.015,999,999);  // conform with ATLAS, Corrado, Stefan
  AliMedium(15,  "GLUE$",15,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);

  AliMaterial(16,"ALUMINUM$",0.26982E+02,0.13000E+02,0.26989E+01,0.89000E+01,0.99900E+03);
  AliMedium(16,"ALUMINUM$",16,0,ifield,fieldm,tmaxfd,stemax,deemax,epsil,stmin);
}

void AliceO2::ITS::Detector::EndOfEvent()
{
  fO2itsPointCollection->Clear();
}

void AliceO2::ITS::Detector::Register()
{
  /** This will create a branch in the output tree called
      Point, setting the last parameter to kFALSE means:
      this collection will not be written to the file, it will exist
      only during the simulation.
  */

  FairRootManager::Instance()->Register("AliceO2::ITS::Point", "ITS",
                                        fO2itsPointCollection, kTRUE);
}

TClonesArray* AliceO2::ITS::Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0) { return fO2itsPointCollection; }
  else { return NULL; }
}

void AliceO2::ITS::Detector::Reset()
{
  fO2itsPointCollection->Clear();
}

void AliceO2::ITS::Detector::SetNWrapVolumes(Int_t n)
{
  // book arrays for wrapper volumes
  if (fNWrapVol) {
    LOG(FATAL) << fNWrapVol << " wrapper volumes already defined" << FairLogger::endl;
  }
  
  if (n<1) {
    return;
  }
  
  fNWrapVol = n;
  fWrapRMin = new Double_t[fNWrapVol];
  fWrapRMax = new Double_t[fNWrapVol];
  fWrapZSpan= new Double_t[fNWrapVol];
  
  for (int i=fNWrapVol;i--;) {
    fWrapRMin[i]=fWrapRMax[i]=fWrapZSpan[i]=-1;
  }
}

void AliceO2::ITS::Detector::DefineWrapVolume(Int_t id, Double_t rmin,Double_t rmax, Double_t zspan)
{
  // set parameters of id-th wrapper volume
  if (id>=fNWrapVol||id<0) {
    LOG(FATAL)<<"id "<<id<<" of wrapper volume is not in 0-"<<fNWrapVol-1<<" range"<<FairLogger::endl;
  }
  
  fWrapRMin[id] = rmin;
  fWrapRMax[id] = rmax;
  fWrapZSpan[id] = zspan;
}

void AliceO2::ITS::Detector::DefineLayer(Int_t nlay, double phi0, Double_t r,
			    Double_t zlen, Int_t nstav,
			    Int_t nunit, Double_t lthick,
			    Double_t dthick, UInt_t dettypeID,
			    Int_t buildLevel)
{
  //     Sets the layer parameters
  // Inputs:
  //          nlay    layer number
  //          phi0    layer phi0
  //          r       layer radius
  //          zlen    layer length
  //          nstav   number of staves
  //          nunit   IB: number of chips per stave
  //                  OB: number of modules per half stave
  //          lthick  stave thickness (if omitted, defaults to 0)
  //          dthick  detector thickness (if omitted, defaults to 0)
  //          dettypeID  ??
  //          buildLevel (if 0, all geometry is build, used for material budget studies)
  // Outputs:
  //   none.
  // Return:
  //   none.

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " DZ:" << zlen << " Nst:"
						<< nstav << " Nunit:" << nunit << " Lthick:" << lthick << " Dthick:" << dthick
						<< " DetID:" << dettypeID << " B:" << buildLevel << FairLogger::endl;

  if (nlay >= fNLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }
  
  fLayTurbo[nlay] = kFALSE;
  fLayPhi0[nlay] = phi0;
  fLayRadii[nlay] = r;
  fLayZLength[nlay] = zlen;
  fStavPerLay[nlay] = nstav;
  fUnitPerStave[nlay] = nunit;
  fStaveThick[nlay] = lthick;
  fDetThick[nlay] = dthick;
  fChipTypeID[nlay] = dettypeID;
  fBuildLevel[nlay] = buildLevel;
}

void AliceO2::ITS::Detector::DefineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nstav,
				 Int_t nunit, Double_t width, Double_t tilt,
				 Double_t lthick,Double_t dthick,
				 UInt_t dettypeID, Int_t buildLevel)
{
  //     Sets the layer parameters for a "turbo" layer
  //     (i.e. a layer whose staves overlap in phi)
  // Inputs:
  //          nlay    layer number
  //          phi0    phi of 1st stave
  //          r       layer radius
  //          zlen    layer length
  //          nstav   number of staves
  //          nunit   IB: number of chips per stave
  //                  OB: number of modules per half stave
  //          width   stave width
  //          tilt    layer tilt angle (degrees)
  //          lthick  stave thickness (if omitted, defaults to 0)
  //          dthick  detector thickness (if omitted, defaults to 0)
  //          dettypeID  ??
  //          buildLevel (if 0, all geometry is build, used for material budget studies)
  // Outputs:
  //   none.
  // Return:
  //   none.

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " DZ:" << zlen << " Nst:"
						<< nstav << " Nunit:" << nunit << " W:" << width << " Tilt:" << tilt << " Lthick:"
						<< lthick << " Dthick:" << dthick << " DetID:" << dettypeID << " B:" << buildLevel
						<< FairLogger::endl;

  if (nlay >= fNLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }

  fLayTurbo[nlay] = kTRUE;
  fLayPhi0[nlay] = phi0;
  fLayRadii[nlay] = r;
  fLayZLength[nlay] = zlen;
  fStavPerLay[nlay] = nstav;
  fUnitPerStave[nlay] = nunit;
  fStaveThick[nlay] = lthick;
  fStaveWidth[nlay] = width;
  fStaveTilt[nlay] = tilt;
  fDetThick[nlay] = dthick;
  fChipTypeID[nlay] = dettypeID;
  fBuildLevel[nlay] = buildLevel;
}

void AliceO2::ITS::Detector::GetLayerParameters(Int_t nlay, Double_t &phi0,
				   Double_t &r, Double_t &zlen,
				   Int_t &nstav, Int_t &nmod,
				   Double_t &width, Double_t &tilt,
				   Double_t &lthick, Double_t &dthick,
				   UInt_t &dettype) const
{
  //     Gets the layer parameters
  // Inputs:
  //          nlay    layer number
  // Outputs:
  //          phi0    phi of 1st stave
  //          r       layer radius
  //          zlen    layer length
  //          nstav   number of staves
  //          nmod    IB: number of chips per stave
  //                  OB: number of modules per half stave
  //          width   stave width
  //          tilt    stave tilt angle
  //          lthick  stave thickness
  //          dthick  detector thickness
  //          dettype detector type
  // Return:
  //   none.

  if (nlay >= fNLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }
  
  phi0   = fLayPhi0[nlay];
  r      = fLayRadii[nlay];
  zlen   = fLayZLength[nlay];
  nstav  = fStavPerLay[nlay];
  nmod   = fUnitPerStave[nlay];
  width  = fStaveWidth[nlay];
  tilt   = fStaveTilt[nlay];
  lthick = fStaveThick[nlay];
  dthick = fDetThick[nlay];
  dettype= fChipTypeID[nlay];
}

TGeoVolume* AliceO2::ITS::Detector::CreateWrapperVolume(Int_t id)
{
	// Creates an air-filled wrapper cylindrical volume

  if (fWrapRMin[id]<0 || fWrapRMax[id]<0 || fWrapZSpan[id]<0) {
    LOG(FATAL) << "Wrapper volume " << id << " was requested but not defined" << FairLogger::endl;
  }
  
  // Now create the actual shape and volume
  TGeoTube *tube = new TGeoTube(fWrapRMin[id], fWrapRMax[id], fWrapZSpan[id]/2.);

  TGeoMedium *medAir = gGeoManager->GetMedium("ITS_AIR$");

  char volnam[30];
  snprintf(volnam, 29, "%s%d", UpgradeGeometryTGeo::GetITSWrapVolPattern(),id);

  TGeoVolume *wrapper = new TGeoVolume(volnam, tube, medAir);

  return wrapper;
}

void AliceO2::ITS::Detector::ConstructGeometry()
{
  // Create the detector materials
  CreateMaterials();
  
  // Construct the detector geometry
  ConstructDetectorGeometry();
  
  // Define the list of sensitive volumes
  DefineSensitiveVolumes();
}

void AliceO2::ITS::Detector::ConstructDetectorGeometry()
{
  // Create the geometry and insert it in the mother volume ITSV
  TGeoManager *geoManager = gGeoManager;

  TGeoVolume *vALIC = geoManager->GetVolume("cave");
  
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume" << FairLogger::endl;
  }

  new TGeoVolumeAssembly(UpgradeGeometryTGeo::GetITSVolPattern());
  TGeoVolume *vITSV = geoManager->GetVolume(UpgradeGeometryTGeo::GetITSVolPattern());
  vITSV->SetUniqueID(UpgradeGeometryTGeo::GetUIDShift()); // store modID -> midUUID bitshift
  vALIC->AddNode(vITSV, 2, 0);  // Copy number is 2 to cheat AliGeoManager::CheckSymNamesLUT

  const Int_t kLength=100;
  Char_t vstrng[kLength] = "xxxRS"; //?
  vITSV->SetTitle(vstrng);

  // Check that we have all needed parameters
  if (fNLayers <= 0) {
    LOG(FATAL) << "Wrong number of layers (" << fNLayers << ")" << FairLogger::endl;
  }

  for (Int_t j=0; j<fNLayers; j++) {
    if (fLayRadii[j] <= 0) {
      LOG(FATAL) << "Wrong layer radius for layer " << j << "(" << fLayRadii[j] << ")"
								 << FairLogger::endl;
    }
    if (fLayZLength[j] <= 0) {
      LOG(FATAL) << "Wrong layer length for layer " << j << "(" << fLayZLength[j] << ")"
								 << FairLogger::endl;
    }
    if (fStavPerLay[j] <= 0) {
      LOG(FATAL) << "Wrong number of staves for layer " << j << "(" << fStavPerLay[j] << ")"
								 << FairLogger::endl;
    }
    if (fUnitPerStave[j] <= 0) {
      LOG(FATAL) << "Wrong number of chips for layer " << j << "(" << fUnitPerStave[j] << ")"
								 << FairLogger::endl;
    }
    if (fStaveThick[j] < 0) {
      LOG(FATAL) << "Wrong stave thickness for layer " << j << "(" << fStaveThick[j] << ")"
								 << FairLogger::endl;
    }
    if (fLayTurbo[j] && fStaveWidth[j] <= 0) {
      LOG(FATAL) << "Wrong stave width for layer " << j << "(" << fStaveWidth[j] << ")"
								 << FairLogger::endl;
    }
    if (fDetThick[j] < 0) {
      LOG(FATAL) << "Wrong chip thickness for layer " << j << "(" << fDetThick[j] << ")"
								 << FairLogger::endl;
    }

    if (j > 0) {
      if (fLayRadii[j]<=fLayRadii[j-1]) {
        LOG(FATAL) << "Layer " << j << " radius (" << fLayRadii[j] << ") is smaller than layer "
                  << j-1 << " radius (" << fLayRadii[j-1] << ")" << FairLogger::endl;
      }
    }

    if (fStaveThick[j] == 0) {
      LOG(INFO) << "Stave thickness for layer " << j << " not set, using default"
							  << FairLogger::endl;
    }
    if (fDetThick[j] == 0) {
      LOG(INFO) << "Chip thickness for layer " << j << " not set, using default"
								<< FairLogger::endl;
    }
  }

  // Create the wrapper volumes
  TGeoVolume **wrapVols = 0;
  
  if (fNWrapVol) {
    wrapVols = new TGeoVolume*[fNWrapVol];
    for (int id=0;id<fNWrapVol;id++) {
      wrapVols[id] = CreateWrapperVolume(id);
      vITSV->AddNode(wrapVols[id], 1, 0);
    }
  }

  fLay2WrapV = new Int_t[fNLayers];

  // Now create the actual geometry
  for (Int_t j=0; j<fNLayers; j++) {
    TGeoVolume* dest = vITSV;
		fLay2WrapV[j] = -1;

    if (fLayTurbo[j]) {
      fUpGeom[j] = new UpgradeV1Layer(j,kTRUE,kFALSE);
      fUpGeom[j]->SetStaveWidth(fStaveWidth[j]);
      fUpGeom[j]->SetStaveTilt(fStaveTilt[j]);
    } else {
      fUpGeom[j] = new UpgradeV1Layer(j,kFALSE);
    }

    fUpGeom[j]->SetPhi0(fLayPhi0[j]);
    fUpGeom[j]->SetRadius(fLayRadii[j]);
    fUpGeom[j]->SetZLength(fLayZLength[j]);
    fUpGeom[j]->SetNStaves(fStavPerLay[j]);
    fUpGeom[j]->SetNUnits(fUnitPerStave[j]);
    fUpGeom[j]->SetChipType(fChipTypeID[j]);
    fUpGeom[j]->SetBuildLevel(fBuildLevel[j]);
    
    if (j < 3) {
      fUpGeom[j]->SetStaveModel(fStaveModelIB);
    } else {
      fUpGeom[j]->SetStaveModel(fStaveModelOB);
    }
    
    LOG(DEBUG1) << "fBuildLevel: " << fBuildLevel[j] << FairLogger::endl;
    
    if (fStaveThick[j] != 0) {
      fUpGeom[j]->SetStaveThick(fStaveThick[j]);
    }
    if (fDetThick[j] != 0) {
      fUpGeom[j]->SetSensorThick(fDetThick[j]);
    }
    
    for (int iw=0;iw<fNWrapVol;iw++) {
      if (fLayRadii[j]>fWrapRMin[iw] && fLayRadii[j]<fWrapRMax[iw]) {
        LOG(INFO) << "Will embed layer " << j << " in wrapper volume " << iw << FairLogger::endl;
        
        if (fLayZLength[j]>=fWrapZSpan[iw]) {
          LOG(FATAL) << "ZSpan " << fWrapZSpan[iw] << " of wrapper volume " << iw
										 << " is less than ZSpan " << fLayZLength[j] << " of layer " << j << FairLogger::endl;
        }

        dest = wrapVols[iw];
				fLay2WrapV[j] = iw;
        break;
      }
    }
    fUpGeom[j]->CreateLayer(dest);
  }
	CreateSuppCyl(kTRUE,wrapVols[0]);
  CreateSuppCyl(kFALSE,wrapVols[2]);

  delete[] wrapVols; // delete pointer only, not the volumes
}

//Service Barrel
void AliceO2::ITS::Detector::CreateSuppCyl(const Bool_t innerBarrel,TGeoVolume *dest,const TGeoManager *mgr){
  // Creates the Service Barrel (as a simple cylinder) for IB and OB
  // Inputs:
  //         innerBarrel : if true, build IB service barrel, otherwise for OB
  //         dest        : the mother volume holding the service barrel
  //         mgr         : the gGeoManager pointer (used to get the material)
  //

  Double_t rminIB =  4.7;
  Double_t rminOB = 43.4;
  Double_t zLenOB ;
  Double_t cInt	= 0.22; //dimensioni cilindro di supporto interno
  Double_t cExt	= 1.00; //dimensioni cilindro di supporto esterno
//  Double_t phi1   =  180;
//  Double_t phi2   =  360;


  TGeoMedium *medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  if (innerBarrel){
    zLenOB=((TGeoTube*)(dest->GetShape()))->GetDz();
//    TGeoTube*ibSuppSh = new TGeoTubeSeg(rminIB,rminIB+cInt,zLenOB,phi1,phi2);
    TGeoTube*ibSuppSh = new TGeoTube(rminIB,rminIB+cInt,zLenOB);
    TGeoVolume *ibSupp = new TGeoVolume("ibSuppCyl",ibSuppSh,medCarbonFleece);
    dest->AddNode(ibSupp,1);
  }
  else {
    zLenOB=((TGeoTube*)(dest->GetShape()))->GetDz();
    TGeoTube*obSuppSh=new TGeoTube(rminOB,rminOB+cExt,zLenOB);
    TGeoVolume *obSupp=new TGeoVolume("obSuppCyl",obSuppSh,medCarbonFleece);
    dest->AddNode(obSupp,1);
  }

  return;
}

void AliceO2::ITS::Detector::DefineSensitiveVolumes()
{
  TGeoManager *geoManager = gGeoManager;
  TGeoVolume *v;
  
  TString volumeName;
  
  // The names of the ITS sensitive volumes have the format: ITSUSensor(0...fNLayers-1)
  for (Int_t j=0; j<fNLayers; j++) {
    volumeName = UpgradeGeometryTGeo::GetITSSensorPattern() + TString::Itoa(j, 10);
    v = geoManager->GetVolume(volumeName.Data());
    AddSensitiveVolume(v);
  }
}

Point* AliceO2::ITS::Detector::AddHit(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
                          Double_t startTime, Double_t time, Double_t length, Double_t eLoss,
                          Int_t shunt)
{
  TClonesArray& clref = *fO2itsPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new(clref[size]) Point(trackID, detID, startPos, pos, mom, 
         startTime, time, length, eLoss, shunt);
}

TParticle* AliceO2::ITS::Detector::GetParticle() const
{
    // Returns the pointer to the TParticle for the particle that created
    // this hit. From the TParticle all kinds of information about this 
    // particle can be found. See the TParticle class.
    // Inputs:
    //   none.
    // Outputs:
    //   none.
    // Return:
    //   The TParticle of the track that created this hit.

   return ((AliStack*) gMC->GetStack())->GetParticle(GetTrack());
}

void AliceO2::ITS::Detector::Print(ostream *os) const
{
    // Standard output format for this class.
    // Inputs:
    //   ostream *os   The output stream
    // Outputs:
    //   none.
    // Return:
    //   none.

#if defined __GNUC__
#if __GNUC__ > 2
    ios::fmtflags fmt;
#else
    Int_t fmt;
#endif
#else
#if defined __ICC || defined __ECC || defined __xlC__
    ios::fmtflags fmt;
#else
    Int_t fmt;
#endif
#endif
 
    fmt = os->setf(ios::scientific);  // set scientific floating point output
    *os << fTrack << " " << fX << " " << fY << " " << fZ << " ";
    fmt = os->setf(ios::hex); // set hex for fStatus only.
    *os << fStatus << " ";
    fmt = os->setf(ios::dec); // every thing else decimel.
    *os << fModule << " ";
    *os << fPx << " " << fPy << " " << fPz << " ";
    *os << fDestep << " " << fTof;
    *os << " " << fx0 << " " << fy0 << " " << fz0;
//    *os << " " << endl;
    os->flags(fmt); // reset back to old formating.
    return;
}

void AliceO2::ITS::Detector::Read(istream *is)
{
    // Standard input format for this class.
    // Inputs:
    //   istream *is  the input stream
    // Outputs:
    //   none.
    // Return:
    //   none.

    *is >> fTrack >> fX >> fY >> fZ;
    *is >> fStatus >> fModule >> fPx >> fPy >> fPz >> fDestep >> fTof;
    *is >> fx0 >> fy0 >> fz0;
    return;
}

ostream &operator<<(ostream &os, AliceO2::ITS::Detector &p)
{
    // Standard output streaming function.
    // Inputs:
    //   ostream os  The output stream
    //   Detector p The his to be printed out
    // Outputs:
    //   none.
    // Return:
    //   The input stream

    p.Print(&os);
    return os;
}

istream &operator>>(istream &is, AliceO2::ITS::Detector &r)
{
    // Standard input streaming function.
    // Inputs:
    //   istream is  The input stream
    //   Detector p The Detector class to be filled from this input stream
    // Outputs:
    //   none.
    // Return:
    //   The input stream

    r.Read(&is);
    return is;
}

ClassImp(AliceO2::ITS::Detector)
