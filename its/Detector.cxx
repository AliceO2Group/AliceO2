/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "Detector.h"

#include "Point.h"
#include "UpgradeV1Layer.h"
#include "UpgradeGeometryTGeo.h"

#include "Data/DetectorList.h"
#include "Data/Stack.h"

#include "FairVolume.h"
#include "FairGeoVolume.h"
#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairGeoLoader.h"
#include "FairGeoInterface.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"

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
  : AliceO2::Base::Detector("ITS", kTRUE, kAliIts),
    mLayerID(0),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mEntrancePosition(),
    mPosition(),
    mMomentum(),
    mEntranceTime(-1.),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mShunt(),
    mPointCollection(new TClonesArray("AliceO2::ITS::Point")),
    mGeometryHandler(new GeometryHandler()),
    mMisalignmentParameter(NULL),
    mNumberOfDetectors(-1),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),
    mModifyGeometry(kFALSE),
    mNumberOfWrapperVolumes(0),
    mWrapperMinRadius(0),
    mWrapperMaxRadius(0),
    mWrapperZSpan(0),
    mWrapperLayerId(0),
    mTurboLayer(0),
    mLayerPhi0(0),
    mLayerRadii(0),
    mLayerZLength(0),
    mStavePerLayer(0),
    mUnitPerStave(0),
    mStaveThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mDetectorThickness(0),
    mChipTypeID(0),
    mBuildLevel(0),
    mUpgradeGeometry(0),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{
}

AliceO2::ITS::Detector::Detector(const char* name, Bool_t active, const Int_t nlay)
  : AliceO2::Base::Detector(name, active, kAliIts),
    mLayerID(0),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mEntrancePosition(),
    mPosition(),
    mMomentum(),
    mEntranceTime(-1.),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mShunt(),
    mPointCollection(new TClonesArray("AliceO2::ITS::Point")),
    mGeometryHandler(new GeometryHandler()),
    mMisalignmentParameter(NULL),
    mNumberOfDetectors(-1),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),
    mModifyGeometry(kFALSE),
    mNumberOfWrapperVolumes(0),
    mWrapperMinRadius(0),
    mWrapperMaxRadius(0),
    mWrapperZSpan(0),
    mWrapperLayerId(0),
    mTurboLayer(0),
    mLayerPhi0(0),
    mLayerRadii(0),
    mLayerZLength(0),
    mStavePerLayer(0),
    mUnitPerStave(0),
    mStaveThickness(0),
    mStaveWidth(0),
    mStaveTilt(0),
    mDetectorThickness(0),
    mChipTypeID(0),
    mBuildLevel(0),
    mUpgradeGeometry(0),
    mNumberLayers(nlay),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{
  mLayerName = new TString[mNumberLayers];

  for (Int_t j = 0; j < mNumberLayers; j++)
    mLayerName[j].Form("%s%d", UpgradeGeometryTGeo::GetITSSensorPattern(), j); // See UpgradeV1Layer

  mTurboLayer = new Bool_t[mNumberLayers];
  mLayerPhi0 = new Double_t[mNumberLayers];
  mLayerRadii = new Double_t[mNumberLayers];
  mLayerZLength = new Double_t[mNumberLayers];
  mStavePerLayer = new Int_t[mNumberLayers];
  mUnitPerStave = new Int_t[mNumberLayers];
  mStaveThickness = new Double_t[mNumberLayers];
  mStaveWidth = new Double_t[mNumberLayers];
  mStaveTilt = new Double_t[mNumberLayers];
  mDetectorThickness = new Double_t[mNumberLayers];
  mChipTypeID = new UInt_t[mNumberLayers];
  mBuildLevel = new Int_t[mNumberLayers];

  mUpgradeGeometry = new UpgradeV1Layer* [mNumberLayers];

  if (mNumberLayers > 0) { // if not, we'll Fatal-ize in CreateGeometry
    for (Int_t j = 0; j < mNumberLayers; j++) {
      mLayerPhi0[j] = 0;
      mLayerRadii[j] = 0.;
      mLayerZLength[j] = 0.;
      mStavePerLayer[j] = 0;
      mUnitPerStave[j] = 0;
      mStaveWidth[j] = 0.;
      mDetectorThickness[j] = 0.;
      mChipTypeID[j] = 0;
      mBuildLevel[j] = 0;
      mUpgradeGeometry[j] = 0;
    }
  }
}

AliceO2::ITS::Detector::~Detector()
{
  delete[] mTurboLayer;
  delete[] mLayerPhi0;
  delete[] mLayerRadii;
  delete[] mLayerZLength;
  delete[] mStavePerLayer;
  delete[] mUnitPerStave;
  delete[] mStaveThickness;
  delete[] mStaveWidth;
  delete[] mStaveTilt;
  delete[] mDetectorThickness;
  delete[] mChipTypeID;
  delete[] mBuildLevel;
  delete[] mUpgradeGeometry;
  delete[] mWrapperMinRadius;
  delete[] mWrapperMaxRadius;
  delete[] mWrapperZSpan;
  delete[] mWrapperLayerId;

  if (mPointCollection) {
    mPointCollection->Delete();
    delete mPointCollection;
  }

  delete[] mLayerID;
}

AliceO2::ITS::Detector& AliceO2::ITS::Detector::operator=(const AliceO2::ITS::Detector& h)
{
  // The standard = operator
  // Inputs:
  //   Detector   &h the sourse of this copy
  // Outputs:
  //   none.
  // Return:
  //  A copy of the sourse hit h

  if (this == &h) {
    return *this;
  }
  this->mStatus = h.mStatus;
  this->mModule = h.mModule;
  this->mParticlePx = h.mParticlePx;
  this->mParticlePy = h.mParticlePy;
  this->mParticlePz = h.mParticlePz;
  this->mEnergyDepositionStep = h.mEnergyDepositionStep;
  this->mTof = h.mTof;
  this->mStatus0 = h.mStatus0;
  this->mStartingStepX = h.mStartingStepX;
  this->mStartingStepY = h.mStartingStepY;
  this->mStartingStepZ = h.mStartingStepZ;
  this->mStartingStepT = h.mStartingStepT;
  return *this;
}

void AliceO2::ITS::Detector::Initialize()
{
  if (!mLayerID) {
    mLayerID = new Int_t[mNumberLayers];
  }

  for (int i = 0; i < mNumberLayers; i++) {
    mLayerID[i] = gMC ? gMC->VolId(mLayerName[i]) : 0;
  }

  mGeometryTGeo = new UpgradeGeometryTGeo(kTRUE);

  FairDetector::Initialize();

  //  FairRuntimeDb* rtdb= FairRun::Instance()->GetRuntimeDb();
  //  O2itsGeoPar* par=(O2itsGeoPar*)(rtdb->getContainer("O2itsGeoPar"));
}

void AliceO2::ITS::Detector::InitParameterContainers()
{
  LOG(INFO) << "Initialize aliitsdet misallign parameters" << FairLogger::endl;
  mNumberOfDetectors = mMisalignmentParameter->GetNumberOfDetectors();
  mShiftX = mMisalignmentParameter->GetShiftX();
  mShiftY = mMisalignmentParameter->GetShiftY();
  mShiftZ = mMisalignmentParameter->GetShiftZ();
  mRotX = mMisalignmentParameter->GetRotX();
  mRotY = mMisalignmentParameter->GetRotY();
  mRotZ = mMisalignmentParameter->GetRotZ();
}

void AliceO2::ITS::Detector::SetParameterContainers()
{
  LOG(INFO) << "Set tutdet misallign parameters" << FairLogger::endl;
  // Get Base Container
  FairRun* sim = FairRun::Instance();
  LOG_IF(FATAL, !sim) << "No run object" << FairLogger::endl;
  FairRuntimeDb* rtdb = sim->GetRuntimeDb();
  LOG_IF(FATAL, !rtdb) << "No runtime database" << FairLogger::endl;

  mMisalignmentParameter = (MisalignmentParameter*)(rtdb->getContainer("MisallignmentParameter"));
}

Bool_t AliceO2::ITS::Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(gMC->TrackCharge())) {
    return kFALSE;
  }

  // FIXME: Is copy actually needed?
  Int_t copy = vol->getCopyNo();
  Int_t id = vol->getMCid();
  Int_t lay = 0;
  Int_t cpn0, cpn1, mod;

  // FIXME: Determine the layer number. Is this information available directly from the FairVolume?
  while ((lay < mNumberLayers) && id != mLayerID[lay]) {
    ++lay;
  }

  // FIXME: Is it needed to keep a track reference when the outer ITS volume is encountered?
  // if(gMC->IsTrackExiting()) {
  //  AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kITS);
  // } // if Outer ITS mother Volume

  // Retrieve the indices with the volume path
  copy = 1;
  gMC->CurrentVolOffID(1, cpn1);
  gMC->CurrentVolOffID(2, cpn0);

  mod = mGeometryTGeo->GetChipIndex(lay, cpn0, cpn1);

  // Record information on the points
  mEnergyLoss = gMC->Edep();
  mTime = gMC->TrackTime();
  mTrackNumberID = gMC->GetStack()->GetCurrentTrackNumber();
  mVolumeID = vol->getMCid();

  // FIXME: Set a temporary value to mShunt for now, determine its use at a later stage
  mShunt = 0;

  gMC->TrackPosition(mPosition);
  gMC->TrackMomentum(mMomentum);

  // mLength = gMC->TrackLength();

  if (gMC->IsTrackEntering()) {
    mEntrancePosition = mPosition;
    mEntranceTime = mTime;
    return kFALSE; // don't save entering hit.
  }

  // Create Point on every step of the active volume
  AddHit(mTrackNumberID, mVolumeID,
         TVector3(mEntrancePosition.X(), mEntrancePosition.Y(), mEntrancePosition.Z()),
         TVector3(mPosition.X(), mPosition.Y(), mPosition.Z()),
         TVector3(mMomentum.Px(), mMomentum.Py(), mMomentum.Pz()), mEntranceTime, mTime, mLength,
         mEnergyLoss, mShunt);

  // Increment number of Detector det points in TParticle
  AliceO2::Data::Stack* stack = (AliceO2::Data::Stack*)gMC->GetStack();
  stack->AddPoint(kAliIts);

  // Save old position for the next hit.
  mEntrancePosition = mPosition;
  mEntranceTime = mTime;

  return kTRUE;
}

void AliceO2::ITS::Detector::CreateMaterials()
{
  // Int_t   ifield = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Integ();
  // Float_t fieldm = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Max();
  // FIXME: values taken from the AliMagF constructor. These must (?) be provided by the run_sim
  // macro instead
  Int_t ifield = 2;
  Float_t fieldm = 10.;

  Float_t tmaxfd = 0.1;   // 1.0; // Degree
  Float_t stemax = 1.0;   // cm
  Float_t deemax = 0.1;   // 30.0; // Fraction of particle's energy 0<deemax<=1
  Float_t epsil = 1.0E-4; // 1.0; // cm
  Float_t stmin = 0.0;    // cm "Default value used"

  Float_t tmaxfdSi = 0.1;    // .10000E+01; // Degree
  Float_t stemaxSi = 0.0075; //  .10000E+01; // cm
  Float_t deemaxSi = 0.1;    // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilSi = 1.0E-4;  // .10000E+01;
  Float_t stminSi = 0.0;     // cm "Default value used"

  Float_t tmaxfdAir = 0.1;        // .10000E+01; // Degree
  Float_t stemaxAir = .10000E+01; // cm
  Float_t deemaxAir = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilAir = 1.0E-4;      // .10000E+01;
  Float_t stminAir = 0.0;         // cm "Default value used"

  // AIR
  Float_t aAir[4] = { 12.0107, 14.0067, 15.9994, 39.948 };
  Float_t zAir[4] = { 6., 7., 8., 18. };
  Float_t wAir[4] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 1.20479E-3;

  // Water
  Float_t aWater[2] = { 1.00794, 15.9994 };
  Float_t zWater[2] = { 1., 8. };
  Float_t wWater[2] = { 0.111894, 0.888106 };
  Float_t dWater = 1.0;

  // Kapton
  Float_t aKapton[4] = { 1.00794, 12.0107, 14.010, 15.9994 };
  Float_t zKapton[4] = { 1., 6., 7., 8. };
  Float_t wKapton[4] = { 0.026362, 0.69113, 0.07327, 0.209235 };
  Float_t dKapton = 1.42;

  AliceO2::Base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  AliceO2::Base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir,
                                  epsilAir, stminAir);

  AliceO2::Base::Detector::Mixture(2, "WATER$", aWater, zWater, dWater, 2, wWater);
  AliceO2::Base::Detector::Medium(2, "WATER$", 2, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  AliceO2::Base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01,
                                    0.99900E+03);
  AliceO2::Base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);

  AliceO2::Base::Detector::Material(4, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7); // From AliPIPEv3
  AliceO2::Base::Detector::Medium(4, "BERILLIUM$", 4, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);

  AliceO2::Base::Detector::Material(5, "COPPER$", 0.63546E+02, 0.29000E+02, 0.89600E+01,
                                    0.14300E+01, 0.99900E+03);
  AliceO2::Base::Detector::Medium(5, "COPPER$", 5, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  // needed for STAVE , Carbon, kapton, Epoxy, flexcable

  // AliceO2::Base::Detector::Material(6,"CARBON$",12.0107,6,2.210,999,999);
  AliceO2::Base::Detector::Material(6, "CARBON$", 12.0107, 6, 2.210 / 1.3, 999, 999);
  AliceO2::Base::Detector::Medium(6, "CARBON$", 6, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);

  AliceO2::Base::Detector::Mixture(7, "KAPTON(POLYCH2)$", aKapton, zKapton, dKapton, 4, wKapton);
  AliceO2::Base::Detector::Medium(7, "KAPTON(POLYCH2)$", 7, 0, ifield, fieldm, tmaxfd, stemax,
                                  deemax, epsil, stmin);

  // values below modified as compared to source AliITSv11 !

  // All types of carbon
  // Unidirectional prepreg
  AliceO2::Base::Detector::Material(8, "K13D2U2k$", 12.0107, 6, 1.643, 999, 999);
  AliceO2::Base::Detector::Medium(8, "K13D2U2k$", 8, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Impregnated thread
  AliceO2::Base::Detector::Material(9, "M60J3K$", 12.0107, 6, 2.21, 999, 999);
  AliceO2::Base::Detector::Medium(9, "M60J3K$", 9, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);
  // Impregnated thread
  AliceO2::Base::Detector::Material(10, "M55J6K$", 12.0107, 6, 1.63, 999, 999);
  AliceO2::Base::Detector::Medium(10, "M55J6K$", 10, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Fabric(0/90)
  AliceO2::Base::Detector::Material(11, "T300$", 12.0107, 6, 1.725, 999, 999);
  AliceO2::Base::Detector::Medium(11, "T300$", 11, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);
  // AMEC Thermasol
  AliceO2::Base::Detector::Material(12, "FGS003$", 12.0107, 6, 1.6, 999, 999);
  AliceO2::Base::Detector::Medium(12, "FGS003$", 12, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Carbon fleece
  AliceO2::Base::Detector::Material(13, "CarbonFleece$", 12.0107, 6, 0.4, 999, 999);
  AliceO2::Base::Detector::Medium(13, "CarbonFleece$", 13, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);

  // Flex cable
  Float_t aFCm[5] = { 12.0107, 1.00794, 14.0067, 15.9994, 26.981538 };
  Float_t zFCm[5] = { 6., 1., 7., 8., 13. };
  Float_t wFCm[5] = { 0.520088819984, 0.01983871336, 0.0551367996, 0.157399667056, 0.247536 };
  // Float_t dFCm = 1.6087;  // original
  // Float_t dFCm = 2.55;   // conform with STAR
  Float_t dFCm = 2.595; // conform with Corrado

  AliceO2::Base::Detector::Mixture(14, "FLEXCABLE$", aFCm, zFCm, dFCm, 5, wFCm);
  AliceO2::Base::Detector::Medium(14, "FLEXCABLE$", 14, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);

  // AliceO2::Base::Detector::Material(7,"GLUE$",0.12011E+02,0.60000E+01,0.1930E+01/2.015,999,999);
  // // original
  AliceO2::Base::Detector::Material(15, "GLUE$", 12.011, 6, 1.93 / 2.015, 999,
                                    999); // conform with ATLAS, Corrado, Stefan
  AliceO2::Base::Detector::Medium(15, "GLUE$", 15, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  AliceO2::Base::Detector::Material(16, "ALUMINUM$", 0.26982E+02, 0.13000E+02, 0.26989E+01,
                                    0.89000E+01, 0.99900E+03);
  AliceO2::Base::Detector::Medium(16, "ALUMINUM$", 16, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);
}

void AliceO2::ITS::Detector::EndOfEvent()
{
  mPointCollection->Clear();
}

void AliceO2::ITS::Detector::Register()
{
  // This will create a branch in the output tree called Point, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  FairRootManager::Instance()->Register("AliceO2::ITS::Point", "ITS", mPointCollection, kTRUE);
}

TClonesArray* AliceO2::ITS::Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0) {
    return mPointCollection;
  } else {
    return NULL;
  }
}

void AliceO2::ITS::Detector::Reset()
{
  mPointCollection->Clear();
}

void AliceO2::ITS::Detector::SetNumberOfWrapperVolumes(Int_t n)
{
  // book arrays for wrapper volumes
  if (mNumberOfWrapperVolumes) {
    LOG(FATAL) << mNumberOfWrapperVolumes << " wrapper volumes already defined" << FairLogger::endl;
  }

  if (n < 1) {
    return;
  }

  mNumberOfWrapperVolumes = n;
  mWrapperMinRadius = new Double_t[mNumberOfWrapperVolumes];
  mWrapperMaxRadius = new Double_t[mNumberOfWrapperVolumes];
  mWrapperZSpan = new Double_t[mNumberOfWrapperVolumes];

  for (int i = mNumberOfWrapperVolumes; i--;) {
    mWrapperMinRadius[i] = mWrapperMaxRadius[i] = mWrapperZSpan[i] = -1;
  }
}

void AliceO2::ITS::Detector::DefineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax,
                                                 Double_t zspan)
{
  // set parameters of id-th wrapper volume
  if (id >= mNumberOfWrapperVolumes || id < 0) {
    LOG(FATAL) << "id " << id << " of wrapper volume is not in 0-" << mNumberOfWrapperVolumes - 1
               << " range" << FairLogger::endl;
  }

  mWrapperMinRadius[id] = rmin;
  mWrapperMaxRadius[id] = rmax;
  mWrapperZSpan[id] = zspan;
}

void AliceO2::ITS::Detector::DefineLayer(Int_t nlay, double phi0, Double_t r, Double_t zlen,
                                         Int_t nstav, Int_t nunit, Double_t lthick, Double_t dthick,
                                         UInt_t dettypeID, Int_t buildLevel)
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

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " DZ:" << zlen << " Nst:" << nstav
            << " Nunit:" << nunit << " Lthick:" << lthick << " Dthick:" << dthick
            << " DetID:" << dettypeID << " B:" << buildLevel << FairLogger::endl;

  if (nlay >= mNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }

  mTurboLayer[nlay] = kFALSE;
  mLayerPhi0[nlay] = phi0;
  mLayerRadii[nlay] = r;
  mLayerZLength[nlay] = zlen;
  mStavePerLayer[nlay] = nstav;
  mUnitPerStave[nlay] = nunit;
  mStaveThickness[nlay] = lthick;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
}

void AliceO2::ITS::Detector::DefineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen,
                                              Int_t nstav, Int_t nunit, Double_t width,
                                              Double_t tilt, Double_t lthick, Double_t dthick,
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

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " DZ:" << zlen << " Nst:" << nstav
            << " Nunit:" << nunit << " W:" << width << " Tilt:" << tilt << " Lthick:" << lthick
            << " Dthick:" << dthick << " DetID:" << dettypeID << " B:" << buildLevel
            << FairLogger::endl;

  if (nlay >= mNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }

  mTurboLayer[nlay] = kTRUE;
  mLayerPhi0[nlay] = phi0;
  mLayerRadii[nlay] = r;
  mLayerZLength[nlay] = zlen;
  mStavePerLayer[nlay] = nstav;
  mUnitPerStave[nlay] = nunit;
  mStaveThickness[nlay] = lthick;
  mStaveWidth[nlay] = width;
  mStaveTilt[nlay] = tilt;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
}

void AliceO2::ITS::Detector::GetLayerParameters(Int_t nlay, Double_t& phi0, Double_t& r,
                                                Double_t& zlen, Int_t& nstav, Int_t& nmod,
                                                Double_t& width, Double_t& tilt, Double_t& lthick,
                                                Double_t& dthick, UInt_t& dettype) const
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

  if (nlay >= mNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay << FairLogger::endl;
    return;
  }

  phi0 = mLayerPhi0[nlay];
  r = mLayerRadii[nlay];
  zlen = mLayerZLength[nlay];
  nstav = mStavePerLayer[nlay];
  nmod = mUnitPerStave[nlay];
  width = mStaveWidth[nlay];
  tilt = mStaveTilt[nlay];
  lthick = mStaveThickness[nlay];
  dthick = mDetectorThickness[nlay];
  dettype = mChipTypeID[nlay];
}

TGeoVolume* AliceO2::ITS::Detector::CreateWrapperVolume(Int_t id)
{
  // Creates an air-filled wrapper cylindrical volume

  if (mWrapperMinRadius[id] < 0 || mWrapperMaxRadius[id] < 0 || mWrapperZSpan[id] < 0) {
    LOG(FATAL) << "Wrapper volume " << id << " was requested but not defined" << FairLogger::endl;
  }

  // Now create the actual shape and volume
  TGeoTube* tube =
    new TGeoTube(mWrapperMinRadius[id], mWrapperMaxRadius[id], mWrapperZSpan[id] / 2.);

  TGeoMedium* medAir = gGeoManager->GetMedium("ITS_AIR$");

  char volnam[30];
  snprintf(volnam, 29, "%s%d", UpgradeGeometryTGeo::GetITSWrapVolPattern(), id);

  TGeoVolume* wrapper = new TGeoVolume(volnam, tube, medAir);

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
  TGeoManager* geoManager = gGeoManager;

  TGeoVolume* vALIC = geoManager->GetVolume("cave");

  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume" << FairLogger::endl;
  }

  new TGeoVolumeAssembly(UpgradeGeometryTGeo::GetITSVolPattern());
  TGeoVolume* vITSV = geoManager->GetVolume(UpgradeGeometryTGeo::GetITSVolPattern());
  vITSV->SetUniqueID(UpgradeGeometryTGeo::GetUIDShift()); // store modID -> midUUID bitshift
  vALIC->AddNode(vITSV, 2, 0); // Copy number is 2 to cheat AliGeoManager::CheckSymNamesLUT

  const Int_t kLength = 100;
  Char_t vstrng[kLength] = "xxxRS"; //?
  vITSV->SetTitle(vstrng);

  // Check that we have all needed parameters
  if (mNumberLayers <= 0) {
    LOG(FATAL) << "Wrong number of layers (" << mNumberLayers << ")" << FairLogger::endl;
  }

  for (Int_t j = 0; j < mNumberLayers; j++) {
    if (mLayerRadii[j] <= 0) {
      LOG(FATAL) << "Wrong layer radius for layer " << j << "(" << mLayerRadii[j] << ")"
                 << FairLogger::endl;
    }
    if (mLayerZLength[j] <= 0) {
      LOG(FATAL) << "Wrong layer length for layer " << j << "(" << mLayerZLength[j] << ")"
                 << FairLogger::endl;
    }
    if (mStavePerLayer[j] <= 0) {
      LOG(FATAL) << "Wrong number of staves for layer " << j << "(" << mStavePerLayer[j] << ")"
                 << FairLogger::endl;
    }
    if (mUnitPerStave[j] <= 0) {
      LOG(FATAL) << "Wrong number of chips for layer " << j << "(" << mUnitPerStave[j] << ")"
                 << FairLogger::endl;
    }
    if (mStaveThickness[j] < 0) {
      LOG(FATAL) << "Wrong stave thickness for layer " << j << "(" << mStaveThickness[j] << ")"
                 << FairLogger::endl;
    }
    if (mTurboLayer[j] && mStaveWidth[j] <= 0) {
      LOG(FATAL) << "Wrong stave width for layer " << j << "(" << mStaveWidth[j] << ")"
                 << FairLogger::endl;
    }
    if (mDetectorThickness[j] < 0) {
      LOG(FATAL) << "Wrong chip thickness for layer " << j << "(" << mDetectorThickness[j] << ")"
                 << FairLogger::endl;
    }

    if (j > 0) {
      if (mLayerRadii[j] <= mLayerRadii[j - 1]) {
        LOG(FATAL) << "Layer " << j << " radius (" << mLayerRadii[j] << ") is smaller than layer "
                   << j - 1 << " radius (" << mLayerRadii[j - 1] << ")" << FairLogger::endl;
      }
    }

    if (mStaveThickness[j] == 0) {
      LOG(INFO) << "Stave thickness for layer " << j << " not set, using default"
                << FairLogger::endl;
    }
    if (mDetectorThickness[j] == 0) {
      LOG(INFO) << "Chip thickness for layer " << j << " not set, using default"
                << FairLogger::endl;
    }
  }

  // Create the wrapper volumes
  TGeoVolume** wrapVols = 0;

  if (mNumberOfWrapperVolumes) {
    wrapVols = new TGeoVolume* [mNumberOfWrapperVolumes];
    for (int id = 0; id < mNumberOfWrapperVolumes; id++) {
      wrapVols[id] = CreateWrapperVolume(id);
      vITSV->AddNode(wrapVols[id], 1, 0);
    }
  }

  mWrapperLayerId = new Int_t[mNumberLayers];

  // Now create the actual geometry
  for (Int_t j = 0; j < mNumberLayers; j++) {
    TGeoVolume* dest = vITSV;
    mWrapperLayerId[j] = -1;

    if (mTurboLayer[j]) {
      mUpgradeGeometry[j] = new UpgradeV1Layer(j, kTRUE, kFALSE);
      mUpgradeGeometry[j]->SetStaveWidth(mStaveWidth[j]);
      mUpgradeGeometry[j]->SetStaveTilt(mStaveTilt[j]);
    } else {
      mUpgradeGeometry[j] = new UpgradeV1Layer(j, kFALSE);
    }

    mUpgradeGeometry[j]->SetPhi0(mLayerPhi0[j]);
    mUpgradeGeometry[j]->SetRadius(mLayerRadii[j]);
    mUpgradeGeometry[j]->SetZLength(mLayerZLength[j]);
    mUpgradeGeometry[j]->SetNumberOfStaves(mStavePerLayer[j]);
    mUpgradeGeometry[j]->SetNumberOfUnits(mUnitPerStave[j]);
    mUpgradeGeometry[j]->SetChipType(mChipTypeID[j]);
    mUpgradeGeometry[j]->SetBuildLevel(mBuildLevel[j]);

    if (j < 3) {
      mUpgradeGeometry[j]->SetStaveModel(mStaveModelInnerBarrel);
    } else {
      mUpgradeGeometry[j]->SetStaveModel(mStaveModelOuterBarrel);
    }

    LOG(DEBUG1) << "mBuildLevel: " << mBuildLevel[j] << FairLogger::endl;

    if (mStaveThickness[j] != 0) {
      mUpgradeGeometry[j]->SetStaveThick(mStaveThickness[j]);
    }
    if (mDetectorThickness[j] != 0) {
      mUpgradeGeometry[j]->SetSensorThick(mDetectorThickness[j]);
    }

    for (int iw = 0; iw < mNumberOfWrapperVolumes; iw++) {
      if (mLayerRadii[j] > mWrapperMinRadius[iw] && mLayerRadii[j] < mWrapperMaxRadius[iw]) {
        LOG(INFO) << "Will embed layer " << j << " in wrapper volume " << iw << FairLogger::endl;

        if (mLayerZLength[j] >= mWrapperZSpan[iw]) {
          LOG(FATAL) << "ZSpan " << mWrapperZSpan[iw] << " of wrapper volume " << iw
                     << " is less than ZSpan " << mLayerZLength[j] << " of layer " << j
                     << FairLogger::endl;
        }

        dest = wrapVols[iw];
        mWrapperLayerId[j] = iw;
        break;
      }
    }
    mUpgradeGeometry[j]->CreateLayer(dest);
  }
  CreateServiceBarrel(kTRUE, wrapVols[0]);
  CreateServiceBarrel(kFALSE, wrapVols[2]);

  delete[] wrapVols; // delete pointer only, not the volumes
}

// Service Barrel
void AliceO2::ITS::Detector::CreateServiceBarrel(const Bool_t innerBarrel, TGeoVolume* dest,
                                                 const TGeoManager* mgr)
{
  // Creates the Service Barrel (as a simple cylinder) for IB and OB
  // Inputs:
  //         innerBarrel : if true, build IB service barrel, otherwise for OB
  //         dest        : the mother volume holding the service barrel
  //         mgr         : the gGeoManager pointer (used to get the material)
  //

  Double_t rminIB = 4.7;
  Double_t rminOB = 43.4;
  Double_t zLenOB;
  Double_t cInt = 0.22; // dimensioni cilindro di supporto interno
  Double_t cExt = 1.00; // dimensioni cilindro di supporto esterno
  //  Double_t phi1   =  180;
  //  Double_t phi2   =  360;

  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  if (innerBarrel) {
    zLenOB = ((TGeoTube*)(dest->GetShape()))->GetDz();
    //    TGeoTube*ibSuppSh = new TGeoTubeSeg(rminIB,rminIB+cInt,zLenOB,phi1,phi2);
    TGeoTube* ibSuppSh = new TGeoTube(rminIB, rminIB + cInt, zLenOB);
    TGeoVolume* ibSupp = new TGeoVolume("ibSuppCyl", ibSuppSh, medCarbonFleece);
    dest->AddNode(ibSupp, 1);
  } else {
    zLenOB = ((TGeoTube*)(dest->GetShape()))->GetDz();
    TGeoTube* obSuppSh = new TGeoTube(rminOB, rminOB + cExt, zLenOB);
    TGeoVolume* obSupp = new TGeoVolume("obSuppCyl", obSuppSh, medCarbonFleece);
    dest->AddNode(obSupp, 1);
  }

  return;
}

void AliceO2::ITS::Detector::DefineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;

  // The names of the ITS sensitive volumes have the format: ITSUSensor(0...mNumberLayers-1)
  for (Int_t j = 0; j < mNumberLayers; j++) {
    volumeName = UpgradeGeometryTGeo::GetITSSensorPattern() + TString::Itoa(j, 10);
    v = geoManager->GetVolume(volumeName.Data());
    AddSensitiveVolume(v);
  }
}

Point* AliceO2::ITS::Detector::AddHit(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos,
                                      TVector3 mom, Double_t startTime, Double_t time,
                                      Double_t length, Double_t eLoss, Int_t shunt)
{
  TClonesArray& clref = *mPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new (clref[size])
    Point(trackID, detID, startPos, pos, mom, startTime, time, length, eLoss, shunt);
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

  return ((AliceO2::Data::Stack*)gMC->GetStack())->GetParticle(GetTrack());
}

void AliceO2::ITS::Detector::Print(ostream* os) const
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

  fmt = os->setf(ios::scientific); // set scientific floating point output
  *os << mTrackNumber << " " << mPositionX << " " << mPositionY << " " << mPositionZ << " ";
  fmt = os->setf(ios::hex); // set hex for mStatus only.
  *os << mStatus << " ";
  fmt = os->setf(ios::dec); // every thing else decimel.
  *os << mModule << " ";
  *os << mParticlePx << " " << mParticlePy << " " << mParticlePz << " ";
  *os << mEnergyDepositionStep << " " << mTof;
  *os << " " << mStartingStepX << " " << mStartingStepY << " " << mStartingStepZ;
  //    *os << " " << endl;
  os->flags(fmt); // reset back to old formating.
  return;
}

void AliceO2::ITS::Detector::Read(istream* is)
{
  // Standard input format for this class.
  // Inputs:
  //   istream *is  the input stream
  // Outputs:
  //   none.
  // Return:
  //   none.

  *is >> mTrackNumber >> mPositionX >> mPositionY >> mPositionZ;
  *is >> mStatus >> mModule >> mParticlePx >> mParticlePy >> mParticlePz >> mEnergyDepositionStep >>
    mTof;
  *is >> mStartingStepX >> mStartingStepY >> mStartingStepZ;
  return;
}

ostream& operator<<(ostream& os, AliceO2::ITS::Detector& p)
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

istream& operator>>(istream& is, AliceO2::ITS::Detector& r)
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
