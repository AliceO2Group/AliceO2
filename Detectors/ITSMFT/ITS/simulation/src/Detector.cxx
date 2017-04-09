/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "ITSMFTSimulation/Point.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"
#include "ITSSimulation/GeometryHandler.h"
#include "ITSSimulation/V1Layer.h"

#include "ITSBase/MisalignmentParameter.h"  // for MisalignmentParameter

#include "SimulationDataFormat/DetectorList.h"
#include "SimulationDataFormat/Stack.h"

//FairRoot includes
#include "FairDetector.h"           // for FairDetector
#include "FairLogger.h"             // for LOG, LOG_IF
#include "FairGenericRootManager.h"        // for FairGenericRootManager
#include "FairRun.h"                // for FairRun
#include "FairRuntimeDb.h"          // for FairRuntimeDb
#include "FairVolume.h"             // for FairVolume

#include "TClonesArray.h"           // for TClonesArray
#include "TGeoManager.h"            // for TGeoManager, gGeoManager
#include "TGeoTube.h"               // for TGeoTube
#include "TGeoVolume.h"             // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"                // for TString, operator+
#include "TVirtualMC.h"             // for gMC, TVirtualMC
#include "TVirtualMCStack.h"        // for TVirtualMCStack

#include <cstdio>                  // for NULL, snprintf

class FairModule;

class TGeoMedium;

class TParticle;

using std::cout;
using std::endl;

using o2::ITSMFT::Point;
using namespace o2::ITS;

Detector::Detector()
  : o2::Base::Detector("ITS", kTRUE, kAliIts),
    mLayerID(nullptr),
    mNumberLayers(),
    mTrackData(),
    /*
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(-1),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),
    mModifyGeometry(kFALSE),
    mNumberOfWrapperVolumes(0),
    mWrapperMinRadius(nullptr),
    mWrapperMaxRadius(nullptr),
    mWrapperZSpan(nullptr),
    mWrapperLayerId(nullptr),
    mTurboLayer(nullptr),
    mLayerPhi0(nullptr),
    mLayerRadii(nullptr),
    mLayerZLength(nullptr),
    mStavePerLayer(nullptr),
    mUnitPerStave(nullptr),
    mStaveThickness(nullptr),
    mStaveWidth(nullptr),
    mStaveTilt(nullptr),
    mDetectorThickness(nullptr),
    mChipTypeID(nullptr),
    mBuildLevel(nullptr),
    mPointCollection(new TClonesArray("o2::ITSMFT::Point")),
    
    mGeometryHandler(new GeometryHandler()),
    mMisalignmentParameter(nullptr),
    mGeometry(nullptr),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{
}

Detector::Detector(const char *name, Bool_t active, const Int_t nlay)
  : o2::Base::Detector(name, active, kAliIts),
    mLayerID(nullptr),
    mNumberLayers(nlay),
    mLayerName(new TString[nlay]),
    mTrackData(),
    /*
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(-1),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),
    mModifyGeometry(kFALSE),
    mNumberOfWrapperVolumes(0),
    mWrapperMinRadius(nullptr),
    mWrapperMaxRadius(nullptr),
    mWrapperZSpan(nullptr),
    mWrapperLayerId(nullptr),
    mTurboLayer(nullptr),
    mLayerPhi0(nullptr),
    mLayerRadii(nullptr),
    mLayerZLength(nullptr),
    mStavePerLayer(nullptr),
    mUnitPerStave(nullptr),
    mStaveThickness(nullptr),
    mStaveWidth(nullptr),
    mStaveTilt(nullptr),
    mDetectorThickness(nullptr),
    mChipTypeID(nullptr),
    mBuildLevel(nullptr),

    mPointCollection(new TClonesArray("o2::ITSMFT::Point")),
    mGeometryHandler(new GeometryHandler()),
    mMisalignmentParameter(nullptr),
    
    mGeometry(nullptr),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{

  for (Int_t j = 0; j < mNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V1Layer
  }

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

  mGeometry = new V1Layer *[mNumberLayers];

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
      mGeometry[j] = nullptr;
    }
  }
}

Detector::Detector(const Detector &rhs)
  : o2::Base::Detector(rhs),
    mLayerID(nullptr),
    mNumberLayers(rhs.mNumberLayers),
    mLayerName(nullptr),
    mTrackData(),
    /*    
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(rhs.mNumberOfDetectors),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),

    mModifyGeometry(rhs.mModifyGeometry),

    mNumberOfWrapperVolumes(rhs.mNumberOfWrapperVolumes),
  // the following parameters may be shared with master if needed
  // let's try not to set them and keep dtor simple
    mWrapperMinRadius(nullptr),
    mWrapperMaxRadius(nullptr),
    mWrapperZSpan(nullptr),
    mWrapperLayerId(nullptr),
    mTurboLayer(nullptr),
    mLayerPhi0(nullptr),
    mLayerRadii(nullptr),
    mLayerZLength(nullptr),
    mStavePerLayer(nullptr),
    mUnitPerStave(nullptr),
    mStaveThickness(nullptr),
    mStaveWidth(nullptr),
    mStaveTilt(nullptr),
    mDetectorThickness(nullptr),
    mChipTypeID(nullptr),
    mBuildLevel(nullptr),

  /// Container for data points
    mPointCollection(new TClonesArray("o2::ITSMFT::Point")),
    mGeometryHandler(rhs.mGeometryHandler), // CHECK
    mMisalignmentParameter(nullptr),

    mGeometry(rhs.mGeometry),
    mStaveModelInnerBarrel(rhs.mStaveModelInnerBarrel),
    mStaveModelOuterBarrel(rhs.mStaveModelInnerBarrel)
{
  mLayerName = new TString[mNumberLayers];

  for (Int_t j = 0; j < mNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V1Layer
  }
}

Detector::~Detector()
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
  delete[] mGeometry;
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

Detector &Detector::operator=(const Detector &rhs)
{
  // The standard = operator
  // Inputs:
  //   Detector   &h the sourse of this copy
  // Outputs:
  //   none.
  // Return:
  //  A copy of the sourse hit h

  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  Base::Detector::operator=(rhs);

  mLayerID = nullptr;
  mNumberLayers = rhs.mNumberLayers;
  mLayerName = nullptr;

  mNumberOfDetectors = rhs.mNumberOfDetectors;

  mModifyGeometry = rhs.mModifyGeometry;

  mNumberOfWrapperVolumes = rhs.mNumberOfWrapperVolumes;
  // the following parameters may be shared with master if needed
  // let's try not to set them and keep dtor simple
  mWrapperMinRadius = nullptr;
  mWrapperMaxRadius = nullptr;
  mWrapperZSpan = nullptr;
  mWrapperLayerId = nullptr;
  mTurboLayer = nullptr;
  mLayerPhi0 = nullptr;
  mLayerRadii = nullptr;
  mLayerZLength = nullptr;
  mStavePerLayer = nullptr;
  mUnitPerStave = nullptr;
  mStaveThickness = nullptr;
  mStaveWidth = nullptr;
  mStaveTilt = nullptr;
  mDetectorThickness = nullptr;
  mChipTypeID = nullptr;
  mBuildLevel = nullptr;

  /// Container for data points
  mPointCollection = nullptr;

  mGeometryHandler = nullptr;
  mMisalignmentParameter = nullptr;

  mGeometry = nullptr;
  mStaveModelInnerBarrel = rhs.mStaveModelInnerBarrel;
  mStaveModelOuterBarrel = rhs.mStaveModelInnerBarrel;

  mLayerName = new TString[mNumberLayers];
  for (Int_t j = 0; j < mNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V1Layer
  }

  return *this;
}

void Detector::Initialize()
{
  if (!mLayerID) {
    mLayerID = new Int_t[mNumberLayers];
  }

  for (int i = 0; i < mNumberLayers; i++) {
    mLayerID[i] = gMC ? TVirtualMC::GetMC()->VolId(mLayerName[i]) : 0;
  }

  mGeometryTGeo = new GeometryTGeo(kTRUE);

  FairDetector::Initialize();

  //  FairRuntimeDb* rtdb= FairRun::Instance()->GetRuntimeDb();
  //  O2itsGeoPar* par=(O2itsGeoPar*)(rtdb->getContainer("O2itsGeoPar"));
}

void Detector::initializeParameterContainers()
{
  LOG(INFO) << "Initialize aliitsdet misallign parameters" << FairLogger::endl;
  mNumberOfDetectors = mMisalignmentParameter->getNumberOfDetectors();
  mShiftX = mMisalignmentParameter->getShiftX();
  mShiftY = mMisalignmentParameter->getShiftY();
  mShiftZ = mMisalignmentParameter->getShiftZ();
  mRotX = mMisalignmentParameter->getRotX();
  mRotY = mMisalignmentParameter->getRotY();
  mRotZ = mMisalignmentParameter->getRotZ();
}

void Detector::setParameterContainers()
{
  LOG(INFO) << "Set tutdet misallign parameters" << FairLogger::endl;
  // Get Base Container
  FairRun *sim = FairRun::Instance();
  LOG_IF(FATAL, !sim) << "No run object" << FairLogger::endl;
  FairRuntimeDb *rtdb = sim->GetRuntimeDb();
  LOG_IF(FATAL, !rtdb) << "No runtime database" << FairLogger::endl;

  mMisalignmentParameter = (MisalignmentParameter *) (rtdb->getContainer("MisallignmentParameter"));
}

Bool_t Detector::ProcessHits(FairVolume *vol)
{
  // This method is called from the MC stepping
  if (!(TVirtualMC::GetMC()->TrackCharge())) {
    return kFALSE;
  }

  Int_t lay=0, volID = vol->getMCid();

  // FIXME: Determine the layer number. Is this information available directly from the FairVolume?
  bool notSens = false;
  while ((lay < mNumberLayers) && (notSens=(volID != mLayerID[lay]))) {
    ++lay;
  }
  if (notSens) return kFALSE; //RS: can this happen? This method must be called for sensors only?
  
  // FIXME: Is it needed to keep a track reference when the outer ITS volume is encountered?
  // if(TVirtualMC::GetMC()->IsTrackExiting()) {
  //  AddTrackReference(gAlice->GetMCApp()->GetCurrentTrackNumber(), AliTrackReference::kITS);
  // } // if Outer ITS mother Volume
  bool startHit=false, stopHit=false;
  unsigned char status = 0;
  if (TVirtualMC::GetMC()->IsTrackEntering()) { status |= Point::kTrackEntering; }
  if (TVirtualMC::GetMC()->IsTrackInside())   { status |= Point::kTrackInside; }
  if (TVirtualMC::GetMC()->IsTrackExiting())  { status |= Point::kTrackExiting; }
  if (TVirtualMC::GetMC()->IsTrackOut())      { status |= Point::kTrackOut; }
  if (TVirtualMC::GetMC()->IsTrackStop())     { status |= Point::kTrackStopped; }
  if (TVirtualMC::GetMC()->IsTrackAlive())    { status |= Point::kTrackAlive; }

  // track is entering or created in the volume
  if ( (status & Point::kTrackEntering) || (status & Point::kTrackInside && !mTrackData.mHitStarted) ) {
    startHit = true;
  }
  else if ( (status & (Point::kTrackExiting|Point::kTrackOut|Point::kTrackStopped)) ) {
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
    // Retrieve the indices with the volume path
    int stave(0), halfstave(0), chipinmodule(0), module;
    TVirtualMC::GetMC()->CurrentVolOffID(1, chipinmodule);
    TVirtualMC::GetMC()->CurrentVolOffID(2, module);
    TVirtualMC::GetMC()->CurrentVolOffID(3, halfstave);
    TVirtualMC::GetMC()->CurrentVolOffID(4, stave);
    int chipindex = mGeometryTGeo->getChipIndex(lay, stave, halfstave, module, chipinmodule);
    
    Point *p = addHit(TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber(), chipindex,
		      mTrackData.mPositionStart.Vect(),positionStop.Vect(),mTrackData.mMomentumStart.Vect(),
		      mTrackData.mMomentumStart.E(),positionStop.T(),mTrackData.mEnergyLoss,
		      mTrackData.mTrkStatusStart,status);
    //p->SetTotalEnergy(TVirtualMC::GetMC()->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    o2::Data::Stack *stack = (o2::Data::Stack *) TVirtualMC::GetMC()->GetStack();
    stack->AddPoint(kAliIts);
  }
  
  return kTRUE;
}

void Detector::createMaterials()
{
  // Int_t   ifield = ((AliceO2::Field::MagneticField*)TGeoGlobalMagField::Instance()->GetField())->Integral();
  // Float_t fieldm = ((AliceO2::Field::MagneticField*)TGeoGlobalMagField::Instance()->GetField())->Max();

  // until we solve the problem of reading the field from files with changed class names we
  //  need to hard code some values here to be able to run the macros  M.Al-Turany (Nov.14)
  Int_t ifield = 2;
  Float_t fieldm = 10.0;
  ////////////

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
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;

  // Water
  Float_t aWater[2] = {1.00794, 15.9994};
  Float_t zWater[2] = {1., 8.};
  Float_t wWater[2] = {0.111894, 0.888106};
  Float_t dWater = 1.0;

  // Kapton
  Float_t aKapton[4] = {1.00794, 12.0107, 14.010, 15.9994};
  Float_t zKapton[4] = {1., 6., 7., 8.};
  Float_t wKapton[4] = {0.026362, 0.69113, 0.07327, 0.209235};
  Float_t dKapton = 1.42;

  o2::Base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::Base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir,
                                  epsilAir, stminAir);

  o2::Base::Detector::Mixture(2, "WATER$", aWater, zWater, dWater, 2, wWater);
  o2::Base::Detector::Medium(2, "WATER$", 2, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  o2::Base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01,
                                    0.99900E+03);
  o2::Base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);

  o2::Base::Detector::Material(4, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7); // From AliPIPEv3
  o2::Base::Detector::Medium(4, "BERILLIUM$", 4, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);

  o2::Base::Detector::Material(5, "COPPER$", 0.63546E+02, 0.29000E+02, 0.89600E+01,
                                    0.14300E+01, 0.99900E+03);
  o2::Base::Detector::Medium(5, "COPPER$", 5, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  // needed for STAVE , Carbon, kapton, Epoxy, flexcable

  // AliceO2::Base::Detector::Material(6,"CARBON$",12.0107,6,2.210,999,999);
  o2::Base::Detector::Material(6, "CARBON$", 12.0107, 6, 2.210 / 1.3, 999, 999);
  o2::Base::Detector::Medium(6, "CARBON$", 6, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);

  o2::Base::Detector::Mixture(7, "KAPTON(POLYCH2)$", aKapton, zKapton, dKapton, 4, wKapton);
  o2::Base::Detector::Medium(7, "KAPTON(POLYCH2)$", 7, 0, ifield, fieldm, tmaxfd, stemax,
                                  deemax, epsil, stmin);

  // values below modified as compared to source AliITSv11 !

  // All types of carbon
  // Unidirectional prepreg
  o2::Base::Detector::Material(8, "K13D2U2k$", 12.0107, 6, 1.643, 999, 999);
  o2::Base::Detector::Medium(8, "K13D2U2k$", 8, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Impregnated thread
  o2::Base::Detector::Material(9, "M60J3K$", 12.0107, 6, 2.21, 999, 999);
  o2::Base::Detector::Medium(9, "M60J3K$", 9, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);
  // Impregnated thread
  o2::Base::Detector::Material(10, "M55J6K$", 12.0107, 6, 1.63, 999, 999);
  o2::Base::Detector::Medium(10, "M55J6K$", 10, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Fabric(0/90)
  o2::Base::Detector::Material(11, "T300$", 12.0107, 6, 1.725, 999, 999);
  o2::Base::Detector::Medium(11, "T300$", 11, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi,
                                  epsilSi, stminSi);
  // AMEC Thermasol
  o2::Base::Detector::Material(12, "FGS003$", 12.0107, 6, 1.6, 999, 999);
  o2::Base::Detector::Medium(12, "FGS003$", 12, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);
  // Carbon fleece
  o2::Base::Detector::Material(13, "CarbonFleece$", 12.0107, 6, 0.4, 999, 999);
  o2::Base::Detector::Medium(13, "CarbonFleece$", 13, 0, ifield, fieldm, tmaxfdSi, stemaxSi,
                                  deemaxSi, epsilSi, stminSi);

  // Flex cable
  Float_t aFCm[5] = {12.0107, 1.00794, 14.0067, 15.9994, 26.981538};
  Float_t zFCm[5] = {6., 1., 7., 8., 13.};
  Float_t wFCm[5] = {0.520088819984, 0.01983871336, 0.0551367996, 0.157399667056, 0.247536};
  // Float_t dFCm = 1.6087;  // original
  // Float_t dFCm = 2.55;   // conform with STAR
  Float_t dFCm = 2.595; // conform with Corrado

  o2::Base::Detector::Mixture(14, "FLEXCABLE$", aFCm, zFCm, dFCm, 5, wFCm);
  o2::Base::Detector::Medium(14, "FLEXCABLE$", 14, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);

  // AliceO2::Base::Detector::Material(7,"GLUE$",0.12011E+02,0.60000E+01,0.1930E+01/2.015,999,999);
  // // original
  o2::Base::Detector::Material(15, "GLUE$", 12.011, 6, 1.93 / 2.015, 999,
                                    999); // conform with ATLAS, Corrado, Stefan
  o2::Base::Detector::Medium(15, "GLUE$", 15, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                  stmin);

  o2::Base::Detector::Material(16, "ALUMINUM$", 0.26982E+02, 0.13000E+02, 0.26989E+01,
                                    0.89000E+01, 0.99900E+03);
  o2::Base::Detector::Medium(16, "ALUMINUM$", 16, 0, ifield, fieldm, tmaxfd, stemax, deemax,
                                  epsil, stmin);
}

void Detector::EndOfEvent()
{
  if (mPointCollection) { mPointCollection->Clear(); }
}

void Detector::Register()
{
  // This will create a branch in the output tree called Point, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairGenericRootManager::Instance()) {
    FairGenericRootManager::Instance()->Register("ITSPoint", "ITS", mPointCollection, kTRUE);
  }
}

TClonesArray *Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0) {
    return mPointCollection;
  } else {
    return nullptr;
  }
}

void Detector::Reset()
{
  mPointCollection->Clear();
}

void Detector::setNumberOfWrapperVolumes(Int_t n)
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

void Detector::defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax,
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

void Detector::defineLayer(Int_t nlay, double phi0, Double_t r, Double_t zlen,
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

void Detector::defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen,
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

void Detector::getLayerParameters(Int_t nlay, Double_t &phi0, Double_t &r,
                                  Double_t &zlen, Int_t &nstav, Int_t &nmod,
                                  Double_t &width, Double_t &tilt, Double_t &lthick,
                                  Double_t &dthick, UInt_t &dettype) const
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

TGeoVolume *Detector::createWrapperVolume(Int_t id)
{
  // Creates an air-filled wrapper cylindrical volume

  if (mWrapperMinRadius[id] < 0 || mWrapperMaxRadius[id] < 0 || mWrapperZSpan[id] < 0) {
    LOG(FATAL) << "Wrapper volume " << id << " was requested but not defined" << FairLogger::endl;
  }

  // Now create the actual shape and volume
  auto *tube =
    new TGeoTube(mWrapperMinRadius[id], mWrapperMaxRadius[id], mWrapperZSpan[id] / 2.);

  TGeoMedium *medAir = gGeoManager->GetMedium("ITS_AIR$");

  char volnam[30];
  snprintf(volnam, 29, "%s%d", GeometryTGeo::getITSWrapVolPattern(), id);

  auto *wrapper = new TGeoVolume(volnam, tube, medAir);

  return wrapper;
}

void Detector::ConstructGeometry()
{
  // Create the detector materials
  createMaterials();

  // Construct the detector geometry
  constructDetectorGeometry();

  // Define the list of sensitive volumes
  defineSensitiveVolumes();
}

void Detector::constructDetectorGeometry()
{
  // Create the geometry and insert it in the mother volume ITSV
  TGeoManager *geoManager = gGeoManager;

  TGeoVolume *vALIC = geoManager->GetVolume("cave");

  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume" << FairLogger::endl;
  }

  new TGeoVolumeAssembly(GeometryTGeo::getITSVolPattern());
  TGeoVolume *vITSV = geoManager->GetVolume(GeometryTGeo::getITSVolPattern());
  vITSV->SetUniqueID(GeometryTGeo::getUIDShift()); // store modID -> midUUID bitshift
  vALIC->AddNode(vITSV, 2, nullptr); // Copy number is 2 to cheat AliGeoManager::CheckSymNamesLUT

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
  TGeoVolume **wrapVols = nullptr;

  if (mNumberOfWrapperVolumes) {
    wrapVols = new TGeoVolume *[mNumberOfWrapperVolumes];
    for (int id = 0; id < mNumberOfWrapperVolumes; id++) {
      wrapVols[id] = createWrapperVolume(id);
      vITSV->AddNode(wrapVols[id], 1, nullptr);
    }
  }

  mWrapperLayerId = new Int_t[mNumberLayers];

  // Now create the actual geometry
  for (Int_t j = 0; j < mNumberLayers; j++) {
    TGeoVolume *dest = vITSV;
    mWrapperLayerId[j] = -1;

    if (mTurboLayer[j]) {
      mGeometry[j] = new V1Layer(j, kTRUE, kFALSE);
      mGeometry[j]->setStaveWidth(mStaveWidth[j]);
      mGeometry[j]->setStaveTilt(mStaveTilt[j]);
    } else {
      mGeometry[j] = new V1Layer(j, kFALSE);
    }

    mGeometry[j]->setPhi0(mLayerPhi0[j]);
    mGeometry[j]->setRadius(mLayerRadii[j]);
    mGeometry[j]->setZLength(mLayerZLength[j]);
    mGeometry[j]->setNumberOfStaves(mStavePerLayer[j]);
    mGeometry[j]->setNumberOfUnits(mUnitPerStave[j]);
    mGeometry[j]->setChipType(mChipTypeID[j]);
    mGeometry[j]->setBuildLevel(mBuildLevel[j]);

    if (j < 3) {
      mGeometry[j]->setStaveModel(mStaveModelInnerBarrel);
    } else {
      mGeometry[j]->setStaveModel(mStaveModelOuterBarrel);
    }

    LOG(DEBUG1) << "mBuildLevel: " << mBuildLevel[j] << FairLogger::endl;

    if (mStaveThickness[j] != 0) {
      mGeometry[j]->setStaveThick(mStaveThickness[j]);
    }
    if (mDetectorThickness[j] != 0) {
      mGeometry[j]->setSensorThick(mDetectorThickness[j]);
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
    mGeometry[j]->createLayer(dest);
  }
  createServiceBarrel(kTRUE, wrapVols[0]);
  createServiceBarrel(kFALSE, wrapVols[2]);

  delete[] wrapVols; // delete pointer only, not the volumes
}

// Service Barrel
void Detector::createServiceBarrel(const Bool_t innerBarrel, TGeoVolume *dest,
                                   const TGeoManager *mgr)
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

  TGeoMedium *medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  if (innerBarrel) {
    zLenOB = ((TGeoTube *) (dest->GetShape()))->GetDz();
    //    TGeoTube*ibSuppSh = new TGeoTubeSeg(rminIB,rminIB+cInt,zLenOB,phi1,phi2);
    auto *ibSuppSh = new TGeoTube(rminIB, rminIB + cInt, zLenOB);
    auto *ibSupp = new TGeoVolume("ibSuppCyl", ibSuppSh, medCarbonFleece);
    dest->AddNode(ibSupp, 1);
  } else {
    zLenOB = ((TGeoTube *) (dest->GetShape()))->GetDz();
    auto *obSuppSh = new TGeoTube(rminOB, rminOB + cExt, zLenOB);
    auto *obSupp = new TGeoVolume("obSuppCyl", obSuppSh, medCarbonFleece);
    dest->AddNode(obSupp, 1);
  }

  return;
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager *geoManager = gGeoManager;
  TGeoVolume *v;

  TString volumeName;

  // The names of the ITS sensitive volumes have the format: ITSUSensor(0...mNumberLayers-1)
  for (Int_t j = 0; j < mNumberLayers; j++) {
    volumeName = GeometryTGeo::getITSSensorPattern() + TString::Itoa(j, 10);
    v = geoManager->GetVolume(volumeName.Data());
    AddSensitiveVolume(v);
  }
}

Point *Detector::addHit(int trackID, int detID, TVector3 startPos, TVector3 endPos, TVector3 startMom, double startE,
			double endTime, double eLoss, unsigned char startStatus, unsigned char endStatus)
{
  TClonesArray &clref = *mPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new(clref[size]) Point(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
}

TParticle *Detector::GetParticle() const
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
  int trc = TVirtualMC::GetMC()->GetStack()->GetCurrentTrackNumber();
  return ((o2::Data::Stack *) TVirtualMC::GetMC()->GetStack())->GetParticle(trc);
}

void Detector::Print(std::ostream *os) const
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
  std::ios::fmtflags fmt;
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
  // RS: why do we need to pring this garbage?
  
  //fmt = os->setf(std::ios::scientific); // set scientific floating point output
  //fmt = os->setf(std::ios::hex); // set hex for mStatus only.
  //fmt = os->setf(std::ios::dec); // every thing else decimel.
  //  *os << mModule << " ";
  //  *os << mEnergyDepositionStep << " " << mTof;
  //  *os << " " << mStartingStepX << " " << mStartingStepY << " " << mStartingStepZ;
  //    *os << " " << endl;
  //os->flags(fmt); // reset back to old formating.
  return;
}

void Detector::Read(std::istream *is)
{
  // Standard input format for this class.
  // Inputs:
  //   istream *is  the input stream
  // Outputs:
  //   none.
  // Return:
  //   none.
  // RS no need to read garbage
  return;
}

FairModule *Detector::CloneModule() const
{
  return new Detector(*this);
}

std::ostream &operator<<(std::ostream &os, Detector &p)
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

std::istream &operator>>(std::istream &is, Detector &r)
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

ClassImp(o2::ITS::Detector)
