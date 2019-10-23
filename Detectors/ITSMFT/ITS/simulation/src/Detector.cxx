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

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSSimulation/Detector.h"
#include "ITSSimulation/V3Layer.h"
#include "ITSSimulation/V3Services.h"

#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"

// FairRoot includes
#include "FairDetector.h"    // for FairDetector
#include "FairLogger.h"      // for LOG, LOG_IF
#include "FairRootManager.h" // for FairRootManager
#include "FairRun.h"         // for FairRun
#include "FairRuntimeDb.h"   // for FairRuntimeDb
#include "FairVolume.h"      // for FairVolume
#include "FairRootManager.h"

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack

#include <cstdio> // for NULL, snprintf

class FairModule;

class TGeoMedium;

class TParticle;

using std::cout;
using std::endl;

using o2::itsmft::Hit;
using Segmentation = o2::itsmft::SegmentationAlpide;
using namespace o2::its;

Detector::Detector()
  : o2::base::DetImpl<Detector>("ITS", kTRUE),
    mTrackData(),
    /*
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(-1),
    mModifyGeometry(kFALSE),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>()),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{
}

static double radii2Turbo(double rMin, double rMid, double rMax, double sensW)
{
  // compute turbo angle from radii and sensor width
  return TMath::ASin((rMax * rMax - rMin * rMin) / (2 * rMid * sensW)) * TMath::RadToDeg();
}

static void configITS(Detector* its)
{
  // build ITS upgrade detector
  const int kNLr = 7;
  const int kNLrInner = 3;
  const int kBuildLevel = 0;
  const int kSensTypeID = 0; // dummy id for Alpide sensor

  const float ChipThicknessIB = 50.e-4;
  const float ChipThicknessOB = 100.e-4;

  enum { kRmn,
         kRmd,
         kRmx,
         kNModPerStave,
         kPhi0,
         kNStave,
         kNPar };
  // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1, rMid is mean value)
  const double tdr5dat[kNLr][kNPar] = {
    {2.24, 2.34, 2.67, 9., 16.42, 12}, // for each inner layer: rMin,rMid,rMax,NChip/Stave, phi0, nStaves
    {3.01, 3.15, 3.46, 9., 12.18, 16},
    {3.78, 3.93, 4.21, 9., 9.55, 20},
    {-1, 19.6, -1, 4., 0., 24},  // for others: -, rMid, -, NMod/HStave, phi0, nStaves // 24 was 49
    {-1, 24.55, -1, 4., 0., 30}, // 30 was 61
    {-1, 34.39, -1, 7., 0., 42}, // 42 was 88
    {-1, 39.34, -1, 7., 0., 48}  // 48 was 100
  };
  const int nChipsPerModule = 7;  // For OB: how many chips in a row
  const double zChipGap = 0.01;   // For OB: gap in Z between chips
  const double zModuleGap = 0.01; // For OB: gap in Z between modules

  double dzLr, rLr, phi0, turbo;
  int nStaveLr, nModPerStaveLr;

  its->setStaveModelIB(o2::its::Detector::kIBModel4);
  its->setStaveModelOB(o2::its::Detector::kOBModel2);

  const int kNWrapVol = 3;
  const double wrpRMin[kNWrapVol] = {2.1, 19.3, 32.0};
  const double wrpRMax[kNWrapVol] = {14.0, 30.0, 46.0};
  const double wrpZSpan[kNWrapVol] = {70., 93., 165.8};

  for (int iw = 0; iw < kNWrapVol; iw++) {
    its->defineWrapperVolume(iw, wrpRMin[iw], wrpRMax[iw], wrpZSpan[iw]);
  }

  for (int idLr = 0; idLr < kNLr; idLr++) {
    rLr = tdr5dat[idLr][kRmd];
    phi0 = tdr5dat[idLr][kPhi0];

    nStaveLr = TMath::Nint(tdr5dat[idLr][kNStave]);
    nModPerStaveLr = TMath::Nint(tdr5dat[idLr][kNModPerStave]);
    int nChipsPerStaveLr = nModPerStaveLr;
    if (idLr >= kNLrInner) {
      its->defineLayer(idLr, phi0, rLr, nStaveLr, nModPerStaveLr, ChipThicknessOB, Segmentation::SensorLayerThickness,
                       kSensTypeID, kBuildLevel);
    } else {
      turbo = radii2Turbo(tdr5dat[idLr][kRmn], rLr, tdr5dat[idLr][kRmx], Segmentation::SensorSizeRows);
      its->defineLayerTurbo(idLr, phi0, rLr, nStaveLr, nChipsPerStaveLr, Segmentation::SensorSizeRows, turbo,
                            ChipThicknessIB, Segmentation::SensorLayerThickness, kSensTypeID, kBuildLevel);
    }
  }
}

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("ITS", active),
    mTrackData(),
    /*
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(-1),
    mModifyGeometry(kFALSE),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>()),
    mStaveModelInnerBarrel(kIBModel0),
    mStaveModelOuterBarrel(kOBModel0)
{

  for (Int_t j = 0; j < sNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }

  if (sNumberLayers > 0) { // if not, we'll Fatal-ize in CreateGeometry
    for (Int_t j = 0; j < sNumberLayers; j++) {
      mLayerPhi0[j] = 0;
      mLayerRadii[j] = 0.;
      mStavePerLayer[j] = 0;
      mUnitPerStave[j] = 0;
      mChipThickness[j] = 0.;
      mStaveWidth[j] = 0.;
      mStaveTilt[j] = 0.;
      mDetectorThickness[j] = 0.;
      mChipTypeID[j] = 0;
      mBuildLevel[j] = 0;
      mGeometry[j] = nullptr;
    }
  }
  mServicesGeometry = nullptr;

  for (int i = sNumberOfWrapperVolumes; i--;) {
    mWrapperMinRadius[i] = mWrapperMaxRadius[i] = mWrapperZSpan[i] = -1;
  }

  configITS(this);
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),
    /*
    mHitStarted(false),
    mTrkStatusStart(),
    mPositionStart(),
    mMomentumStart(),
    mEnergyLoss(),
    */
    mNumberOfDetectors(rhs.mNumberOfDetectors),
    mModifyGeometry(rhs.mModifyGeometry),

    /// Container for data points
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>()),
    mStaveModelInnerBarrel(rhs.mStaveModelInnerBarrel),
    mStaveModelOuterBarrel(rhs.mStaveModelOuterBarrel)
{

  for (Int_t j = 0; j < sNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }
}

Detector::~Detector()
{

  if (mHits) {
    // delete mHits;
    o2::utils::freeSimVector(mHits);
  }
}

Detector& Detector::operator=(const Detector& rhs)
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
  base::Detector::operator=(rhs);

  mNumberOfDetectors = rhs.mNumberOfDetectors;

  mModifyGeometry = rhs.mModifyGeometry;

  /// Container for data points
  mHits = nullptr;

  mStaveModelInnerBarrel = rhs.mStaveModelInnerBarrel;
  mStaveModelOuterBarrel = rhs.mStaveModelOuterBarrel;

  for (Int_t j = 0; j < sNumberLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }

  return *this;
}

void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  defineSensitiveVolumes();

  for (int i = 0; i < sNumberLayers; i++) {
    mLayerID[i] = gMC ? TVirtualMC::GetMC()->VolId(mLayerName[i]) : 0;
  }

  mGeometryTGeo = GeometryTGeo::Instance();
  //  FairRuntimeDb* rtdb= FairRun::Instance()->GetRuntimeDb();
  //  O2itsGeoPar* par=(O2itsGeoPar*)(rtdb->getContainer("O2itsGeoPar"));
}

Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  Int_t lay = 0, volID = vol->getMCid();

  // FIXME: Determine the layer number. Is this information available directly from the FairVolume?
  bool notSens = false;
  while ((lay < sNumberLayers) && (notSens = (volID != mLayerID[lay]))) {
    ++lay;
  }
  if (notSens)
    return kFALSE; // RS: can this happen? This method must be called for sensors only?

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();
  if (fMC->IsTrackExiting() && (lay == 0 || lay == 6)) {
    // Keep the track refs for the innermost and outermost layers only
    o2::TrackReference tr(*fMC, GetDetId());
    tr.setTrackID(stack->GetCurrentTrackNumber());
    tr.setUserId(lay);
    stack->addTrackReference(tr);
  }
  bool startHit = false, stopHit = false;
  unsigned char status = 0;
  if (fMC->IsTrackEntering()) {
    status |= Hit::kTrackEntering;
  }
  if (fMC->IsTrackInside()) {
    status |= Hit::kTrackInside;
  }
  if (fMC->IsTrackExiting()) {
    status |= Hit::kTrackExiting;
  }
  if (fMC->IsTrackOut()) {
    status |= Hit::kTrackOut;
  }
  if (fMC->IsTrackStop()) {
    status |= Hit::kTrackStopped;
  }
  if (fMC->IsTrackAlive()) {
    status |= Hit::kTrackAlive;
  }

  // track is entering or created in the volume
  if ((status & Hit::kTrackEntering) || (status & Hit::kTrackInside && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((status & (Hit::kTrackExiting | Hit::kTrackOut | Hit::kTrackStopped))) {
    stopHit = true;
  }

  // increment energy loss at all steps except entrance
  if (!startHit)
    mTrackData.mEnergyLoss += fMC->Edep();
  if (!(startHit | stopHit))
    return kFALSE; // do noting

  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = status;
    mTrackData.mHitStarted = true;
  }
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    // Retrieve the indices with the volume path
    int stave(0), halfstave(0), chipinmodule(0), module;
    fMC->CurrentVolOffID(1, chipinmodule);
    fMC->CurrentVolOffID(2, module);
    fMC->CurrentVolOffID(3, halfstave);
    fMC->CurrentVolOffID(4, stave);
    int chipindex = mGeometryTGeo->getChipIndex(lay, stave, halfstave, module, chipinmodule);

    Hit* p = addHit(stack->GetCurrentTrackNumber(), chipindex, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                    mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    // p->SetTotalEnergy(vmc->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    stack->addHit(GetDetId());
  }

  return kTRUE;
}

void Detector::createMaterials()
{
  Int_t ifield = 2;
  Float_t fieldm = 10.0;
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);
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

  // PEEK CF30
  Float_t aPEEK[3] = {12.0107, 1.00794, 15.9994};
  Float_t zPEEK[3] = {6., 1., 8.};
  Float_t wPEEK[3] = {19., 12., 3};
  Float_t dPEEK = 1.32;

  // Kapton
  Float_t aKapton[4] = {1.00794, 12.0107, 14.010, 15.9994};
  Float_t zKapton[4] = {1., 6., 7., 8.};
  Float_t wKapton[4] = {0.026362, 0.69113, 0.07327, 0.209235};
  Float_t dKapton = 1.42;

  // Tungsten Carbide
  Float_t aWC[2] = {183.84, 12.0107};
  Float_t zWC[2] = {74, 6};
  Float_t wWC[2] = {0.5, 0.5};
  Float_t dWC = 15.63;

  // BEOL (Metal interconnection stack in Si sensors)
  Float_t aBEOL[3] = {26.982, 28.086, 15.999};
  Float_t zBEOL[3] = {13, 14, 8}; // Al, Si, O
  Float_t wBEOL[3] = {0.170, 0.388, 0.442};
  Float_t dBEOL = 2.28;

  // Inox 304
  Float_t aInox304[4] = {12.0107, 51.9961, 58.6928, 55.845};
  Float_t zInox304[4] = {6., 24., 28, 26};       // C, Cr, Ni, Fe
  Float_t wInox304[4] = {0.0003, 0.18, 0.10, 0}; // [3] will be computed
  Float_t dInox304 = 7.85;

  // Ceramic (for IB capacitors) (BaTiO3)
  Float_t aCeramic[3] = {137.327, 47.867, 15.999};
  Float_t zCeramic[3] = {56, 22, 8}; // Ba, Ti, O
  Float_t wCeramic[3] = {1, 1, 3};   // Molecular composition
  Float_t dCeramic = 6.02;

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  o2::base::Detector::Mixture(2, "WATER$", aWater, zWater, dWater, 2, wWater);
  o2::base::Detector::Medium(2, "WATER$", 2, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  o2::base::Detector::Material(4, "BERILLIUM$", 9.01, 4., 1.848, 35.3, 36.7); // From AliPIPEv3
  o2::base::Detector::Medium(4, "BERILLIUM$", 4, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Material(5, "COPPER$", 0.63546E+02, 0.29000E+02, 0.89600E+01, 0.14300E+01, 0.99900E+03);
  o2::base::Detector::Medium(5, "COPPER$", 5, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // needed for STAVE , Carbon, kapton, Epoxy, flexcable

  // AliceO2::Base::Detector::Material(6,"CARBON$",12.0107,6,2.210,999,999);
  o2::base::Detector::Material(6, "CARBON$", 12.0107, 6, 2.210 / 1.3, 999, 999);
  o2::base::Detector::Medium(6, "CARBON$", 6, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  o2::base::Detector::Mixture(7, "KAPTON(POLYCH2)$", aKapton, zKapton, dKapton, 4, wKapton);
  o2::base::Detector::Medium(7, "KAPTON(POLYCH2)$", 7, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // values below modified as compared to source AliITSv11 !

  // BEOL (Metal interconnection stack in Si sensors)
  o2::base::Detector::Mixture(29, "METALSTACK$", aBEOL, zBEOL, dBEOL, 3, wBEOL);
  o2::base::Detector::Medium(29, "METALSTACK$", 29, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // Glue between IB chip and FPC: density reduced to take into account
  // empty spaces (160 glue spots/chip , diam. 1 spot = 1 mm)
  o2::base::Detector::Material(30, "GLUE_IBFPC$", 12.011, 6, 1.05 * 0.3, 999, 999);
  o2::base::Detector::Medium(30, "GLUE_IBFPC$", 30, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // Ceramic for IB capacitors (nmat < 0 => wmat contains number of atoms)
  o2::base::Detector::Mixture(31, "CERAMIC$", aCeramic, zCeramic, dCeramic, -3, wCeramic);
  o2::base::Detector::Medium(31, "CERAMIC$", 31, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // All types of carbon
  // Unidirectional prepreg
  o2::base::Detector::Material(8, "K13D2U2k$", 12.0107, 6, 1.643, 999, 999);
  o2::base::Detector::Medium(8, "K13D2U2k$", 8, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  o2::base::Detector::Material(17, "K13D2U120$", 12.0107, 6, 1.583, 999, 999);
  o2::base::Detector::Medium(17, "K13D2U120$", 17, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // Carbon prepreg woven
  o2::base::Detector::Material(18, "F6151B05M$", 12.0107, 6, 2.133, 999, 999);
  o2::base::Detector::Medium(18, "F6151B05M$", 18, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // Impregnated thread
  o2::base::Detector::Material(9, "M60J3K$", 12.0107, 6, 2.21, 999, 999);
  o2::base::Detector::Medium(9, "M60J3K$", 9, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // Impregnated thread
  o2::base::Detector::Material(10, "M55J6K$", 12.0107, 6, 1.63, 999, 999);
  o2::base::Detector::Medium(10, "M55J6K$", 10, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // Fabric(0/90)
  o2::base::Detector::Material(11, "T300$", 12.0107, 6, 1.725, 999, 999);
  o2::base::Detector::Medium(11, "T300$", 11, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // AMEC Thermasol
  o2::base::Detector::Material(12, "FGS003$", 12.0107, 6, 1.6, 999, 999);
  o2::base::Detector::Medium(12, "FGS003$", 12, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
  // Carbon fleece
  o2::base::Detector::Material(13, "CarbonFleece$", 12.0107, 6, 0.4, 999, 999);
  o2::base::Detector::Medium(13, "CarbonFleece$", 13, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi,
                             stminSi);

  // PEEK CF30
  o2::base::Detector::Mixture(19, "PEEKCF30$", aPEEK, zPEEK, dPEEK, -3, wPEEK);
  o2::base::Detector::Medium(19, "PEEKCF30$", 19, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  // Flex cable
  Float_t aFCm[5] = {12.0107, 1.00794, 14.0067, 15.9994, 26.981538};
  Float_t zFCm[5] = {6., 1., 7., 8., 13.};
  Float_t wFCm[5] = {0.520088819984, 0.01983871336, 0.0551367996, 0.157399667056, 0.247536};
  // Float_t dFCm = 1.6087;  // original
  // Float_t dFCm = 2.55;   // conform with STAR
  Float_t dFCm = 2.595; // conform with Corrado

  o2::base::Detector::Mixture(14, "FLEXCABLE$", aFCm, zFCm, dFCm, 5, wFCm);
  o2::base::Detector::Medium(14, "FLEXCABLE$", 14, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  // AliceO2::Base::Detector::Material(7,"GLUE$",0.12011E+02,0.60000E+01,0.1930E+01/2.015,999,999);
  // // original
  o2::base::Detector::Material(15, "GLUE$", 12.011, 6, 1.93 / 2.015, 999, 999); // conform with ATLAS, Corrado, Stefan
  o2::base::Detector::Medium(15, "GLUE$", 15, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Material(16, "ALUMINUM$", 0.26982E+02, 0.13000E+02, 0.26989E+01, 0.89000E+01, 0.99900E+03);
  o2::base::Detector::Medium(16, "ALUMINUM$", 16, 0, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(20, "TUNGCARB$", aWC, zWC, dWC, 2, wWC);
  o2::base::Detector::Medium(20, "TUNGCARB$", 20, 0, ifield, fieldm, tmaxfd, stemax, deemaxSi, epsilSi, stminSi);

  wInox304[3] = 1. - wInox304[0] - wInox304[1] - wInox304[2];
  o2::base::Detector::Mixture(21, "INOX304$", aInox304, zInox304, dInox304, 4, wInox304);
  o2::base::Detector::Medium(21, "INOX304$", 21, 0, ifield, fieldm, tmaxfd, stemax, deemaxSi, epsilSi, stminSi);

  // Tungsten (for gamma converter rods)
  o2::base::Detector::Material(28, "TUNGSTEN$", 183.84, 74, 19.25, 999, 999);
  o2::base::Detector::Medium(28, "TUNGSTEN$", 28, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
}

void Detector::EndOfEvent() { Reset(); }

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void Detector::defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan)
{
  // set parameters of id-th wrapper volume
  if (id >= sNumberOfWrapperVolumes || id < 0) {
    LOG(FATAL) << "id " << id << " of wrapper volume is not in 0-" << sNumberOfWrapperVolumes - 1 << " range";
  }

  mWrapperMinRadius[id] = rmin;
  mWrapperMaxRadius[id] = rmax;
  mWrapperZSpan[id] = zspan;
}

void Detector::defineLayer(Int_t nlay, double phi0, Double_t r, Int_t nstav, Int_t nunit, Double_t lthick,
                           Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
  //     Sets the layer parameters
  // Inputs:
  //          nlay    layer number
  //          phi0    layer phi0
  //          r       layer radius
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

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " Nst:" << nstav << " Nunit:" << nunit
            << " Lthick:" << lthick << " Dthick:" << dthick << " DetID:" << dettypeID << " B:" << buildLevel;

  if (nlay >= sNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay;
    return;
  }

  mTurboLayer[nlay] = kFALSE;
  mLayerPhi0[nlay] = phi0;
  mLayerRadii[nlay] = r;
  mStavePerLayer[nlay] = nstav;
  mUnitPerStave[nlay] = nunit;
  mChipThickness[nlay] = lthick;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
}

void Detector::defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Int_t nstav, Int_t nunit, Double_t width,
                                Double_t tilt, Double_t lthick, Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
  //     Sets the layer parameters for a "turbo" layer
  //     (i.e. a layer whose staves overlap in phi)
  // Inputs:
  //          nlay    layer number
  //          phi0    phi of 1st stave
  //          r       layer radius
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

  LOG(INFO) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " Nst:" << nstav << " Nunit:" << nunit
            << " W:" << width << " Tilt:" << tilt << " Lthick:" << lthick << " Dthick:" << dthick
            << " DetID:" << dettypeID << " B:" << buildLevel;

  if (nlay >= sNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay;
    return;
  }

  mTurboLayer[nlay] = kTRUE;
  mLayerPhi0[nlay] = phi0;
  mLayerRadii[nlay] = r;
  mStavePerLayer[nlay] = nstav;
  mUnitPerStave[nlay] = nunit;
  mChipThickness[nlay] = lthick;
  mStaveWidth[nlay] = width;
  mStaveTilt[nlay] = tilt;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
}

void Detector::getLayerParameters(Int_t nlay, Double_t& phi0, Double_t& r, Int_t& nstav, Int_t& nmod, Double_t& width,
                                  Double_t& tilt, Double_t& lthick, Double_t& dthick, UInt_t& dettype) const
{
  //     Gets the layer parameters
  // Inputs:
  //          nlay    layer number
  // Outputs:
  //          phi0    phi of 1st stave
  //          r       layer radius
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

  if (nlay >= sNumberLayers || nlay < 0) {
    LOG(ERROR) << "Wrong layer number " << nlay;
    return;
  }

  phi0 = mLayerPhi0[nlay];
  r = mLayerRadii[nlay];
  nstav = mStavePerLayer[nlay];
  nmod = mUnitPerStave[nlay];
  width = mStaveWidth[nlay];
  tilt = mStaveTilt[nlay];
  lthick = mChipThickness[nlay];
  dthick = mDetectorThickness[nlay];
  dettype = mChipTypeID[nlay];
}

TGeoVolume* Detector::createWrapperVolume(Int_t id)
{
  // Creates an air-filled wrapper cylindrical volume
  // For OB a Pcon is needed to host the support rings
  // while avoiding overlaps with MFT structures

  const Double_t suppRingAZlen = 4.;
  const Double_t suppRingCZlen[2] = {4.8, 4.0};
  const Double_t suppRingsRmin[2] = {23.35, 20.05};

  if (mWrapperMinRadius[id] < 0 || mWrapperMaxRadius[id] < 0 || mWrapperZSpan[id] < 0) {
    LOG(FATAL) << "Wrapper volume " << id << " was requested but not defined";
  }

  // Now create the actual shape and volume
  TGeoShape* tube;
  if (id == 1) {
    TGeoPcon* wrap = new TGeoPcon(0, 360, 6);
    Double_t zlen = mWrapperZSpan[id] / 2 + suppRingCZlen[0];
    wrap->DefineSection(0, -zlen, suppRingsRmin[0], mWrapperMaxRadius[id]);
    zlen = mWrapperZSpan[id] / 2 + suppRingCZlen[1];
    wrap->DefineSection(1, -zlen, suppRingsRmin[0], mWrapperMaxRadius[id]);
    wrap->DefineSection(2, -zlen, suppRingsRmin[1], mWrapperMaxRadius[id]);
    wrap->DefineSection(3, -mWrapperZSpan[id] / 2., suppRingsRmin[1], mWrapperMaxRadius[id]);
    wrap->DefineSection(4, -mWrapperZSpan[id] / 2., mWrapperMinRadius[id], mWrapperMaxRadius[id]);
    zlen = mWrapperZSpan[id] / 2 + suppRingAZlen;
    wrap->DefineSection(5, zlen, mWrapperMinRadius[id], mWrapperMaxRadius[id]);
    tube = (TGeoShape*)wrap;
  } else {
    TGeoTube* wrap = new TGeoTube(mWrapperMinRadius[id], mWrapperMaxRadius[id], mWrapperZSpan[id] / 2.);
    tube = (TGeoShape*)wrap;
  }

  TGeoMedium* medAir = gGeoManager->GetMedium("ITS_AIR$");

  char volnam[30];
  snprintf(volnam, 29, "%s%d", GeometryTGeo::getITSWrapVolPattern(), id);

  auto* wrapper = new TGeoVolume(volnam, tube, medAir);

  return wrapper;
}

void Detector::ConstructGeometry()
{
  // Create the detector materials
  createMaterials();

  // Construct the detector geometry
  constructDetectorGeometry();
}

void Detector::constructDetectorGeometry()
{
  // Create the geometry and insert it in the mother volume ITSV
  TGeoManager* geoManager = gGeoManager;

  TGeoVolume* vALIC = geoManager->GetVolume("cave");

  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume";
  }

  new TGeoVolumeAssembly(GeometryTGeo::getITSVolPattern());
  TGeoVolume* vITSV = geoManager->GetVolume(GeometryTGeo::getITSVolPattern());
  vALIC->AddNode(vITSV, 2, nullptr); // Copy number is 2 to cheat AliGeoManager::CheckSymNamesLUT

  const Int_t kLength = 100;
  Char_t vstrng[kLength] = "xxxRS"; //?
  vITSV->SetTitle(vstrng);

  // Check that we have all needed parameters
  for (Int_t j = 0; j < sNumberLayers; j++) {
    if (mLayerRadii[j] <= 0) {
      LOG(FATAL) << "Wrong layer radius for layer " << j << "(" << mLayerRadii[j] << ")";
    }
    if (mStavePerLayer[j] <= 0) {
      LOG(FATAL) << "Wrong number of staves for layer " << j << "(" << mStavePerLayer[j] << ")";
    }
    if (mUnitPerStave[j] <= 0) {
      LOG(FATAL) << "Wrong number of chips for layer " << j << "(" << mUnitPerStave[j] << ")";
    }
    if (mChipThickness[j] < 0) {
      LOG(FATAL) << "Wrong chip thickness for layer " << j << "(" << mChipThickness[j] << ")";
    }
    if (mTurboLayer[j] && mStaveWidth[j] <= 0) {
      LOG(FATAL) << "Wrong stave width for layer " << j << "(" << mStaveWidth[j] << ")";
    }
    if (mDetectorThickness[j] < 0) {
      LOG(FATAL) << "Wrong Sensor thickness for layer " << j << "(" << mDetectorThickness[j] << ")";
    }

    if (j > 0) {
      if (mLayerRadii[j] <= mLayerRadii[j - 1]) {
        LOG(FATAL) << "Layer " << j << " radius (" << mLayerRadii[j] << ") is smaller than layer " << j - 1
                   << " radius (" << mLayerRadii[j - 1] << ")";
      }
    }

    if (mChipThickness[j] == 0) {
      LOG(INFO) << "Chip thickness for layer " << j << " not set, using default";
    }
  }

  // Create the wrapper volumes
  TGeoVolume** wrapVols = nullptr;

  if (sNumberOfWrapperVolumes) {
    wrapVols = new TGeoVolume*[sNumberOfWrapperVolumes];
    for (int id = 0; id < sNumberOfWrapperVolumes; id++) {
      wrapVols[id] = createWrapperVolume(id);
      vITSV->AddNode(wrapVols[id], 1, nullptr);
    }
  }

  // Now create the actual geometry
  for (Int_t j = 0; j < sNumberLayers; j++) {
    TGeoVolume* dest = vITSV;
    mWrapperLayerId[j] = -1;

    if (mTurboLayer[j]) {
      mGeometry[j] = new V3Layer(j, kTRUE, kFALSE);
      mGeometry[j]->setStaveWidth(mStaveWidth[j]);
      mGeometry[j]->setStaveTilt(mStaveTilt[j]);
    } else {
      mGeometry[j] = new V3Layer(j, kFALSE);
    }

    mGeometry[j]->setPhi0(mLayerPhi0[j]);
    mGeometry[j]->setRadius(mLayerRadii[j]);
    mGeometry[j]->setNumberOfStaves(mStavePerLayer[j]);
    mGeometry[j]->setNumberOfUnits(mUnitPerStave[j]);
    mGeometry[j]->setChipType(mChipTypeID[j]);
    mGeometry[j]->setBuildLevel(mBuildLevel[j]);

    if (j < sNumberInnerLayers) {
      mGeometry[j]->setStaveModel(mStaveModelInnerBarrel);
    } else {
      mGeometry[j]->setStaveModel(mStaveModelOuterBarrel);
    }

    LOG(DEBUG1) << "mBuildLevel: " << mBuildLevel[j];

    if (mChipThickness[j] != 0) {
      mGeometry[j]->setChipThick(mChipThickness[j]);
    }
    if (mDetectorThickness[j] != 0) {
      mGeometry[j]->setSensorThick(mDetectorThickness[j]);
    }

    for (int iw = 0; iw < sNumberOfWrapperVolumes; iw++) {
      if (mLayerRadii[j] > mWrapperMinRadius[iw] && mLayerRadii[j] < mWrapperMaxRadius[iw]) {
        LOG(DEBUG) << "Will embed layer " << j << " in wrapper volume " << iw;

        dest = wrapVols[iw];
        mWrapperLayerId[j] = iw;
        break;
      }
    }
    mGeometry[j]->createLayer(dest);
  }

  // Finally create the services
  mServicesGeometry = new V3Services();

  createInnerBarrelServices(wrapVols[0]);
  createMiddlBarrelServices(wrapVols[1]);
  createOuterBarrelServices(wrapVols[2]);

  // TEMPORARY - These routines will be obsoleted once the new services are completed - TEMPORARY
  //  createServiceBarrel(kTRUE, wrapVols[0]);
  //  createServiceBarrel(kFALSE, wrapVols[2]);

  delete[] wrapVols; // delete pointer only, not the volumes
}

void Detector::createInnerBarrelServices(TGeoVolume* motherVolume)
{
  //
  // Creates the Inner Barrel Service structures
  //
  // Input:
  //         motherVolume : the volume hosting the services
  //
  // Output:
  //
  // Return:
  //
  // Created:      15 May 2019  Mario Sitta
  //               (partially based on P.Namwongsa implementation in AliRoot)
  // Updated:      19 Jun 2019  Mario Sitta  IB Side A added
  //

  // Create the End Wheels on Side A
  TGeoVolume* endWheelsA = mServicesGeometry->createIBEndWheelsSideA();

  motherVolume->AddNode(endWheelsA, 1, nullptr);

  // Create the End Wheels on Side C
  TGeoVolume* endWheelsC = mServicesGeometry->createIBEndWheelsSideC();

  motherVolume->AddNode(endWheelsC, 1, nullptr);
}

void Detector::createMiddlBarrelServices(TGeoVolume* motherVolume)
{
  //
  // Creates the Middle Barrel Service structures
  //
  // Input:
  //         motherVolume : the volume hosting the services
  //
  // Output:
  //
  // Return:
  //
  // Created:      24 Sep 2019  Mario Sitta
  //

  // Create the End Wheels on Side A
  mServicesGeometry->createMBEndWheelsSideA(motherVolume);

  // Create the End Wheels on Side C
  mServicesGeometry->createMBEndWheelsSideC(motherVolume);
}

void Detector::createOuterBarrelServices(TGeoVolume* motherVolume)
{
  //
  // Creates the Outer Barrel Service structures
  //
  // Input:
  //         motherVolume : the volume hosting the services
  //
  // Output:
  //
  // Return:
  //
  // Created:      27 Sep 2019  Mario Sitta
  //

  // Create the End Wheels on Side A
  mServicesGeometry->createOBEndWheelsSideA(motherVolume);

  // Create the End Wheels on Side C
  mServicesGeometry->createOBEndWheelsSideC(motherVolume);
}

// Service Barrel
void Detector::createServiceBarrel(const Bool_t innerBarrel, TGeoVolume* dest, const TGeoManager* mgr)
{
  // Creates the Service Barrel (as a simple cylinder) for IB and OB
  // Inputs:
  //         innerBarrel : if true, build IB service barrel, otherwise for OB
  //         dest        : the mother volume holding the service barrel
  //         mgr         : the gGeoManager pointer (used to get the material)
  //

  Double_t rminIB = 4.7;
  Double_t rminOB = 43.9;
  Double_t zLenOB;
  Double_t cInt = 0.22; // dimensioni cilindro di supporto interno
  Double_t cExt = 1.00; // dimensioni cilindro di supporto esterno
  //  Double_t phi1   =  180;
  //  Double_t phi2   =  360;

  TGeoMedium* medCarbonFleece = mgr->GetMedium("ITS_CarbonFleece$");

  if (innerBarrel) {
    zLenOB = ((TGeoTube*)(dest->GetShape()))->GetDz();
    //    TGeoTube*ibSuppSh = new TGeoTubeSeg(rminIB,rminIB+cInt,zLenOB,phi1,phi2);
    auto* ibSuppSh = new TGeoTube(rminIB, rminIB + cInt, zLenOB);
    auto* ibSupp = new TGeoVolume("ibSuppCyl", ibSuppSh, medCarbonFleece);
    dest->AddNode(ibSupp, 1);
  } else {
    zLenOB = ((TGeoTube*)(dest->GetShape()))->GetDz();
    auto* obSuppSh = new TGeoTube(rminOB, rminOB + cExt, zLenOB);
    auto* obSupp = new TGeoVolume("obSuppCyl", obSuppSh, medCarbonFleece);
    dest->AddNode(obSupp, 1);
  }

  return;
}

void Detector::addAlignableVolumes() const
{
  //
  // Creates entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path.
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  LOG(INFO) << "Add ITS alignable volumes";

  if (!gGeoManager) {
    LOG(FATAL) << "TGeoManager doesn't exist !";
    return;
  }

  TString path = Form("/cave_1/%s_2", GeometryTGeo::getITSVolPattern());
  TString sname = GeometryTGeo::composeSymNameITS();

  LOG(DEBUG) << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data()))
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;

  Int_t lastUID = 0;
  for (Int_t lr = 0; lr < sNumberLayers; lr++)
    addAlignableVolumesLayer(lr, path, lastUID);

  return;
}

void Detector::addAlignableVolumesLayer(int lr, TString& parent, Int_t& lastUID) const
{
  //
  // Add alignable volumes for a Layer and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  TString wrpV =
    mWrapperLayerId[lr] != -1 ? Form("%s%d_1", GeometryTGeo::getITSWrapVolPattern(), mWrapperLayerId[lr]) : "";
  TString path = Form("%s/%s/%s%d_1", parent.Data(), wrpV.Data(), GeometryTGeo::getITSLayerPattern(), lr);
  TString sname = GeometryTGeo::composeSymNameLayer(lr);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data()))
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nstaves = lrobj->getNumberOfStavesPerParent();
  for (int st = 0; st < nstaves; st++)
    addAlignableVolumesStave(lr, st, path, lastUID);

  return;
}

void Detector::addAlignableVolumesStave(Int_t lr, Int_t st, TString& parent, Int_t& lastUID) const
{
  //
  // Add alignable volumes for a Stave and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  TString path = Form("%s/%s%d_%d", parent.Data(), GeometryTGeo::getITSStavePattern(), lr, st);
  TString sname = GeometryTGeo::composeSymNameStave(lr, st);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data()))
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nhstave = lrobj->getNumberOfHalfStavesPerParent();
  Int_t start = nhstave > 0 ? 0 : -1;
  for (Int_t sst = start; sst < nhstave; sst++)
    addAlignableVolumesHalfStave(lr, st, sst, path, lastUID);

  return;
}

void Detector::addAlignableVolumesHalfStave(Int_t lr, Int_t st, Int_t hst, TString& parent, Int_t& lastUID) const
{
  //
  // Add alignable volumes for a HalfStave (if any) and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  TString path = parent;
  if (hst >= 0) {
    path = Form("%s/%s%d_%d", parent.Data(), GeometryTGeo::getITSHalfStavePattern(), lr, hst);
    TString sname = GeometryTGeo::composeSymNameHalfStave(lr, st, hst);

    LOG(DEBUG) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data()))
      LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nmodules = lrobj->getNumberOfModulesPerParent();
  Int_t start = nmodules > 0 ? 0 : -1;
  for (Int_t md = start; md < nmodules; md++)
    addAlignableVolumesModule(lr, st, hst, md, path, lastUID);

  return;
}

void Detector::addAlignableVolumesModule(Int_t lr, Int_t st, Int_t hst, Int_t md, TString& parent, Int_t& lastUID) const
{
  //
  // Add alignable volumes for a Module (if any) and its daughters
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  TString path = parent;
  if (md >= 0) {
    path = Form("%s/%s%d_%d", parent.Data(), GeometryTGeo::getITSModulePattern(), lr, md);
    TString sname = GeometryTGeo::composeSymNameModule(lr, st, hst, md);

    LOG(DEBUG) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data()))
      LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nchips = lrobj->getNumberOfChipsPerParent();
  for (Int_t ic = 0; ic < nchips; ic++)
    addAlignableVolumesChip(lr, st, hst, md, ic, path, lastUID);

  return;
}

void Detector::addAlignableVolumesChip(Int_t lr, Int_t st, Int_t hst, Int_t md, Int_t ch, TString& parent,
                                       Int_t& lastUID) const
{
  //
  // Add alignable volumes for a Chip
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  TString path = Form("%s/%s%d_%d", parent.Data(), GeometryTGeo::getITSChipPattern(), lr, ch);
  TString sname = GeometryTGeo::composeSymNameChip(lr, st, hst, md, ch);
  Int_t modUID = chipVolUID(lastUID++);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data(), modUID))
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;

  return;
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;

  // The names of the ITS sensitive volumes have the format: ITSUSensor(0...sNumberLayers-1)
  for (Int_t j = 0; j < sNumberLayers; j++) {
    volumeName = GeometryTGeo::getITSSensorPattern() + TString::Itoa(j, 10);
    v = geoManager->GetVolume(volumeName.Data());
    AddSensitiveVolume(v);
  }
}

Hit* Detector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                      const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                      unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

void Detector::Print(std::ostream* os) const
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

  // fmt = os->setf(std::ios::scientific); // set scientific floating point output
  // fmt = os->setf(std::ios::hex); // set hex for mStatus only.
  // fmt = os->setf(std::ios::dec); // every thing else decimel.
  //  *os << mModule << " ";
  //  *os << mEnergyDepositionStep << " " << mTof;
  //  *os << " " << mStartingStepX << " " << mStartingStepY << " " << mStartingStepZ;
  //    *os << " " << endl;
  // os->flags(fmt); // reset back to old formating.
  return;
}

void Detector::Read(std::istream* is)
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

std::ostream& operator<<(std::ostream& os, Detector& p)
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

std::istream& operator>>(std::istream& is, Detector& r)
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

ClassImp(o2::its::Detector);
