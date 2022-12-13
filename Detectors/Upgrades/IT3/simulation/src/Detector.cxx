// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITS3Base/GeometryTGeo.h"
#include "ITS3Simulation/Detector.h"
#include "ITS3Simulation/V3Layer.h"
#include "ITS3Simulation/V3Services.h"

#include "DetectorsBase/Stack.h"
#include "SimulationDataFormat/TrackReference.h"

// FairRoot includes
#include "FairDetector.h"      // for FairDetector
#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "FairRootManager.h"   // for FairRootManager
#include "FairRun.h"           // for FairRun
#include "FairRuntimeDb.h"     // for FairRuntimeDb
#include "FairVolume.h"        // for FairVolume
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
using namespace o2::its3;

Detector::Detector()
  : o2::base::DetImpl<Detector>("IT3", kTRUE),
    mTrackData(),
    mNumberOfDetectors(-1),
    mModifyGeometry(kFALSE),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

// We need this as a method to access members
void Detector::configITS3()
{
  // build ITS3 upgrade detector
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

  std::vector<std::array<double, 2>> IBtdr5dat; // 18 24 30 its2 = (22 mm, 31 mm and 39 mm and 194 mm, 247 mm, 353 mm and 405 mm)
  IBtdr5dat.emplace_back(std::array<double, 2>{1.8f, 27.15});
  IBtdr5dat.emplace_back(std::array<double, 2>{2.4f, 27.15});
  IBtdr5dat.emplace_back(std::array<double, 2>{3.0f, 27.15});
  IBtdr5dat.emplace_back(std::array<double, 2>{7.0f, 27.15});

  // Radii are from last TDR (ALICE-TDR-017.pdf Tab. 1.1, rMid is mean value)
  // OB only !!
  const double OBtdr5dat[sNumberOuterLayers][kNPar] = {
    {-1, 19.6, -1, 4., 0., 24},  // for others: -, rMid, -, NMod/HStave, phi0, nStaves // 24 was 49
    {-1, 24.55, -1, 4., 0., 30}, // 30 was 61
    {-1, 34.39, -1, 7., 0., 42}, // 42 was 88
    {-1, 39.34, -1, 7., 0., 48}  // 48 was 100
  };

  double rLr, phi0;
  int nStaveLr, nModPerStaveLr;

  const int kNWrapVol = 3;
  const double wrpRMin[kNWrapVol] = {2.1, 19.3, 33.32};
  const double wrpRMax[kNWrapVol] = {15.4, 29.14, 46.0};
  const double wrpZSpan[kNWrapVol] = {70., 93., 163.6};

  for (int iw = 0; iw < kNWrapVol; iw++) {
    defineWrapperVolume(iw, wrpRMin[iw], wrpRMax[iw], wrpZSpan[iw]);
  }

  // Build OB only (4 layers)
  for (int idLr{0}; idLr < sNumberOuterLayers; idLr++) {
    // int im = idLr - mNumberOfInnerLayers + sNumberInnerLayers;
    rLr = OBtdr5dat[idLr][kRmd];
    phi0 = OBtdr5dat[idLr][kPhi0];

    nStaveLr = TMath::Nint(OBtdr5dat[idLr][kNStave]);
    nModPerStaveLr = TMath::Nint(OBtdr5dat[idLr][kNModPerStave]);
    int nChipsPerStaveLr = nModPerStaveLr;
    defineLayer(sNumberInnerLayers + idLr, phi0, rLr, nStaveLr, nModPerStaveLr, ChipThicknessOB, Segmentation::SensorLayerThickness,
                kSensTypeID, kBuildLevel);
  }

  static constexpr float SensorLayerThickness = 30.e-4;
  createOuterBarrel(kTRUE);

  auto idLayer{0};
  for (auto& layerData : IBtdr5dat) {
    defineInnerLayerITS3(idLayer, layerData[0], layerData[1], SensorLayerThickness, 0, 0);
    ++idLayer;
  }

  // Redefine Wrapper Volume 0 using information on IB radii
  static constexpr float safety = 0.5; // Educated guess, may be improved
  defineWrapperVolume(0, IBtdr5dat[0][0] - safety, IBtdr5dat[3][0] + safety, wrpZSpan[0]);
  LOG(info) << "HERE!\n";
}

Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("IT3", active),
    mTrackData(),
    mNumberOfInnerLayers(sNumberInnerLayers),
    mTotalNumberOfLayers(sNumberLayers),
    mNumberOfDetectors(-1),
    mModifyGeometry(kFALSE),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  createAllArrays();
  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }

  if (mTotalNumberOfLayers > 0) { // if not, we'll Fatal-ize in CreateGeometry
    for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
      mITS3Layer[j] = kFALSE;
      mLayerRadii[j] = 0.;
      mLayerZLen[j] = 0.;
      mStavePerLayer[j] = 0;
      mUnitPerStave[j] = 0;
      mChipThickness[j] = 0.;
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

  configITS3();
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),
    mNumberOfInnerLayers(rhs.mNumberOfInnerLayers),
    mNumberOfDetectors(rhs.mNumberOfDetectors),
    mModifyGeometry(rhs.mModifyGeometry),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }
}

Detector::~Detector()
{
  if (mHits) {
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
  mHits = nullptr;

  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
    mLayerName[j].Form("%s%d", GeometryTGeo::getITSSensorPattern(), j); // See V3Layer
  }

  return *this;
}

void Detector::createAllArrays()
{
  // Create all arrays
  // We have to do it here because now the number of Inner Layers is
  // not fixed, so we don't know in advance how many to create: they
  // cannot be anymore static arrays as in the default version
  // M.S. 21 mar 2020 (13th day of Italian Coronavirus lockdown)

  mLayerID = new Int_t[mTotalNumberOfLayers];
  mLayerName = new TString[mTotalNumberOfLayers];
  mWrapperLayerId = new Int_t[mTotalNumberOfLayers];
  mITS3Layer = new Bool_t[mTotalNumberOfLayers];
  mLayerPhi0 = new Double_t[mTotalNumberOfLayers];
  mLayerRadii = new Double_t[mTotalNumberOfLayers];
  mLayerZLen = new Double_t[mTotalNumberOfLayers];
  mStavePerLayer = new Int_t[mTotalNumberOfLayers];
  mUnitPerStave = new Int_t[mTotalNumberOfLayers];
  mChipThickness = new Double_t[mTotalNumberOfLayers];
  mDetectorThickness = new Double_t[mTotalNumberOfLayers];
  mChipTypeID = new UInt_t[mTotalNumberOfLayers];
  mBuildLevel = new Int_t[mTotalNumberOfLayers];
  mGeometry = new V3Layer*[mTotalNumberOfLayers];
}

void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  defineSensitiveVolumes();
  for (int i = 0; i < mTotalNumberOfLayers; i++) {
    mLayerID[i] = gMC ? TVirtualMC::GetMC()->VolId(mLayerName[i]) : 0;
  }
  mGeometryTGeo = GeometryTGeo::Instance();
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
  while ((lay < mTotalNumberOfLayers) && (notSens = (volID != mLayerID[lay]))) {
    ++lay;
  }
  if (notSens) {
    return kFALSE; // RS: can this happen? This method must be called for sensors only?
  }

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
  if (!startHit) {
    mTrackData.mEnergyLoss += fMC->Edep();
  }
  if (!(startHit | stopHit)) {
    return kFALSE; // do noting
  }

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

  // Rohacell (C9 H13 N1 O2)
  Float_t aRohac[4] = {12.01, 1.01, 14.010, 16.};
  Float_t zRohac[4] = {6., 1., 7., 8.};
  Float_t wRohac[4] = {9., 13., 1., 2.};
  Float_t dRohac = 0.05;

  // Araldite 2011
  Float_t dAraldite = 1.05;

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
  // Rohacell
  o2::base::Detector::Mixture(32, "ROHACELL$", aRohac, zRohac, dRohac, -4, wRohac);
  o2::base::Detector::Medium(32, "ROHACELL$", 32, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  // ERG Duocel
  o2::base::Detector::Material(33, "ERGDUOCEL$", 12.0107, 6, 0.06, 999, 999);
  o2::base::Detector::Medium(33, "ERGDUOCEL$", 33, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  // Impregnated carbon fleece
  // (as educated guess we assume 50% carbon fleece 50% Araldite glue)
  o2::base::Detector::Material(34, "IMPREG_FLEECE$", 12.0107, 6, 0.5 * (dAraldite + 0.4), 999, 999);
  o2::base::Detector::Medium(34, "IMPREG_FLEECE$", 34, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

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
    LOG(fatal) << "id " << id << " of wrapper volume is not in 0-" << sNumberOfWrapperVolumes - 1 << " range";
  }

  mWrapperMinRadius[id] = rmin;
  mWrapperMaxRadius[id] = rmax;
  mWrapperZSpan[id] = zspan;
}

void Detector::defineLayer(Int_t nlay, Double_t phi0, Double_t r, Int_t nstav, Int_t nunit, Double_t lthick,
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

  LOG(info) << "L# " << nlay << " Phi:" << phi0 << " R:" << r << " Nst:" << nstav << " Nunit:" << nunit
            << " Lthick:" << lthick << " Dthick:" << dthick << " DetID:" << dettypeID << " B:" << buildLevel;

  if (nlay >= mTotalNumberOfLayers || nlay < 0) {
    LOG(error) << "Wrong layer number " << nlay;
    return;
  }

  mLayerPhi0[nlay] = phi0;
  mLayerRadii[nlay] = r;
  mStavePerLayer[nlay] = nstav;
  mUnitPerStave[nlay] = nunit;
  mChipThickness[nlay] = lthick;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
}

void Detector::defineInnerLayerITS3(Int_t nlay, Double_t r, Double_t zlen, Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
  //     Sets the layer parameters
  // Inputs:
  //          nlay    layer number
  //          r       layer radius
  //          zlen    layer length
  //          dthick  detector thickness (if omitted, defaults to 0)
  //          dettypeID  ??
  //          buildLevel (if 0, all geometry is build, used for material budget studies)
  // Outputs:
  //   none.
  // Return:
  //   none.

  LOG(info) << "L# " << nlay << " with ITS3 geo R:" << r
            << " Dthick:" << dthick << " DetID:" << dettypeID << " B:" << buildLevel;

  if (nlay >= mTotalNumberOfLayers || nlay < 0) {
    LOG(error) << "Wrong layer number " << nlay;
    return;
  }

  mITS3Layer[nlay] = kTRUE;
  mLayerRadii[nlay] = r;
  mLayerZLen[nlay] = zlen;
  mDetectorThickness[nlay] = dthick;
  mChipTypeID[nlay] = dettypeID;
  mBuildLevel[nlay] = buildLevel;
  LOG(info) << "Added layer: " << nlay << std::endl;
}

void Detector::getLayerParameters(Int_t nlay, Double_t& phi0, Double_t& r, Int_t& nstav, Int_t& nmod, Double_t& lthick, Double_t& dthick, UInt_t& dettype) const
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

  if (nlay >= mTotalNumberOfLayers || nlay < 0) {
    LOG(error) << "Wrong layer number " << nlay;
    return;
  }

  phi0 = mLayerPhi0[nlay];
  r = mLayerRadii[nlay];
  nstav = mStavePerLayer[nlay];
  nmod = mUnitPerStave[nlay];
  lthick = mChipThickness[nlay];
  dthick = mDetectorThickness[nlay];
  dettype = mChipTypeID[nlay];
}

TGeoVolume* Detector::createWrapperVolume(Int_t id)
{
  // Creates an air-filled wrapper cylindrical volume
  // For OB a Pcon is needed to host the support rings
  // while avoiding overlaps with MFT structures and OB cones

  const Double_t suppRingAZlen = 4.;
  const Double_t coneRingARmax = 33.96;
  const Double_t coneRingAZlen = 5.6;
  const Double_t suppRingCZlen[3] = {4.8, 4.0, 2.4};
  const Double_t suppRingsRmin[3] = {23.35, 20.05, 35.4};

  if (mWrapperMinRadius[id] < 0 || mWrapperMaxRadius[id] < 0 || mWrapperZSpan[id] < 0) {
    LOG(fatal) << "Wrapper volume " << id << " was requested but not defined";
  }

  // Now create the actual shape and volume
  TGeoShape* tube;
  Double_t zlen;
  switch (id) {
    case 0: // IB Layer 0,1,2: simple cylinder
    {
      TGeoTube* wrap = new TGeoTube(mWrapperMinRadius[id], mWrapperMaxRadius[id], mWrapperZSpan[id] / 2.);
      tube = (TGeoShape*)wrap;
    } break;
    case 1: // MB Layer 3,4: complex Pcon to avoid MFT overlaps
    {
      TGeoPcon* wrap = new TGeoPcon(0, 360, 6);
      zlen = mWrapperZSpan[id] / 2 + suppRingCZlen[0];
      wrap->DefineSection(0, -zlen, suppRingsRmin[0], mWrapperMaxRadius[id]);
      zlen = mWrapperZSpan[id] / 2 + suppRingCZlen[1];
      wrap->DefineSection(1, -zlen, suppRingsRmin[0], mWrapperMaxRadius[id]);
      wrap->DefineSection(2, -zlen, suppRingsRmin[1], mWrapperMaxRadius[id]);
      wrap->DefineSection(3, -mWrapperZSpan[id] / 2., suppRingsRmin[1], mWrapperMaxRadius[id]);
      wrap->DefineSection(4, -mWrapperZSpan[id] / 2., mWrapperMinRadius[id], mWrapperMaxRadius[id]);
      zlen = mWrapperZSpan[id] / 2 + suppRingAZlen;
      wrap->DefineSection(5, zlen, mWrapperMinRadius[id], mWrapperMaxRadius[id]);
      tube = (TGeoShape*)wrap;
    } break;
    case 2: // OB Layer 5,6: simpler Pcon to avoid OB cones overlaps
    {
      TGeoPcon* wrap = new TGeoPcon(0, 360, 6);
      zlen = mWrapperZSpan[id] / 2;
      wrap->DefineSection(0, -zlen, suppRingsRmin[2], mWrapperMaxRadius[id]);
      zlen -= suppRingCZlen[2];
      wrap->DefineSection(1, -zlen, suppRingsRmin[2], mWrapperMaxRadius[id]);
      wrap->DefineSection(2, -zlen, mWrapperMinRadius[id], mWrapperMaxRadius[id]);
      zlen = mWrapperZSpan[id] / 2 - coneRingAZlen;
      wrap->DefineSection(3, zlen, mWrapperMinRadius[id], mWrapperMaxRadius[id]);
      wrap->DefineSection(4, zlen, coneRingARmax, mWrapperMaxRadius[id]);
      wrap->DefineSection(5, mWrapperZSpan[id] / 2, coneRingARmax, mWrapperMaxRadius[id]);
      tube = (TGeoShape*)wrap;
    } break;
    default: // Can never happen, keeps gcc quiet
      break;
  }

  TGeoMedium* medAir = gGeoManager->GetMedium("IT3_AIR$");

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
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");

  if (!vALIC) {
    LOG(fatal) << "Could not find the top volume";
  }
  new TGeoVolumeAssembly(GeometryTGeo::getITSVolPattern());
  TGeoVolume* vITSV = geoManager->GetVolume(GeometryTGeo::getITSVolPattern());
  vALIC->AddNode(vITSV, 2, new TGeoTranslation(0, 30., 0)); // Copy number is 2 to cheat AliGeoManager::CheckSymNamesLUT

  const Int_t kLength = 100;
  Char_t vstrng[kLength] = "xxxRS"; //?
  vITSV->SetTitle(vstrng);

  // Check that we have all needed parameters
  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
    if (mLayerRadii[j] <= 0) {
      LOG(fatal) << "Wrong layer radius for layer " << j << "(" << mLayerRadii[j] << ")";
    }
    if (mStavePerLayer[j] <= 0 && !mITS3Layer[j]) {
      LOG(fatal) << "Wrong number of staves for layer " << j << "(" << mStavePerLayer[j] << ")";
    }
    if (mUnitPerStave[j] <= 0 && !mITS3Layer[j]) {
      LOG(fatal) << "Wrong number of chips for layer " << j << "(" << mUnitPerStave[j] << ")";
    }
    if (mChipThickness[j] < 0 && !mITS3Layer[j]) {
      LOG(fatal) << "Wrong chip thickness for layer " << j << "(" << mChipThickness[j] << ")";
    }
    if (mDetectorThickness[j] < 0) {
      LOG(fatal) << "Wrong Sensor thickness for layer " << j << "(" << mDetectorThickness[j] << ")";
    }

    if (j > 0 && // Always check IB, check OB only if present
        ((j < mNumberOfInnerLayers) || mCreateOuterBarrel)) {
      if (mLayerRadii[j] <= mLayerRadii[j - 1]) {
        LOG(fatal) << "Layer " << j << " radius (" << mLayerRadii[j] << ") is smaller than layer " << j - 1
                   << " radius (" << mLayerRadii[j - 1] << ")";
      }
    }

    if (mChipThickness[j] == 0 && !mITS3Layer[j]) {
      LOG(info) << "Chip thickness for layer " << j << " not set, using default";
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
  if (!mCreateOuterBarrel) {
    mTotalNumberOfLayers = mNumberOfInnerLayers;
  }

  // Now create the actual geometry
  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
    TGeoVolume* dest = vITSV;
    mWrapperLayerId[j] = -1;
    mGeometry[j] = new V3Layer(j, kFALSE);
    mGeometry[j]->setNumberOfInnerLayers(mNumberOfInnerLayers);

    if (mITS3Layer[j]) {
      mGeometry[j]->setIsITS3(kTRUE);
      mGeometry[j]->setIBModuleZLength(mLayerZLen[j]);
    }

    mGeometry[j]->setPhi0(mLayerPhi0[j]);
    mGeometry[j]->setRadius(mLayerRadii[j]);
    mGeometry[j]->setNumberOfStaves(mStavePerLayer[j]);
    mGeometry[j]->setNumberOfUnits(mUnitPerStave[j]);
    mGeometry[j]->setChipType(mChipTypeID[j]);
    mGeometry[j]->setBuildLevel(mBuildLevel[j]);

    LOG(debug1) << "mBuildLevel: " << mBuildLevel[j];

    if (mChipThickness[j] != 0) {
      mGeometry[j]->setChipThick(mChipThickness[j]);
    }
    if (mDetectorThickness[j] != 0) {
      mGeometry[j]->setSensorThick(mDetectorThickness[j]);
    }

    for (int iw = 0; iw < sNumberOfWrapperVolumes; iw++) {
      if (mLayerRadii[j] > mWrapperMinRadius[iw] && mLayerRadii[j] < mWrapperMaxRadius[iw]) {
        LOG(debug) << "Will embed layer " << j << " in wrapper volume " << iw;

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
  createOuterBarrelSupports(vITSV);

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
  // Created:      11 Feb 2021  Mario Sitta
  //               (Version rewritten for ITS3 geometry)
  //

  // Create and insert the IB Layer supports (cylinder and foam wedges)
  TGeoVolume* ibSupports = mServicesGeometry->createInnerBarrelSupports();

  motherVolume->AddNode(ibSupports, 1, nullptr);
  motherVolume->AddNode(ibSupports, 2, new TGeoRotation("", 180, 0, 0));
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

void Detector::createOuterBarrelSupports(TGeoVolume* motherVolume)
{
  //
  // Creates the Outer Barrel Service structures
  //
  // Input:
  //         motherVolume : the volume hosting the supports
  //
  // Output:
  //
  // Return:
  //
  // Created:      26 Jan 2020  Mario Sitta
  //

  // Create the Cone on Side A
  mServicesGeometry->createOBConeSideA(motherVolume);

  // Create the Cone on Side C
  mServicesGeometry->createOBConeSideC(motherVolume);
}

void Detector::addAlignableVolumes() const
{
  //
  // Creates entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path.
  //
  // Created:      06 Mar 2018  Mario Sitta First version (mainly ported from AliRoot)
  //

  LOG(info) << "Add ITS3 alignable volumes";

  if (!gGeoManager) {
    LOG(fatal) << "TGeoManager doesn't exist !";
    return;
  }

  TString path = Form("/cave_1/barrel_1/%s_2", GeometryTGeo::getITSVolPattern());
  TString sname = GeometryTGeo::composeSymNameITS3();

  LOG(debug) << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  Int_t lastUID = 0;
  for (Int_t lr = 0; lr < mTotalNumberOfLayers; lr++) {
    addAlignableVolumesLayer(lr, path, lastUID);
  }

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
  TString path;
  path = Form("%s/%s/%s%d_1", parent.Data(), wrpV.Data(), GeometryTGeo::getITSLayerPattern(), lr);
  TString sname = GeometryTGeo::composeSymNameLayer(lr);

  LOG(debug) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nstaves = lrobj->getNumberOfStavesPerParent();
  for (int st = 0; st < nstaves; st++) {
    addAlignableVolumesStave(lr, st, path, lastUID);
  }

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

  LOG(debug) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nhstave = lrobj->getNumberOfHalfStavesPerParent();
  Int_t start = nhstave > 0 ? 0 : -1;
  for (Int_t sst = start; sst < nhstave; sst++) {
    addAlignableVolumesHalfStave(lr, st, sst, path, lastUID);
  }

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

    LOG(debug) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nmodules = lrobj->getNumberOfModulesPerParent();
  Int_t start = nmodules > 0 ? 0 : -1;
  for (Int_t md = start; md < nmodules; md++) {
    addAlignableVolumesModule(lr, st, hst, md, path, lastUID);
  }

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

    LOG(debug) << "Add " << sname << " <-> " << path;

    if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
      LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
    }
  }

  const V3Layer* lrobj = mGeometry[lr];
  Int_t nchips = lrobj->getNumberOfChipsPerParent();
  for (Int_t ic = 0; ic < nchips; ic++) {
    addAlignableVolumesChip(lr, st, hst, md, ic, path, lastUID);
  }

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

  LOG(debug) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data(), modUID)) {
    LOG(fatal) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  return;
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;
  TString volumeName;

  // The names of the ITS3 sensitive volumes have the format: ITSUSensor(0...sNumberLayers-1)
  for (Int_t j = 0; j < mTotalNumberOfLayers; j++) {
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
ClassImp(o2::its3::Detector);
