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

#include "ITSMFTSimulation/Hit.h"
#include "EC0Base/GeometryTGeo.h"
#include "EC0Simulation/Detector.h"
#include "EC0Simulation/EC0Layer.h"

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

using namespace o2::ec0;
using o2::itsmft::Hit;

//_________________________________________________________________________________________________
Detector::Detector()
  : o2::base::DetImpl<Detector>("EC0", kTRUE),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

//_________________________________________________________________________________________________
void Detector::buildBasicEC0(int nLayers, Float_t z_first, Float_t z_length, Float_t etaIn, Float_t etaOut, Float_t Layerx2X0)
{
  // Build a basic parametrized EC0 detector with nLayers equally spaced between z_first and z_first+z_length
  // Covering pseudo rapidity [etaIn,etaOut]. Passive silicon thinkness computed to match layer x/X0

  mNumberOfLayers = nLayers;
  Float_t sensorThickness = 30.e-4;
  mLayerName.resize(mNumberOfLayers);
  mLayerID.resize(mNumberOfLayers);

  for (int layerNumber = 0; layerNumber < mNumberOfLayers; layerNumber++) {
    std::string layerName = GeometryTGeo::getEC0LayerPattern() + std::to_string(layerNumber);
    mLayerName[layerNumber] = layerName;

    // Adds evenly spaced layers
    Float_t layerZ = z_first + (layerNumber * z_length / (mNumberOfLayers - 1)) * std::copysign(1, z_first);
    Float_t rIn = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaIn))));
    Float_t rOut = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaOut))));

    auto& thisLayer = mLayers.emplace_back(layerNumber, layerName, layerZ, rIn, rOut, sensorThickness, Layerx2X0);
  }
}

//_________________________________________________________________________________________________
void Detector::buildEC0V1()
{
  //Build EC0 detector according to
  //https://indico.cern.ch/event/992488/contributions/4174473/attachments/2168881/3661331/tracker_parameters_werner_jan_11_2021.pdf

  mNumberOfLayers = 10;
  Float_t sensorThickness = 30.e-4;
  Float_t layersx2X0 = 1.e-2;
  std::vector<std::array<Float_t, 5>> layersConfig{
    {-16., .5, 3., sensorThickness, 0.1f * layersx2X0}, // {z_layer, r_in, r_out, sensor_thickness, Layerx2X0}
    {-20., .5, 3., sensorThickness, 0.1f * layersx2X0},
    {-24., .5, 3., sensorThickness, 0.1f * layersx2X0},
    {-77., 3.5, 35., sensorThickness, layersx2X0},
    {-100., 3.5, 35., sensorThickness, layersx2X0},
    {-122., 3.5, 35., sensorThickness, layersx2X0},
    {-150., 3.5, 100., sensorThickness, layersx2X0},
    {-180., 3.5, 100., sensorThickness, layersx2X0},
    {-220., 3.5, 100., sensorThickness, layersx2X0},
    {-279., 3.5, 100., sensorThickness, layersx2X0}};

  mLayerName.resize(mNumberOfLayers);
  mLayerID.resize(mNumberOfLayers);

  for (int layerNumber = 0; layerNumber < mNumberOfLayers; layerNumber++) {
    std::string layerName = GeometryTGeo::getEC0LayerPattern() + std::to_string(layerNumber);
    mLayerName[layerNumber] = layerName;
    auto& z = layersConfig[layerNumber][0];
    auto& rIn = layersConfig[layerNumber][1];
    auto& rOut = layersConfig[layerNumber][2];
    auto& thickness = layersConfig[layerNumber][3];
    auto& x0 = layersConfig[layerNumber][4];

    // Adds evenly spaced layers
    auto& thisLayer = mLayers.emplace_back(layerNumber, layerName, z, rIn, rOut, thickness, x0);
  }
}

//_________________________________________________________________________________________________
Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("EC0", active),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{

  //buildBasicEC0(); // BasicEC0 = Parametrized detector equidistant layers
  buildEC0V1(); // EC0V1 = Werner's layout
}

//_________________________________________________________________________________________________
Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),

    /// Container for data points
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  mLayerID = rhs.mLayerID;
  mLayerName = rhs.mLayerName;
  mNumberOfLayers = rhs.mNumberOfLayers;
}

//_________________________________________________________________________________________________
Detector::~Detector()
{

  if (mHits) {
    // delete mHits;
    o2::utils::freeSimVector(mHits);
  }
}

//_________________________________________________________________________________________________
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

  mLayerID = rhs.mLayerID;
  mLayerName = rhs.mLayerName;
  mNumberOfLayers = rhs.mNumberOfLayers;
  mLayers = rhs.mLayers;
  mTrackData = rhs.mTrackData;

  /// Container for data points
  mHits = nullptr;

  return *this;
}

//_________________________________________________________________________________________________
void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  LOG(INFO) << "Initialize EC0 O2Detector";

  mGeometryTGeo = GeometryTGeo::Instance();

  defineSensitiveVolumes();
}

//_________________________________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  Int_t lay = 0, volID = vol->getMCid();
  while ((lay <= mNumberOfLayers) && (volID != mLayerID[lay])) {
    ++lay;
  }

  auto stack = (o2::data::Stack*)fMC->GetStack();

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
    int chipindex = lay;

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

//_________________________________________________________________________________________________
void Detector::createMaterials()
{
  Int_t ifield = 2;
  Float_t fieldm = 10.0;
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);

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

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  o2::base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
}

//_________________________________________________________________________________________________
void Detector::EndOfEvent() { Reset(); }

//_________________________________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

//_________________________________________________________________________________________________
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

//_________________________________________________________________________________________________
void Detector::ConstructGeometry()
{
  // Create detector materials
  createMaterials();

  // Construct the detector geometry
  createGeometry();
}

//_________________________________________________________________________________________________
void Detector::createGeometry()
{

  mGeometryTGeo = GeometryTGeo::Instance();

  TGeoVolume* volEC0 = new TGeoVolumeAssembly(GeometryTGeo::getEC0VolPattern());

  LOG(INFO) << "GeometryBuilder::buildGeometry volume name = " << GeometryTGeo::getEC0VolPattern();

  TGeoVolume* vALIC = gGeoManager->GetVolume("barrel");
  if (!vALIC) {
    LOG(FATAL) << "Could not find the top volume";
  }

  LOG(DEBUG) << "buildGeometry: "
             << Form("gGeoManager name is %s title is %s", gGeoManager->GetName(), gGeoManager->GetTitle());

  for (int iLayer = 0; iLayer < mNumberOfLayers; iLayer++) {
    mLayers[iLayer].createLayer(volEC0);
  }

  vALIC->AddNode(volEC0, 2, new TGeoTranslation(0., 30., 0.));

  for (int iLayer = 0; iLayer < mNumberOfLayers; iLayer++) {
    mLayerID[iLayer] = gMC ? TVirtualMC::GetMC()->VolId(Form("%s%d", GeometryTGeo::getEC0SensorPattern(), iLayer)) : 0;
    LOG(INFO) << "mLayerID for layer " << iLayer << " = " << mLayerID[iLayer];
  }
}

//_________________________________________________________________________________________________
void Detector::addAlignableVolumes() const
{
  LOG(INFO) << "Add EC0 alignable volumes";

  if (!gGeoManager) {
    LOG(FATAL) << "TGeoManager doesn't exist !";
    return;
  }

  TString path = Form("/cave_1/barrel_1/%s_2", GeometryTGeo::getEC0VolPattern());
  TString sname = GeometryTGeo::composeSymNameEC0();

  LOG(DEBUG) << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname.Data(), path.Data())) {
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  Int_t lastUID = 0;
  for (Int_t lr = 0; lr < mNumberOfLayers; lr++) {
    addAlignableVolumesLayer(lr, path, lastUID);
  }

  return;
}

//_________________________________________________________________________________________________
void Detector::addAlignableVolumesLayer(int lr, TString& parent, Int_t& lastUID) const
{

  TString path = Form("%s/%s%d_1", parent.Data(), GeometryTGeo::getEC0LayerPattern(), lr);

  TString sname = GeometryTGeo::composeSymNameLayer(lr);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data())) {
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }
  addAlignableVolumesChip(lr, path, lastUID);
  return;
}

//_________________________________________________________________________________________________
void Detector::addAlignableVolumesChip(int lr, TString& parent, Int_t& lastUID) const
{

  TString path = Form("%s/%s%d_1", parent.Data(), GeometryTGeo::getEC0ChipPattern(), lr);

  TString sname = GeometryTGeo::composeSymNameChip(lr);
  Int_t modUID = chipVolUID(lastUID++);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data(), modUID)) {
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }
  //addAlignableVolumesSensor(lr, path, lastUID);
  return;
}

//_________________________________________________________________________________________________
void Detector::addAlignableVolumesSensor(int lr, TString& parent, Int_t& lastUID) const
{

  TString path = Form("%s/%s%d_1", parent.Data(), GeometryTGeo::getEC0SensorPattern(), lr);

  TString sname = GeometryTGeo::composeSymNameSensor(lr);
  Int_t modUID = chipVolUID(lastUID++);

  LOG(DEBUG) << "Add " << sname << " <-> " << path;

  if (!gGeoManager->SetAlignableEntry(sname, path.Data(), modUID)) {
    LOG(FATAL) << "Unable to set alignable entry ! " << sname << " : " << path;
  }

  return;
}

//_________________________________________________________________________________________________
void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;

  // The names of the EC0 sensitive volumes have the format: EC0Sensor(0...sNumberLayers-1)
  for (Int_t j = 0; j < mNumberOfLayers; j++) {
    volumeName = o2::ec0::GeometryTGeo::getEC0SensorPattern() + std::to_string(j);
    v = geoManager->GetVolume(volumeName.Data());
    LOG(INFO) << "Adding EC0 Sensitive Volume => " << v->GetName() << std::endl;
    AddSensitiveVolume(v);
  }
}

//_________________________________________________________________________________________________
Hit* Detector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                      const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                      unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

//_________________________________________________________________________________________________
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

//_________________________________________________________________________________________________
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

//_________________________________________________________________________________________________
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

//_________________________________________________________________________________________________
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

ClassImp(o2::ec0::Detector);
