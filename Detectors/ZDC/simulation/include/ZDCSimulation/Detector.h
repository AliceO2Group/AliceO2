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

#ifndef ALICEO2_ZDC_DETECTOR_H_
#define ALICEO2_ZDC_DETECTOR_H_

#include <vector>                             // for vector
#include "TGeoManager.h"                      // for gGeoManager, TGeoManager (ptr only)
#include "DetectorsBase/GeometryManager.h"    // for getSensID
#include "DetectorsBase/Detector.h"           // for Detector
#include "DetectorsCommonDataFormats/DetID.h" // for Detector
#include "ZDCBase/Geometry.h"
#include "DataFormatsZDC/Hit.h"
#include "ZDCSimulation/SpatialPhotonResponse.h"
#include "TParticle.h"
#include <utility>
#include "ZDCBase/Constants.h"

// inclusions and forward decl for fast sim
#ifdef ZDC_FASTSIM_ONNX
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
namespace o2::zdc
{
namespace fastsim
{
class NeuralFastSimulation;
namespace processors
{
class StandardScaler;
}
} // namespace fastsim
} // namespace o2::zdc
#endif

class FairVolume;

namespace o2
{
namespace zdc
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  enum ZDCMaterial {
    kWalloy = 1,
    kCuZn = 2,
    kSiO2pmc = 3,
    kSiO2pmq = 4,
    kPb = 5,
    kCu = 6,
    kFe = 7,
    kFeLowTh = 8,
    kCuLumi = 9,
    kVoidNoField = 10,
    kVoidwField = 11,
    kAir = 12,
    kAl = 13,
    kGraphite = 14
  };

  Detector(Bool_t active = true);

// if building fastsim non trivial destructor is required
#ifdef ZDC_FASTSIM_ONNX
  ~Detector() override;
#endif
#ifndef ZDC_FASTSIM_ONNX
  ~Detector() override = default;
#endif

  void InitializeO2Detector() final;

  Bool_t ProcessHits(FairVolume* v = nullptr) final;

  bool createHitsFromImage(SpatialPhotonResponse const& image, int detector);

  void Register() override;

  /// Gets the produced collections
  std::vector<o2::zdc::Hit>* getHits(int32_t iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() final;
  void EndOfEvent() final;
  void FinishPrimary() final;

  void BeginPrimary() final;

  void ConstructGeometry() final;

  void createMaterials();
  void addAlignableVolumes() const override {}

  o2::zdc::Hit* addHit(int32_t trackID, int32_t parentID, int32_t sFlag, float primaryEnergy, int32_t detID, int32_t secID,
                       math_utils::Vector3D<float> pos, math_utils::Vector3D<float> mom, float tof, math_utils::Vector3D<float> xImpact, double energyloss,
                       int32_t nphePMC, int32_t nphePMQ);

 private:
  /// copy constructor
  Detector(const Detector& rhs);

  void createAsideBeamLine();
  void createCsideBeamLine();
  void createMagnets();
  void createDetectors();

  // determine detector; sector/tower and impact coordinates given volumename and position
  void getDetIDandSecID(TString const& volname, math_utils::Vector3D<float> const& x,
                        math_utils::Vector3D<float>& xDet, int& detector, int& sector) const;

  // Define sensitive volumes
  void defineSensitiveVolumes();

  // Methods to calculate the light outpu
  Bool_t calculateTableIndexes(int& ibeta, int& iangle, int& iradius);

  void resetHitIndices();

  // common function for hit creation (can be called from multiple interfaces)
  bool createOrAddHit(int detector,
                      int sector,
                      int currentMediumid,
                      bool issecondary,
                      int nphe,
                      int trackn,
                      int parent,
                      float tof,
                      float trackenergy,
                      math_utils::Vector3D<float> const& xImp,
                      float eDep, float x, float y, float z, float px, float py, float pz)
  {
    // A new hit is created when there is nothing yet for this det + sector
    if (mCurrentHitsIndices[detector - 1][sector] == -1) {
      mTotLightPMC = mTotLightPMQ = 0;
      if (currentMediumid == mMediumPMCid) {
        mTotLightPMC = nphe;
      } else if (currentMediumid == mMediumPMQid) {
        mTotLightPMQ = nphe;
      }

      math_utils::Vector3D<float> pos(x, y, z);
      math_utils::Vector3D<float> mom(px, py, pz);
      addHit(trackn, parent, issecondary, trackenergy, detector, sector,
             pos, mom, tof, xImp, eDep, mTotLightPMC, mTotLightPMQ);
      // stack->addHit(GetDetId());
      mCurrentHitsIndices[detector - 1][sector] = mHits->size() - 1;

      mXImpact = xImp;
      return true;
    } else {
      auto& curHit = (*mHits)[mCurrentHitsIndices[detector - 1][sector]];
      // summing variables that needs to be updated (Eloss and light yield)
      curHit.setNoNumContributingSteps(curHit.getNumContributingSteps() + 1);
      int nPMC{0}, nPMQ{0};
      if (currentMediumid == mMediumPMCid) {
        mTotLightPMC += nphe;
        nPMC = nphe;
      } else if (currentMediumid == mMediumPMQid) {
        mTotLightPMQ += nphe;
        nPMQ = nphe;
      }
      if (nphe > 0) {
        curHit.SetEnergyLoss(curHit.GetEnergyLoss() + eDep);
        curHit.setPMCLightYield(curHit.getPMCLightYield() + nPMC);
        curHit.setPMQLightYield(curHit.getPMQLightYield() + nPMQ);
      }
      return true;
    }
  }

  // helper function taking care of writing the photon response pattern at certain moments
  void flushSpatialResponse();

  float mTrackEta;
  float mPrimaryEnergy;
  math_utils::Vector3D<float> mXImpact;
  float mTotLightPMC;
  float mTotLightPMQ;
  int32_t mMediumPMCid = -1;
  int32_t mMediumPMQid = -2;

  //
  /// Container for hit data
  std::vector<o2::zdc::Hit>* mHits;

  float mLumiLength = 0;         //TODO: make part of configurable params
  float mTCLIAAPERTURE = 3.5;    //TODO: make part of configurable params
  float mTCLIAAPERTURENEG = 3.5; //TODO: make part of configurable params
  float mVCollSideCCentreY = 0.; //TODO: make part of configurable params

  int mZNENVVolID = -1; // the volume id for the neutron det envelope volume
  int mZPENVVolID = -1; // the volume id for the proton det envelope volume
  int mZEMVolID = -1;   // the volume id for the e-m envelope volume

  // last principal track entered for each of the 5 detectors
  // this is the main trackID causing showering/response in the detectors
  int mLastPrincipalTrackEntered = -1;

  static constexpr int NUMDETS = 5; // number of detectors
  static constexpr int NUMSECS = 5; // number of (max) possible sectors (including COMMON one)

  // current hits per detector and per sector FOR THE CURRENT track first entering a detector
  // (as given by mLastPrincipalTrackEntered)
  // This is given as index where to find in mHits container
  int mCurrentHitsIndices[NUMDETS][NUMSECS] = {-1};

  static constexpr int ZNRADIUSBINS = 18;
  static constexpr int ZPRADIUSBINS = 28;
  static constexpr int ANGLEBINS = 90;

  float mLightTableZN[4][ANGLEBINS][ZNRADIUSBINS] = {1.}; //!
  float mLightTableZP[4][ANGLEBINS][ZPRADIUSBINS] = {1.}; //!

  SpatialPhotonResponse mNeutronResponseImage;
  // there is only one proton detector per side
  SpatialPhotonResponse mProtonResponseImage;

  TParticle mCurrentPrincipalParticle{};

  // collecting the responses for the current event
  using ParticlePhotonResponse = std::vector<std::pair<TParticle,
                                                       std::pair<SpatialPhotonResponse, SpatialPhotonResponse>>>;

  ParticlePhotonResponse mResponses;
  ParticlePhotonResponse* mResponsesPtr = &mResponses;

// fastsim model wrapper
#ifdef ZDC_FASTSIM_ONNX
  fastsim::NeuralFastSimulation* mFastSimClassifier = nullptr; //! no ROOT serialization
  fastsim::NeuralFastSimulation* mFastSimModelNeutron = nullptr; //!
  fastsim::NeuralFastSimulation* mFastSimModelProton = nullptr;  //!

  // Scalers for models inputs
  fastsim::processors::StandardScaler* mClassifierScaler = nullptr; //!
  fastsim::processors::StandardScaler* mModelScalerNeutron = nullptr; //!
  fastsim::processors::StandardScaler* mModelScalerProton = nullptr;  //!

  // container for fastsim model responses
  using FastSimResults = std::vector<std::array<long, 5>>; //!
  FastSimResults mFastSimResults;                          //!

  // converts FastSim model results to Hit
  bool FastSimToHits(const Ort::Value& response, const TParticle& particle, int detector);

  // determines detector geometry "pixel sizes"
  constexpr std::pair<const int, const int> determineDetectorSize(int detector)
  {
    if (detector == ZNA || detector == ZNC) {
      return {Geometry::ZNDIVISION[0] * Geometry::ZNSECTORS[0] * 2, Geometry::ZNDIVISION[1] * Geometry::ZNSECTORS[1] * 2};
    } else if (detector == ZPA || detector == ZPC) {
      return {Geometry::ZPDIVISION[0] * Geometry::ZPSECTORS[0] * 2, Geometry::ZPDIVISION[1] * Geometry::ZPSECTORS[1] * 2};
    } else {
      return {-1, -1};
    }
  }
#endif

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace zdc
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::zdc::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
