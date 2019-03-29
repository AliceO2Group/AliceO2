// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_V0_DETECTOR_H_
#define ALICEO2_V0_DETECTOR_H_

#include "TLorentzVector.h"

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h" // for Detector
#include "V0Base/Geometry.h"
#include "V0Simulation/Hit.h"

class FairModule;

class FairVolume;
class TGeoVolume;
class TGraph;

// TODO: Check run/O2HitMerger.h:471 - problem with merging stage of o2sim (no o2sim.root file is created)
// TODO: Perhaps it will start working correctly once any geometry with sensitive parts is defined and hit processing works

namespace o2
{
namespace v0
{
class Geometry;
}
} // namespace o2

namespace o2
{
namespace v0
{
// using HitType = o2::BasicXYZEHit<float>;
class Geometry;
class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  /// Default constructor
  Detector();

  /// Default destructor
  ~Detector() override;

  /// Constructor with on/off flag
  /// \param isActive  kTRUE for active detectors (ProcessHits() will be called),
  ///                  kFALSE for inactive detectors
  Detector(Bool_t isActive);

  /// Initializes the detector (adds sensitive volume)
  void InitializeO2Detector() override;

  /// This method is called for each step during simulation (see FairMCApplication::Stepping())
  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  // ------------------------------------------------------------------

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  std::vector<o2::v0::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }
  /// Gets the produced hits

  /// Has to be called after each event to reset the containers
  void Reset() override;

  /// Called at the end of event
  void EndOfEvent() override;

  // TODO: From MFT -> are they needed?
  //    void FinishPrimary() override { ; }
  //    void FinishRun() override { ; }
  //    void BeginPrimary() override { ; }
  //    void PostTrack() override { ; }
  //    void PreTrack() override { ; }
  //    void SetSpecialPhysicsCuts() override { ; }

  // TODO: move to private
  /// Creates materials for the detector
  void createMaterials();

  /// Creates materials and geometry
  void ConstructGeometry() override; // inherited from FairModule

  enum EMedia {
    Zero,
    Air,
    Scintillator
  }; // media IDs used in CreateMaterials

 private:
  /// Container for hits
  std::vector<o2::v0::Hit>* mHits = nullptr;

  /// Geometry pointer
  Geometry* mGeometry = nullptr; //! Geometry

  /// Transient data about track passing the sensor, needed by ProcessHits()
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  o2::v0::Hit* addHit(Int_t trackId, Int_t cellId, Int_t particleId,
                      TVector3 startPos, TVector3 endPos,
                      TVector3 startMom, double startE,
                      double endTime, double eLoss, float eTot, float eDep);

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1)
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
} // namespace v0
} // namespace o2
#ifdef USESHM
namespace o2
{
namespace Base
{
template <>
struct UseShm<o2::v0::Detector> {
  static constexpr bool value = true;
};
} // namespace Base
} // namespace o2
#endif

#endif
