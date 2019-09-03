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

#ifndef ALICEO2_FV0_DETECTOR_H_
#define ALICEO2_FV0_DETECTOR_H_

#include "TLorentzVector.h"

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h" // for Detector
#include "FV0Base/Geometry.h"
#include "DataFormatsFV0/Hit.h"

class FairModule;

class FairVolume;
class TGeoVolume;
class TGraph;

namespace o2
{
namespace fv0
{
class Geometry;
}
} // namespace o2

namespace o2
{
namespace fv0
{
class Geometry;
class Detector : public o2::base::DetImpl<Detector>
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

  /// Gets the produced hits
  std::vector<o2::fv0::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

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

  /// Registers new materials in o2::base::Detector
  void createMaterials();

  /// Creates materials and geometry
  void ConstructGeometry() override; // inherited from FairModule

  enum EMedia {
    Zero,
    Air,
    Scintillator
  }; // media IDs used in createMaterials

 private:
  /// Container for hits
  std::vector<o2::fv0::Hit>* mHits = nullptr;

  /// Geometry pointer
  Geometry* mGeometry = nullptr; //!

  /// Transient data about track passing the sensor, needed by ProcessHits()
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  o2::fv0::Hit* addHit(Int_t trackId, Int_t cellId,
                       const Point3D<float>& startPos, const Point3D<float>& endPos,
                       const Vector3D<float>& startMom, double startE,
                       double endTime, double eLoss, Int_t particlePdg);

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);
} // namespace fv0
} // namespace o2
#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::fv0::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif
