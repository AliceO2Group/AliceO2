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

#ifndef ALICEO2_FVD_DETECTOR_H_
#define ALICEO2_FVD_DETECTOR_H_

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h"
#include "DataFormatsFVD/Hit.h"
#include "FVDBase/GeometryTGeo.h"

class FairModule;
class FairVolume;
class TGeoVolume;

namespace o2
{
namespace fvd
{
class GeometryTGeo;
class Hit;
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(Bool_t Active);

  Detector() = default;
  ~Detector() override;

  void InitializeO2Detector() override;

  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  o2::fvd::Hit* addHit(Int_t trackId, Int_t cellId,
                       const math_utils::Point3D<float>& startPos, const math_utils::Point3D<float>& endPos,
                       const math_utils::Vector3D<float>& startMom, double startE,
                       double endTime, double eLoss, Int_t particlePdg);

  std::vector<o2::fvd::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() override;
  void EndOfEvent() override { Reset(); }

  void CreateMaterials();
  void ConstructGeometry() override;

  enum EMedia {
    Scintillator,
  };



 private:
  Detector(const Detector& rhs);
  Detector& operator=(const Detector&);

  std::vector<o2::fvd::Hit>* mHits = nullptr;
  GeometryTGeo* mGeometry = nullptr; 

  /// Transient data about track passing the sensor, needed by ProcessHits()
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    TLorentzVector mPositionStart; //! position at entrance 
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 2);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);

} // namespace fvd
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::fvd::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
