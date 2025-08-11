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

/// \file Detector.h
/// \brief Definition of the Detector class

#ifndef ALICEO2_FD_DETECTOR_H_
#define ALICEO2_FD_DETECTOR_H_

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h"
#include "FDBase/GeometryTGeo.h"
#include "FDBase/FDBaseParam.h"
#include "DataFormatsFD/Hit.h"
#include "Rtypes.h"
#include "TGeoManager.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include <cmath>

class FairVolume;
class TGeoVolume;

namespace o2
{
namespace fd
{
class GeometryTGeo;
}
} // namespace o2

namespace o2
{
namespace fd
{

class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool Active);

  Detector() = default;

  ~Detector() override;

  void ConstructGeometry() override;

  /// This method is an example of how to add your own point of type Hit to the clones array
  o2::fd::Hit* addHit(int trackId, unsigned int detId,
                      const math_utils::Point3D<float>& startPos,
                      const math_utils::Point3D<float>& endPos,
                      const math_utils::Vector3D<float>& startMom, double startE,
                      double endTime, double eLoss, int particlePdg);
  //   unsigned int startStatus,
  //   unsigned int endStatus);

  std::vector<o2::fd::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  // Mandatory overrides
  void BeginPrimary() override { ; }
  void FinishPrimary() override { ; }
  void InitializeO2Detector() override;
  void PostTrack() override { ; }
  void PreTrack() override { ; }
  bool ProcessHits(FairVolume* v = nullptr) override;
  void EndOfEvent() override;
  void Register() override;
  void Reset() override;

  void createMaterials();
  void buildModules();

  enum EMedia {
    Scintillator,
    Aluminium
  };

 private:
  Detector(const Detector&);
  Detector& operator=(const Detector&);

  std::vector<o2::fd::Hit>* mHits = nullptr;
  GeometryTGeo* mGeometryTGeo = nullptr;

  TGeoVolumeAssembly* buildModuleA();
  TGeoVolumeAssembly* buildModuleC();

  float ringSize(float zmod, float eta);

  unsigned int mNumberOfRingsA, mNumberOfRingsC, mNumberOfSectors;
  float mDzScint, mDzPlate;

  std::vector<float> mRingSizesA = {}, mRingSizesC = {};

  float mEtaMaxA, mEtaMaxC, mEtaMinA, mEtaMinC;
  float mZA, mZC;

  bool mPlateBehindA, mFullContainer;

  void defineSensitiveVolumes();
  void definePassiveVolumes();

  /// Transient data about track passing the sensor, needed by ProcessHits()
  struct TrackData {               // this is transient
    bool mHitStarted;              //! hit creation started
    unsigned char mTrkStatusStart; //! track status flag
    TLorentzVector mPositionStart; //! position at entrance
    TLorentzVector mMomentumStart; //! momentum
    double mEnergyLoss;            //! energy loss
  } mTrackData;                    //!

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);

} // namespace fd
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::fd::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
