// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DETECTOR_H_
#define ALICEO2_TRD_DETECTOR_H_

#include <vector>
#include "DetectorsBase/Detector.h"
#include "SimulationDataFormat/BaseHits.h"
#include "CommonUtils/ShmAllocator.h"

class FairVolume;

namespace o2
{
namespace trd
{
class HitType : public o2::BasicXYZEHit<float>
{
 public:
  using BasicXYZEHit<float>::BasicXYZEHit;
};
} // namespace trd
} // namespace o2

#ifdef USESHM
namespace std
{
template <>
class allocator<o2::trd::HitType> : public o2::utils::ShmAllocator<o2::trd::HitType>
{
};
} // namespace std
#endif

namespace o2
{
namespace trd
{
class TRDGeometry;

class Detector : public o2::Base::DetImpl<Detector>
{
 public:
  Detector(Bool_t active=true);

  ~Detector() override;

  void InitializeO2Detector() override;

  bool ProcessHits(FairVolume* v = nullptr) override;

  void Register() override;

  std::vector<HitType>* getHits(int iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void Reset() override;
  void EndOfEvent() override;

  void createMaterials();
  void ConstructGeometry() override;

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  // defines/sets-up the sensitive volumes
  void defineSensitiveVolumes();

  // addHit
  template <typename T>
  void addHit(T x, T y, T z, T time, T energy, int trackId, int detId);

  std::vector<HitType>* mHits = nullptr; ///!< Collection of TRD hits

  float mFoilDensity;
  float mGasNobleFraction;
  float mGasDensity;

  TRDGeometry* mGeom = nullptr;

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1)
};

template <typename T>
void Detector::addHit(T x, T y, T z, T time, T energy, int trackId, int detId)
{
  mHits->emplace_back(x, y, z, time, energy, trackId, detId);
}

} // end namespace trd
} // end global namespace

#ifdef USESHM
namespace o2
{
namespace Base
{
template <>
struct UseShm<o2::trd::Detector> {
  static constexpr bool value = true;
};
} // namespace Base
} // namespace o2
#endif
#endif
