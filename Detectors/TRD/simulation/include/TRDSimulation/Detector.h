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

#include "TRDSimulation/TRsim.h"

class FairVolume;

namespace o2
{
namespace trd
{
class HitType : public o2::BaseHit
{
  Point3D<float> mPos; // cartesian position of Hit
  float mTime;         // time of flight
  int mCharge;         // energy loss
  short mDetectorID;   // the detector/sensor id

 public:
  HitType() = default; // for ROOT IO
  // constructor
  HitType(float x, float y, float z, float tof, int q, int trackid, short did)
    : mPos(x, y, z),
      mTime(tof),
      mCharge(q),
      BaseHit(trackid),
      mDetectorID(did)
  {
  }

  // getting the cartesian coordinates
  float GetX() const { return mPos.X(); }
  float GetY() const { return mPos.Y(); }
  float GetZ() const { return mPos.Z(); }
  Point3D<float> GetPos() const { return mPos; }
  // getting charge
  int GetCharge() const { return mCharge; }
  // getting the time
  float GetTime() const { return mTime; }
  // get detector + track information
  short GetDetectorID() const { return mDetectorID; }

  // modifiers
  void SetTime(float time) { mTime = time; }
  void SetCharge(int q) { mCharge = q; }
  void SetDetectorID(short detID) { mDetectorID = detID; }
  void SetX(float x) { mPos.SetX(x); }
  void SetY(float y) { mPos.SetY(y); }
  void SetZ(float z) { mPos.SetZ(z); }
  void SetXYZ(float x, float y, float z)
  {
    SetX(x);
    SetY(y);
    SetZ(z);
  }
  void SetPos(Point3D<float> const& p) { mPos = p; }

  ClassDefNV(HitType, 1);
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
  Detector(Bool_t active = true);
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
  void FinishEvent() override;
  void Reset() override;
  void EndOfEvent() override;
  void createMaterials();
  void ConstructGeometry() override;
  /// Add alignable top volumes
  void addAlignableVolumes() const override;

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  // defines/sets-up the sensitive volumes
  void defineSensitiveVolumes();

  // addHit
  template <typename T>
  void addHit(T x, T y, T z, T tof, int charge, int trackId, int detId);

  // Create TR hits
  void createTRhit(int);

  std::vector<HitType>* mHits = nullptr; ///!< Collection of TRD hits

  float mFoilDensity;
  float mGasNobleFraction;
  float mGasDensity;

  bool mTRon; // Switch for TR simulation
  TRsim* mTR; // Access to TR simulation

  float mWion; // Ionization potential

  TRDGeometry* mGeom = nullptr;

  template <typename Det>
  friend class o2::Base::DetImpl;
  ClassDefOverride(Detector, 1)
};

template <typename T>
void Detector::addHit(T x, T y, T z, T tof, int charge, int trackId, int detId)
{
  mHits->emplace_back(x, y, z, tof, charge, trackId, detId);
}

} // namespace trd
} // namespace o2

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
