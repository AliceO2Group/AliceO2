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

#ifndef ALICEO2_HMPID_DETECTOR_H_
#define ALICEO2_HMPID_DETECTOR_H_

#include <vector>
#include "DetectorsBase/Detector.h"
#include "DataFormatsHMP/Hit.h"

class TGeoVolume;
class TGeoHMatrix;

namespace o2
{
namespace hmpid
{
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(Bool_t active = true);
  ~Detector() override = default;

  std::vector<o2::hmpid::HitType>* getHits(int iColl) const
  {
    if (iColl == 0) {
      return mHits;
    }
    return nullptr;
  }

  void InitializeO2Detector() override;
  bool ProcessHits(FairVolume* v) override;
  o2::hmpid::HitType* AddHit(float x, float y, float z, float time, float energy, Int_t trackId, Int_t detId);
  void GenFee(float qtot);
  Bool_t IsLostByFresnel();
  float Fresnel(float ene, float pdoti, Bool_t pola);
  void Register() override;
  void Reset() override;
  // void AddAlignableVolumes() const;
  void IdealPosition(int iCh, TGeoHMatrix* pMatrix);
  void IdealPositionCradle(int iCh, TGeoHMatrix* pMatrix);
  void createMaterials();
  void ConstructGeometry() override;
  void ConstructOpGeometry() override;
  void defineOpticalProperties();
  void EndOfEvent() override { Reset(); }

  // for the geometry sub-parts
  TGeoVolume* createAbsorber(float tickness);
  TGeoVolume* createChamber(int number);
  TGeoVolume* CreateCradle();
  TGeoVolume* CradleBaseVolume(TGeoMedium* med, double l[7], const char* name);

 private:
  // copy constructor for CloneModule
  Detector(const Detector&);

  std::vector<o2::hmpid::HitType>* mHits = nullptr; ///!< Collection of HMPID hits
  enum EMedia {
    kAir = 1,
    kRoha = 2,
    kSiO2 = 3,
    kC6F14 = 4,
    kCH4 = 5,
    kCsI = 6,
    kAl = 7,
    kCu = 8,
    kW = 9,
    kNeo = 10,
    kAr = 11
  };

  std::vector<TGeoVolume*> mSensitiveVolumes; //!

  // Define volume IDs
  int mHpad0VolID = -1;
  int mHpad1VolID = -1;
  int mHpad2VolID = -1;
  int mHpad3VolID = -1;
  int mHpad4VolID = -1;
  int mHpad5VolID = -1;
  int mHpad6VolID = -1;

  int mHcel0VolID = -1;
  int mHcel1VolID = -1;
  int mHcel2VolID = -1;
  int mHcel3VolID = -1;
  int mHcel4VolID = -1;
  int mHcel5VolID = -1;
  int mHcel6VolID = -1;

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};

} // end namespace hmpid
} // end namespace o2

// enable shared mem for HMPID
#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::hmpid::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif /* ALICEO2_HMPID_DETECTOR_H_ */
