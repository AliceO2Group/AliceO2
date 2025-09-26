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
/// \brief ECal geometry creation and hit processing
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_DETECTOR_H
#define ALICEO2_ECAL_DETECTOR_H

#include <DetectorsBase/Detector.h>
#include <ECalBase/Hit.h>
#include <ECalBase/GeometryTGeo.h>
#include <TLorentzVector.h>
#include <TString.h>

namespace o2
{
namespace ecal
{
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(bool active = 1);
  ~Detector();

  // Mandatory overrides
  void ConstructGeometry() override;
  void BeginPrimary() override {}
  void FinishPrimary() override {}
  void InitializeO2Detector() override;
  void PostTrack() override {}
  void PreTrack() override {}
  bool ProcessHits(FairVolume* v = nullptr) override;
  void EndOfEvent() override { Reset(); }
  void Register() override;
  void Reset() override;
  std::vector<o2::ecal::Hit>* getHits(int iColl) const { return !iColl ? mHits : nullptr; }

 private:
  void createMaterials();
  void createGeometry();
  void defineSamplingFactor();
  std::unordered_map<int, int> mSuperParentIndices; //! Super parent indices (track index - superparent index)
  int currentTrackId = -1;                          // current track index
  int superparentId = -1;                           // superparent index
  GeometryTGeo* mGeometryTGeo;                      //!
  std::vector<o2::ecal::Hit>* mHits;                //!
  double mSamplingFactorTransportModel = 1.;

 protected:
  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1);
};
} // namespace ecal
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::ecal::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif

#endif // ALICEO2_ECAL_DETECTOR_H