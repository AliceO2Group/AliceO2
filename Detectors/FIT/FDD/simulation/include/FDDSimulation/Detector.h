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
/// \author michal.broz@cern.ch

#ifndef ALICEO2_FDD_DETECTOR_H_
#define ALICEO2_FDD_DETECTOR_H_

#include "SimulationDataFormat/BaseHits.h"
#include "DetectorsBase/Detector.h"
#include "DataFormatsFDD/Hit.h"
#include "FDDBase/Geometry.h"

class FairModule;
class FairVolume;
class TGeoVolume;

namespace o2
{
namespace fdd
{
class Geometry;
class Hit;
class Detector : public o2::base::DetImpl<Detector>
{
 public:
  Detector(Bool_t Active);

  /// Default constructor
  Detector() = default;

  /// Default destructor
  ~Detector() override;

  void InitializeO2Detector() override;

  Bool_t ProcessHits(FairVolume* v = nullptr) override;

  /// Registers the produced collections in FAIRRootManager
  void Register() override;

  o2::fdd::Hit* addHit(int trackID, unsigned short detID, const TVector3& Pos, double Time, double eLoss, int nPhot);

  std::vector<o2::fdd::Hit>* getHits(Int_t iColl)
  {
    if (iColl == 0)
      return mHits;
    return nullptr;
  }

  void Reset() override;
  void EndOfEvent() override { Reset(); }

  void CreateMaterials();
  void ConstructGeometry() override;

 private:
  /// copy constructor (used in MT)
  Detector(const Detector& rhs);

  /// Container for data points
  std::vector<o2::fdd::Hit>* mHits = nullptr;

  Detector& operator=(const Detector&);

  Geometry* mGeometry = nullptr; //! Geometry

  template <typename Det>
  friend class o2::base::DetImpl;
  ClassDefOverride(Detector, 1)
};

// Input and output function for standard C++ input/output.
std::ostream& operator<<(std::ostream& os, Detector& source);
std::istream& operator>>(std::istream& os, Detector& source);

} // namespace fdd
} // namespace o2

#ifdef USESHM
namespace o2
{
namespace base
{
template <>
struct UseShm<o2::fdd::Detector> {
  static constexpr bool value = true;
};
} // namespace base
} // namespace o2
#endif
#endif
