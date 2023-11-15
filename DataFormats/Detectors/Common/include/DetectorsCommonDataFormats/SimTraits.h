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

/// @brief global simulation properties per detectors ID
///
/// Specifying branch names, hit types, and other properties
/// per DetID. Allows to write generic code that works for data manipulation
/// across detectors.
///
/// @author Sandro Wenzel, sandro.wenzel@cern.ch

#ifndef O2_BASE_SIMTRAITS_
#define O2_BASE_SIMTRAITS_

#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace detectors
{

/// Static class standardizing names / branches / other properties for simulation properties
class SimTraits
{
  using VS = std::vector<std::string>;

 public:
  // initialization fragile since depends on correct order. Can we do better?

  // clang-format off
  static inline const std::array<std::vector<std::string>, DetID::nDetectors> DETECTORBRANCHNAMES =
    { /*ITS*/ VS{ "ITSHit" },
      /*TPC*/ VS{ "TPCHitsShiftedSector0",
                  "TPCHitsShiftedSector1",
                  "TPCHitsShiftedSector2",
                  "TPCHitsShiftedSector3",
                  "TPCHitsShiftedSector4",
                  "TPCHitsShiftedSector5",
                  "TPCHitsShiftedSector6",
                  "TPCHitsShiftedSector7",
                  "TPCHitsShiftedSector8",
                  "TPCHitsShiftedSector9",
                  "TPCHitsShiftedSector10",
                  "TPCHitsShiftedSector11",
                  "TPCHitsShiftedSector12",
                  "TPCHitsShiftedSector13",
                  "TPCHitsShiftedSector14",
                  "TPCHitsShiftedSector15",
                  "TPCHitsShiftedSector16",
                  "TPCHitsShiftedSector17",
                  "TPCHitsShiftedSector18",
                  "TPCHitsShiftedSector19",
                  "TPCHitsShiftedSector20",
                  "TPCHitsShiftedSector21",
                  "TPCHitsShiftedSector22",
                  "TPCHitsShiftedSector23",
                  "TPCHitsShiftedSector24",
                  "TPCHitsShiftedSector25",
                  "TPCHitsShiftedSector26",
                  "TPCHitsShiftedSector27",
                  "TPCHitsShiftedSector28",
                  "TPCHitsShiftedSector29",
                  "TPCHitsShiftedSector30",
                  "TPCHitsShiftedSector31",
                  "TPCHitsShiftedSector32",
                  "TPCHitsShiftedSector33",
                  "TPCHitsShiftedSector34",
                  "TPCHitsShiftedSector35"},
      /*TRD*/ VS{ "TRDHit" },
      /*TOF*/ VS{ "TOFHit" },
      /*PHS*/ VS{ "PHSHit" },
      /*CPV*/ VS{ "CPVHit" },
      /*EMC*/ VS{ "EMCHit" },
      /*HMP*/ VS{ "HMPHit" },
      /*MFT*/ VS{ "MFTHit" },
      /*MCH*/ VS{ "MCHHit" },
      /*MID*/ VS{ "MIDHit" },
      /*ZDC*/ VS{ "ZDCHit" },
      /*FT0*/ VS{ "FT0Hit" },
      /*FV0*/ VS{ "FV0Hit" },
      /*FDD*/ VS{ "FDDHit" },
      /*TST*/ VS{ "TSTHit" }, // former ACO
      /*CTP*/ VS{ "CTPHit" },
      /*FOC*/ VS{ "FOCHit" }
#ifdef ENABLE_UPGRADES
      ,
      /*IT3*/ VS{ "IT3Hit" },
      /*TRK*/ VS{ "TRKHit" },
      /*FT3*/ VS{ "FT3Hit" },
      /*FCT*/ VS{ "FCTHit" }
#endif
    };
  // clang-format on

  // branches that are related to kinematics and general event information
  static inline const std::vector<std::string> KINEMATICSBRANCHES =
    {"MCTrack", "MCEventHeader", "TrackRefs"};

  ClassDefNV(SimTraits, 1);
};

} // namespace detectors

// some forward declarations
// forward declares the HitTypes
namespace trd
{
class Hit;
}
namespace itsmft
{
class Hit;
}
namespace tof
{
class HitType;
}
namespace emcal
{
class Hit;
}
namespace phos
{
class Hit;
}
namespace ft0
{
class HitType;
}
namespace hmpid
{
class HitType;
}
namespace fdd
{
class Hit;
}
namespace fv0
{
class Hit;
}
namespace mch
{
class Hit;
}
namespace mid
{
class Hit;
}
namespace zdc
{
class Hit;
}
namespace tpc
{
class HitGroup;
}

namespace detectors
{

// mapping the DetID to hit types
// traits for detectors
template <int N>
struct DetIDToHitTypes {
  using HitType = void;
};

// specialize for detectors
template <>
struct DetIDToHitTypes<o2::detectors::DetID::TRD> {
  using HitType = o2::trd::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::ITS> {
  using HitType = o2::itsmft::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::MFT> {
  using HitType = o2::itsmft::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::TOF> {
  using HitType = o2::tof::HitType;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::EMC> {
  using HitType = o2::emcal::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::PHS> {
  using HitType = o2::phos::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::FT0> {
  using HitType = o2::ft0::HitType;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::FV0> {
  using HitType = o2::fv0::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::MCH> {
  using HitType = o2::mch::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::MID> {
  using HitType = o2::mid::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::ZDC> {
  using HitType = o2::zdc::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::HMP> {
  using HitType = o2::hmpid::HitType;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::TPC> {
  using HitType = o2::tpc::HitGroup;
};
#ifdef ENABLE_UPGRADES
template <>
struct DetIDToHitTypes<o2::detectors::DetID::IT3> {
  using HitType = o2::itsmft::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::TRK> {
  using HitType = o2::itsmft::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::FT3> {
  using HitType = o2::itsmft::Hit;
};
template <>
struct DetIDToHitTypes<o2::detectors::DetID::FCT> {
  using HitType = o2::itsmft::Hit;
};
#endif

} // namespace detectors

} // namespace o2

#endif
