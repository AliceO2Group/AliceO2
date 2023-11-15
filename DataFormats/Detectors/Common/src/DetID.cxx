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

/// @file   DetID.cxx
/// @author Ruben Shahoyan
/// @brief  detector ids, masks, names class implementation

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/StringUtils.h"
#include <cassert>
#include <string>
#include <fairlogger/Logger.h>

using namespace o2::detectors;

ClassImp(o2::detectors::DetID);

constexpr const char* DetID::sDetNames[DetID::nDetectors + 1];

// redundant declarations
constexpr DetID::ID DetID::ITS, DetID::TPC, DetID::TRD, DetID::TOF, DetID::PHS, DetID::CPV, DetID::EMC,
  DetID::HMP, DetID::MFT, DetID::MCH, DetID::MID, DetID::ZDC, DetID::FT0, DetID::FV0, DetID::FDD, DetID::TST, DetID::CTP, DetID::FOC, DetID::First, DetID::Last;

#ifdef ENABLE_UPGRADES
constexpr DetID::ID DetID::IT3;
constexpr DetID::ID DetID::TRK;
constexpr DetID::ID DetID::FT3;
constexpr DetID::ID DetID::FCT;
#endif

constexpr int DetID::nDetectors;

/// detector masks from any non-alpha-num delimiter-separated list (empty if NONE is supplied)
DetID::mask_t DetID::getMask(const std::string_view detList)
{
  DetID::mask_t mask;
  std::string ss(detList), sname{};
  if (ss.find(NONE) != std::string::npos) {
    return mask;
  }
  if (ss.find(ALL) != std::string::npos) {
    mask = FullMask;
    return mask;
  }
  std::replace(ss.begin(), ss.end(), ' ', ',');
  std::replace(ss.begin(), ss.end(), ';', ',');
  auto dv = o2::utils::Str::tokenize(ss, ',');
  for (auto& dname : dv) {
    auto id = nameToID(dname.c_str());
    if (id < 0) {
      throw std::runtime_error(fmt::format("Wrong entry {:s} in detectors list {:s}", dname, detList));
    }
    mask |= 0x1u << id;
  }
  return mask;
}
/*
//_______________________________
constexpr DetID::DetID(const char* name) : mID(nameToID(name, First))
{
  // construct from the name
  assert(mID < nDetectors);
}
*/
//_______________________________
std::string DetID::getNames(DetID::mask_t mask, char delimiter)
{
  // construct from the name
  std::string ns;
  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (mask[id]) {
      if (!ns.empty()) {
        ns += delimiter;
      }
      ns += getName(id);
    }
  }
  return ns;
}
