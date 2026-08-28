// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Constants.h
/// \brief  General constants in FD3

#ifndef ALICEO2_FD3_CONSTANTS_
#define ALICEO2_FD3_CONSTANTS_

namespace o2
{
namespace fd3
{
struct Constants {
  static constexpr unsigned int nsect = 8;
  static constexpr unsigned int nringsScint = 5;
  static constexpr unsigned int nringsCher = 1;

  static constexpr float dzscint = 4.0f;
  static constexpr float dzcher = 4.0f;

  static constexpr float etaMin_scintA = 2.5f;
  static constexpr float etaMax_scintA_v1 = 4.9f;
  static constexpr float etaMax_scintA_v2 = 5.5f;
  static constexpr float etaMin_cherA = 4.0f;
  static constexpr float etaMax_cherA_v1 = 4.9f;
  static constexpr float etaMax_cherA_v2 = 5.5f;

  static constexpr float etaMin_scintC = -4.9f;
  static constexpr float etaMax_scintC = -2.5f;
  static constexpr float etaMin_cherC = -4.9f;
  static constexpr float etaMax_cherC = -4.0f;

  static constexpr float zscint = 410.0f;
  static constexpr float zcher = 430.0f;

  static constexpr float rsize = 1.6f; // quartz radiator transverse size
};

} // namespace fd3
} // namespace o2
#endif
