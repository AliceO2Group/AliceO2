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

/// \file   MIDGlobalMapping/ExtendedMappingInfo.h
/// \brief  Extended mapping info
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2023

#ifndef O2_MID_EXTENDEDMAPPINGINFO_H
#define O2_MID_EXTENDEDMAPPINGINFO_H

#include <string>

namespace o2
{
namespace mid
{
/// Extended mapping info
struct ExtendedMappingInfo {
  int id;               ///< Unique ID
  std::string rpc;      ///< RPC name
  int deId;             ///< Detection element ID
  int columnId;         ///< Column ID
  int lineId;           ///< Line ID
  int stripId;          ///< Strip ID
  int cathode;          ///< Bending (0) or Non-bending (1) planes
  int locId;            ///< Local board ID
  std::string locIdDcs; ///< Local board ID for DCS
  int xpos;             ///< Position X
  int ypos;             ///< Position Y
  int xwidth;           ///< Width X (signed)
  int ywidth;           ///< Width y
};
} // namespace mid
} // namespace o2
#endif