// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @file   DetID.cxx
/// @author Ruben Shahoyan
/// @brief  detector ids, masks, names class implementation

#include "DetectorsBase/DetID.h"
#include "FairLogger.h"

using namespace o2::Base;

ClassImp(DetID);


DetID::DetID(ID id) : mID(id)
{
  if (id < First || id > Last) {
    LOG(FATAL) << "Unknown detector ID: " << toInt(id) << FairLogger::endl;
  }
}

constexpr std::array<const char[4], DetID::nDetectors> DetID::sDetNames;
constexpr std::array<std::int32_t, DetID::nDetectors> DetID::sMasks;
