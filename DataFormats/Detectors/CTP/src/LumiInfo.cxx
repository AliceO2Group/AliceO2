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

/// \file LumiInfo.cxx

#include "DataFormatsCTP/LumiInfo.h"
#include "DataFormatsCTP/Configuration.h"
#include <fairlogger/Logger.h>

using namespace o2::ctp;

void LumiInfo::printInputs() const
{
  LOG(info) << "Lumi inp1:" << inp1 << ":" << o2::ctp::CTPInputsConfiguration::getInputNameFromIndex(inp1) << " inp2:" << inp2 << ":" << o2::ctp::CTPInputsConfiguration::getInputNameFromIndex(inp2);
}
