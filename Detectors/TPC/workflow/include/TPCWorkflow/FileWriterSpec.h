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

/// \file   FileWriterSpec.h
/// \brief  Writer for calibration data
/// \author Jens Wiechula

#ifndef TPC_FileWriterSpec_H_
#define TPC_FileWriterSpec_H_

#include "Framework/DataProcessorSpec.h"
#include <string>

namespace o2::tpc
{

enum class BranchType {
  Krypton,
  Digits,
};

const std::unordered_map<std::string, BranchType> BranchTypeMap{
  {"krypton", BranchType::Krypton},
  {"digits", BranchType::Digits},
};

const std::unordered_map<BranchType, std::string> BranchName{
  {BranchType::Krypton, "TPCBoxCluster"},
  {BranchType::Digits, "TPCDigit"},
};

const std::unordered_map<BranchType, std::string> TreeName{
  {BranchType::Krypton, "Clusters"},
  {BranchType::Digits, "o2sim"},
};

/// create a processor spec
/// read simulated TPC clusters from file and publish
template <typename T>
o2::framework::DataProcessorSpec getFileWriterSpec(const std::string inputSpec, const BranchType branchType = BranchType::Krypton);

} // namespace o2::tpc

#endif // TPC_RAWTODIGITSSPEC_H_
