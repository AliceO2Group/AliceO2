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

#ifndef EMCAL_PEDESTAL_HELPER_H_
#define EMCAL_PEDESTAL_HELPER_H_

#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "EMCALReconstruction/Channel.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/Mapper.h"
#include "EMCALCalib/CalibDB.h"
#include "EMCALCalib/Pedestal.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <bitset>

namespace o2::emcal
{

class EMCALPedestalHelper
{

 public:
  EMCALPedestalHelper() = default;
  ~EMCALPedestalHelper() = default;

  /// \brief Encodes the pedestal object into a string. This function fills fMeanPed which is then converted to a string in createInstructionString
  /// \param obj pedestal object as stored in production ccdb
  /// \param runNum current runnumber. If -1, will not be added to string that goes in the ccdb, otherwise runNum is the first entrey in the string
  std::vector<char> createPedestalInstruction(const Pedestal& obj, const int runNum = -1);

  /// \brief print the vector produced by createInstructionString in a textfile
  void dumpInstructions(const std::string_view filename, const gsl::span<char>& data);

 private:
  /// \brief initialize fMeanPed with zeros
  void setZero();

  /// \brief converts fMeanPed to a vector of char
  /// \param runNum current runnumber. If -1, will not be added to string that goes in the ccdb, otherwise runNum is the first entrey in the string
  std::vector<char> createInstructionString(const int runNum = -1);

  static constexpr short kNSM = 20;    ///< number of SuperModules
  static constexpr short kNRCU = 2;    ///< number of readout crates (and DDLs) per SM
  static constexpr short kNDTC = 40;   ///< links for full SRU
  static constexpr short kNBranch = 2; ///< low gain/high gain
  static constexpr short kNFEC = 10;   ///< 0..9, when including LED Ref
  static constexpr short kNChip = 5;   ///< really 0,2..4, i.e. skip #1
  static constexpr short kNChan = 16;
  short fMeanPed[kNSM][kNRCU][kNBranch][kNFEC][kNChip][kNChan];
};

} // namespace o2::emcal

#endif