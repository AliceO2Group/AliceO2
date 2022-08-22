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

/// @file   DCSParserSpec.h

#ifndef O2_ITS_DCS_PARSER_SPEC_H
#define O2_ITS_DCS_PARSER_SPEC_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/InputRecordWalker.h"
#include <fairmq/Device.h>

#include <ITSMFTReconstruction/RawPixelDecoder.h> //o2::itsmft::RawPixelDecoder

#include "Framework/RawDeviceService.h"
#include "Headers/DataHeader.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/CcdbApi.h"
#include "DataFormatsDCS/DCSConfigObject.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace its
{

// Defining global constants for variables which have not yet been manually set
const int UNSET_INT = -1111;
const short int UNSET_SHORT = -1111;

class ITSDCSParser : public Task
{
 public:
  ITSDCSParser();
  ~ITSDCSParser() override = default;

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

  //////////////////////////////////////////////////////////////////
 private:
  // Helper functions
  void finalizeOutput();
  void updateMemoryFromInputString(const std::string&);
  void saveToOutput();
  void resetMemory();
  void pushToCCDB(ProcessingContext&);
  void updateAndCheck(int&, const int);
  void updateAndCheck(short int&, const short int);
  void writeChipInfo(o2::dcs::DCSconfigObject_t&, const std::string&, const unsigned short int);
  std::vector<std::string> vectorizeStringList(const std::string&, const std::string&);
  std::vector<unsigned short int> vectorizeStringListInt(const std::string&, const std::string&);
  std::string intVecToStr(const std::vector<unsigned short int>&, const std::string&);

  // Data to be updated in each line of input
  std::string mStaveName = "";
  int mRunNumber = UNSET_INT;
  int mConfigVersion = UNSET_INT;
  short int mRunType = UNSET_SHORT;
  std::vector<unsigned short int> mDisabledChips;
  std::vector<std::vector<unsigned short int>> mMaskedDoubleCols;

  // maps from chip ID to flagged double columns / pixels
  std::map<unsigned short int, std::vector<unsigned short int>> mDoubleColsDisableEOR;
  std::map<unsigned short int, std::vector<unsigned short int>> mPixelFlagsEOR;

  std::string mSelfName = "";

  // Keep track of whether the endOfStream() or stop() has been called
  bool mStopped = false;

  // Whether to use verbose output
  bool mVerboseOutput = false;

  // DCS config object for storing output string
  o2::dcs::DCSconfigObject_t mConfigDCS;

  // Ccdb url for ccdb upload withing the wf
  std::string mCcdbUrl = "";
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSDCSParserSpec();

} // namespace its
} // namespace o2

#endif
