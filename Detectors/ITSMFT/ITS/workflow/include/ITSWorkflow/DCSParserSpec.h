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

///************************************************************************************************///
/// Chip parameters legend        -> - Stave: L<layer_number>_<stave number>
///                                 - Hs_pos: 0 for lower, 1 for upper
///                                 - Hic_pos: 0 for IB, from 1-7 for OB
///                                 - ChipID: 0-8 for IB, 0-6 and 8-14 for OB
///                                 - Disabled: 1 if the chip is disabled, 0 if not
///                                 - Dcol_masked: <col_masked_1>|<col_masked_2>|...
///                                 - Dcol_masked_eor: <col_masked_eor_1>|<col_masked_eor_2>|...
///                                 - Pixel_flags: <region_1>|<region_2>|...
///                                 - String_OK:  1 if the string for the stave is complete,
///                                               0 if the terminator of the string is missing
///                                              -1 if the string for the stave is missing
///
/// String with terminator '!'    -> The string is complete: for each chip appearing in
///                                 the string all the informations are stored.
///                                 - String_OK = 1
///
/// String without terminator '!' -> The string is incomplete: all the chip information
///                                 are stored until the last one that is complete
///                                 (e.g until the last delimiter found).
///                                 All the fields missing in the string are set to -1.
///                                 - String_OK = 0
///
/// Stave missing in the EOR file -> The only parameter stored is the stave name.
///                                 All the other parameters are set to -1.
///                                 - String_OK = -1
///***********************************************************************************************///

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
  void saveMissingToOutput();
  void resetMemory();
  void pushToCCDB(ProcessingContext&);
  bool updatePosition(size_t&, size_t&, const std::string&, const char*, const std::string&, bool ignoreNpos = false);
  void updateAndCheck(int&, const int);
  void updateAndCheck(short int&, const short int);
  void writeChipInfo(const std::string&, const short int);
  std::vector<std::string> listStaves();

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

  // Whether to use verbose output
  bool mVerboseOutput = false;

  // Control on string integrity
  bool mTerminationString = true;

  // DCS config object for storing output string
  o2::dcs::DCSconfigObject_t mConfigDCS;

  // Ccdb url for ccdb upload withing the wf
  std::string mCcdbUrl = "";

  // Vector containing all the staves listed in the EOR file
  std::vector<string> mSavedStaves = {};
};

// Create a processor spec
o2::framework::DataProcessorSpec getITSDCSParserSpec();

} // namespace its
} // namespace o2

#endif
