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

/// \file ReadRaw.h
/// \brief Reads raw data and converts to digits
/// \author Maciej.Slupecki@cern.ch, arvind.khuntia@cern.ch, based on the FT0 code
// RAW data format description: DataFormat/Detectors/FIT/FV0/RawEventData

#ifndef ALICEO2_FV0_READRAW_H_
#define ALICEO2_FV0_READRAW_H_

#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <vector>
#include "TBranch.h"
#include "TTree.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFV0/BCData.h"
#include "DataFormatsFV0/ChannelData.h"
#include "DataFormatsFV0/LookUpTable.h"
#include "DataFormatsFV0/RawEventData.h"

namespace o2
{
namespace fv0
{
class ReadRaw
{
 public:
  ReadRaw() = default;
  ReadRaw(bool doConversionToDigits, const std::string inputRawFilePath = "fv0.raw", const std::string outputRawFilePath = "fv0digitsFromRaw.root");
  void readRawData(const LookUpTable& lut);
  void writeDigits(const std::string& outputDigitsFilePath);
  void close();

 private:
  std::ifstream mRawFileIn;
  std::map<o2::InteractionRecord, std::vector<ChannelData>> mDigitAccum; // digit accumulator

  template <typename T>
  TBranch* getOrMakeBranch(TTree& tree, std::string brname, T* ptr)
  {
    if (auto br = tree.GetBranch(brname.c_str())) {
      br->SetAddress(static_cast<void*>(ptr));
      return br;
    }
    // otherwise make it
    return tree.Branch(brname.c_str(), ptr);
  }

  ClassDefNV(ReadRaw, 1);
};

} // namespace fv0
} // namespace o2
#endif
