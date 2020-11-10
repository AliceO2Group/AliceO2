// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ReadRaw.h
/// \brief Reads raw data and converts to digits
/// \author Maciej.Slupecki@cern.ch, arvind.khuntia@cern.ch, based on the FT0 code
// RAW data format description: DataFormat/Detectors/FIT/FDD/RawEventData

#ifndef ALICEO2_FDD_READRAW_H_
#define ALICEO2_FDD_READRAW_H_

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
#include "DataFormatsFDD/Digit.h"
#include "DataFormatsFDD/ChannelData.h"
#include "DataFormatsFDD/LookUpTable.h"
#include "DataFormatsFDD/RawEventData.h"

namespace o2
{
namespace fdd
{
class ReadRaw
{
 public:
  ReadRaw() = default;
  ReadRaw(bool doConversionToDigits, const std::string inputRawFilePath = "fdd.raw", const std::string outputRawFilePath = "fdddigitsFromRaw.root");
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

} // namespace fdd
} // namespace o2
#endif
