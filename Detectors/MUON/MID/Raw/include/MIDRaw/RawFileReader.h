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

/// \file   MIDRaw/RawFileReader.h
/// \brief  MID raw file reader
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   25 November 2019
#ifndef O2_MID_RAWFILEREADER_H
#define O2_MID_RAWFILEREADER_H

#include <cstdint>
#include <fstream>
#include <vector>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace mid
{
class RawFileReader
{
 public:
  bool init(const char* inFilename, bool readContinuous = false);
  bool readHB(bool sendCompleteHBs = false);
  void clear();

  /// Gets the state
  int getState() { return mState; }

  /// Gets the vector of data
  const std::vector<uint8_t>& getData() { return mBytes; }

  void setCustomRDH(const header::RAWDataHeader& rdh) { mCustomRDH = rdh; }
  void setCustomPayloadSize(uint16_t memorySize = 0x2000, uint16_t offsetToNext = 0x2000);

 private:
  void read(size_t nBytes);
  bool replaceRDH(size_t headerIndex);

  std::ifstream mFile{};                         /// Raw file
  std::vector<uint8_t> mBytes;                   /// Buffer
  static constexpr unsigned int sHeaderSize{64}; /// Header size in bytes
  bool mReadContinuous{false};                   /// Continuous readout mode
  int mState{0};                                 /// Status flag

  header::RAWDataHeader mCustomRDH{}; /// Custom RDH
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWFILEREADER_H */
