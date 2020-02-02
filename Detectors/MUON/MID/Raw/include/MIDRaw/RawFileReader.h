// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <fstream>
#include <vector>
#include <array>
#include "MIDRaw/CrateParameters.h"
#include "MIDRaw/RawBuffer.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
template <typename T>
class RawFileReader
{
 public:
  bool init(const char* inFilename, bool readContinuous = false);
  bool readAllGBTs(bool reset = true);
  bool readHB(bool sendCompleteHBs = true);
  void clear();

  /// Gets the number of HBs read
  unsigned long int getNumberOfHBs(uint16_t feeId) { return mHBCounters.at(feeId); }

  /// Gets the state
  int getState() { return mState; }

  /// Gets the vector of data
  const std::vector<T>& getData() { return mBytes; }

  void setCustomRDH(const header::RAWDataHeader& rdh) { mCustomRDH = rdh; }

 private:
  void read(size_t nBytes);
  bool hasFullInfo();
  bool replaceRDH(size_t headerIndex);

  std::ifstream mFile{};                                            /// Raw file
  std::vector<T> mBytes;                                            /// Buffer
  RawBuffer<T> mBuffer;                                             /// Raw buffer handler
  std::array<unsigned long int, crateparams::sNGBTs> mHBCounters{}; /// HB counters per GBT
  bool mReadContinuous{false};                                      /// Continuous readout mode
  int mState{0};                                                    /// Status flag

  header::RAWDataHeader mCustomRDH{}; /// Custom RDH
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWFILEREADER_H */
