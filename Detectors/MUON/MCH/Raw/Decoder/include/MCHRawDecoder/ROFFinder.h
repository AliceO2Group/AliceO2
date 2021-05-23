// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ROFFinder.h
/// \brief Class to group the fired pads according to their time stamp
///
/// \author Andrea Ferrero, CEA

#ifndef ALICEO2_MCH_ROFFINDER_H_
#define ALICEO2_MCH_ROFFINDER_H_

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <gsl/span>

#include "DataFormatsMCH/Digit.h"
#include "MCHRawDecoder/DataDecoder.h"
#include "DataFormatsMCH/ROFRecord.h"

namespace o2
{
namespace mch
{
namespace raw
{

class ROFFinder
{
 public:
  using RawDigit = DataDecoder::RawDigit;
  using RawDigitId = size_t;
  struct RawDigitIdComp {
    const DataDecoder::RawDigitVector* digits{nullptr};
    RawDigitIdComp() = default;
    bool operator()(const RawDigitId& id1, const RawDigitId& id2)
    {
      const RawDigit& d1 = (*digits)[id1];
      const RawDigit& d2 = (*digits)[id2];
      return (d1 < d2);
    }
  };
  using RawDigitIdVector = std::vector<RawDigitId>;

  ROFFinder(const DataDecoder::RawDigitVector& digits, uint32_t firstTForbit);
  ~ROFFinder();

  void process(bool dummyROFs = false);

  o2::InteractionRecord digitTime2IR(const RawDigit& digit);

  std::optional<DataDecoder::RawDigit> getOrderedDigit(int i);
  RawDigitIdVector getOrderedDigits() { return mOrderedDigits; }
  std::vector<o2::mch::ROFRecord> getROFRecords() { return mOutputROFs; }

  char* saveDigitsToBuffer(size_t& bufSize);
  char* saveROFRsToBuffer(size_t& bufSize);

  bool isRofTimeMonotonic();
  bool isDigitsTimeAligned();
  void dumpOutputDigits();
  void dumpOutputROFs();

 private:
  void sortDigits();
  void storeROF();

  const DataDecoder::RawDigitVector& mInputDigits;
  uint32_t mFirstTForbit;

  int mFirstIdx{-1};
  int mEntries{0};
  o2::InteractionRecord mIR;

  RawDigitIdComp mRawDigitIdComp;
  RawDigitIdVector mOrderedDigits;
  std::vector<o2::mch::ROFRecord> mOutputROFs{};
};

} // namespace raw
} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_ROFFINDERSIMPLE_H_
