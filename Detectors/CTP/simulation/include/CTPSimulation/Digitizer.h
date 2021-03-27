// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_CTP_DIGITIZER_H
#define ALICEO2_CTP_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsCTP/Digits.h"

#include <deque>

namespace o2
{
namespace ctp
{
class Digitizer
{
 public:
  Digitizer() = default;
  ~Digitizer() = default;
  void setInteractionRecord(const o2::InteractionRecord& intrec) { mIntRecord = intrec; }
  void process(CTPDigit digit, std::vector<o2::ctp::CTPDigit>& digits);
  void flush(std::vector<o2::ctp::CTPDigit>& digits);
  void storeBC(const o2::ctp::CTPDigit& cashe, std::vector<o2::ctp::CTPDigit>& digits);
  void init();
 private:
  Int_t mEventID;
  o2::InteractionRecord firstBCinDeque{};
  std::deque<CTPDigit> mCache;
  o2::InteractionRecord mIntRecord;
  ClassDefNV(Digitizer, 1);
};
} // namespace ctp
} // namespace o2
#endif //ALICEO2_CTP_DIGITIZER_H
