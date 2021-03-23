//
// Created by rl on 3/17/21.
//

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
  void setInteractionRecord(const o2::InteractionTimeRecord& src) { mIntRecord = src; }
  void process(std::vector<o2::ctp::CTPDigit>& digits);
  void flush(std::vector<o2::ctp::CTPDigit>& digits);
  void storeBC(const o2::ctp::CTPDigit& cashe, std::vector<o2::ctp::CTPDigit>& digits);

 private:
  Int_t mEventID;
  o2::InteractionRecord firstBCinDeque{};
  std::deque<CTPDigit> mCache;
  o2::InteractionTimeRecord mIntRecord;
  ClassDefNV(Digitizer, 1);
};
} // namespace ctp
} // namespace o2
#endif //ALICEO2_CTP_DIGITIZER_H
