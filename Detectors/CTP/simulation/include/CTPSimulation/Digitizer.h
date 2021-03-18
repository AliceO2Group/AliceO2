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
namespace  ctp
{
class Digitizer
{
 public:
  Digitizer() = default;
  ~Digitizer()= default;
  void setInteractionRecord(const o2::InteractionTimeRecord& src) { mIntRecord = src; }
  void process(std::vector<o2::ctp::CTPdigit>& digits );
  void flush(std::vector<o2::ctp::CTPdigit>& digits);
  void storeBC(o2::ctp::CTPdigit& cashe, std::vector<o2::ctp::CTPdigit>& digits);
 private:
  Int_t mEventID;
  o2::InteractionRecord firstBCinDeque = 0;
  std::deque<CTPdigit> mCache;
  o2::InteractionTimeRecord mIntRecord;
  ClassDefNV(Digitizer,1);
};
}
}
#endif //ALICEO2_CTP_DIGITIZER_H
