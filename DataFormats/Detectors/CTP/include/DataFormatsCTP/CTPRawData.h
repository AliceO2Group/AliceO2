#ifndef ALICEO2_CTPRAWDATA_H
#define ALICEO2_CTPRAWDATA_H

#include <bitset>
#include "CommonDataFormat/InteractionRecord.h"
namespace o2
{
namespace ctp
{
static constexpr uint8_t NumberOfClasses = 64;
static constexpr uint8_t NumberOfLMinputs = 16;
static constexpr uint8_t NumberOfL0inputs = 30;
static constexpr uint8_t NumberOfL1inputs = 18;
static constexpr uint8_t NumberOfInputsIR = NumberOfLMinputs + NumberOfL0inputs;
/**
   * @brief The CTPRawData struct
   * These are CTP Digits
   */
struct CTPRawData {

  InteractionRecord ir;
  std::bitset<NumberOfLMinputs> inputsMaskLM;
  std::bitset<NumberOfL0inputs> inputsMaskL0;
  std::bitset<NumberOfL1inputs> inputsMaskL1;
  std::bitset<NumberOfClasses> triggerClassMask;

  CTPRawData() = default;
  CTPRawData(const CTPRawData& src) = default;
  InteractionRecord getInteractionRecord()
  {
    return ir;
  }
  void test();
  ClassDefNV(CTPRawData, 1);
};
struct CTPIRdigit {
  uint16_t bcid;
  std::bitset<NumberOfInputsIR> inputMaskIR;
};
} // namespace ctp
} // namespace o2
// namespace o2
#endif
