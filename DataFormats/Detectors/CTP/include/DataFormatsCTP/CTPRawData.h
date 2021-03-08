#ifndef ALICEO2_CTPRAWDATA_H
#define ALICEO2_CTPRAWDATA_H
#include InteractionRecord.h
namespace o2
{
 namespace ctp
 {
  struct CTPRawData {
  static constexpr uint8_t NumberOfClasses = 64;
  static constexpr uint8_t NumberOfLMinputs = 16;
  static constexpr uint8_t NumberOfL0inputs = 30;
  static constexpr uint8_t NumberOfL1inputs = 18;
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
  std::bitset<NumberOfLMinputs> getInputsMaskLM()
  {
      return std::bitset<NumberOfLMinputs>;
  }
  ClassDefNV(CTPRawData,1);
  };
 } // namespace ctp
}
 // namespace o2
