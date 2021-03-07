#ifndef ALICEO2_CTP_CTPRAWWRITER_H
#define ALICEO2_CTP_CTPRAWWRITER_H

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <TRandom.h>
#include "Steer/InteractionSampler.h"
//#include "DetectorsRaw/HBFUtils.h"
//#include "DetectorsRaw/RDHUtils.h"
#include "DetectorsRaw/RawFileWriter.h"
//#include "DetectorsRaw/SimpleRawReader.h"
//#include "DetectorsRaw/SimpleSTF.h"
//#include "CommonConstants/Triggers.h"
//#include "Framework/Logger.h"
//#include "Framework/InputRecord.h"
//#include "DPLUtils/DPLRawParser.h"
//#include "CommonUtils/StringUtils.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{
namespace ctp
{
constexpr int NCRU = 1;
constexpr int NLINKS = 2;
constexpr uint64_t TriggerInputMask = 0x3fffffffffff;
constexpr double_t TriggerInputMask_d = TriggerInputMask;
constexpr uint64_t TrgclassInputMask = 0xffffffffffffffff;
constexpr double_t TrgclassInputMask_d = 0xffffffffffffffff;

class CTPRawWriter
{
 public:
  CTPRawWriter() = default;
  void init();
  int createRawFromIRs();

 private:
  o2::raw::RawFileWriter mWriter{"CTP"};
  std::string mConfigName = "rawCTPConfig.cfg";
  int mlink0 = 0;
  int mlink1 = 1;
  int mlink2 = 2;
  int mcru = 0;
  ClassDefNV(CTPRawWriter, 1);
};
} // namespace ctp
} // namespace o2
#endif
