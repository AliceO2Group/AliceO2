#ifndef ALICEO2_CTP_CTPRAWREADER_H
#define ALICEO2_CTP_CTPRAWREADER_H

#include "DetectorsRaw/RawFileReader.h"

namespace o2
{
namespace ctp
{
using namespace o2::raw;
/**
 * @brief The CTPRawReader class
 * Code based on Raw/test/testRawReaderWriter
 */
class CTPRawReader
{
 public:
  CTPRawReader() = default;
  void init(const std::string& cfg);
  int readRaw();

 private:
  std::unique_ptr<RawFileReader> mReader;
  ClassDefNV(CTPRawReader, 1);
};
} // namespace ctp
} // namespace o2
#endif
