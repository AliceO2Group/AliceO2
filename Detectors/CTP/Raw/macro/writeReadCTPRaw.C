#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CTPRaw/CTPRawWriter.h"
#include "CTPRaw/CTPRawReader.h"

#endif
using namespace o2::ctp;
void writeReadCTPRaw()
{
  std::cout << "Writing raw" << std::endl;
  CTPRawWriter rawwriter;
  rawwriter.init();
  rawwriter.createRawFromIRs();
  std::cout << "Reading raw" << std::endl;
  CTPRawReader rawreader;
  rawreader.init("rawCTPConfig.cfg");
  rawreader.readRaw();
}
