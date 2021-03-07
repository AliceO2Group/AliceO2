#include "CTPRaw/CTPRawReader.h"
#include <iostream>

using namespace o2::ctp;

ClassImp(CTPRawReader);

void o2::ctp::CTPRawReader::init(const std::string& cfg = "rawCTPConfig.cfg")
{
  mReader = std::make_unique<RawFileReader>(cfg);
  uint32_t errCheck = 0xffffffff;
  errCheck ^= 0x1 << RawFileReader::ErrNoSuperPageForTF;
  mReader->init();
}

int CTPRawReader::readRaw()
{
  if (!mReader)
    init();
  int nLinks = mReader->getNLinks();
  std::cout << "Number of links:" << std::dec << nLinks << std::endl;
}
