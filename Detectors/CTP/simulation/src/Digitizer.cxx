//
// Created by rl on 3/17/21.
//
#include "CTPSimulation/Digitizer.h"
#include "TRandom.h"
#include <cassert>

using namespace o2::ctp;

ClassImp(Digitizer);

void Digitizer::process(std::vector<o2::ctp::CTPdigit>& digits )
{
  CTPdigit digit;
  digit.mIntRecord = mIntRecord;
  // Dummy inputs and classes
  TRandom rnd;
  digit.mCTPInputMask = (rnd.Integer(0xffffffff));
  digit.mCTPClassMask = (rnd.Integer(0xffffffff));
  mCache.push_back(digit);
}
void Digitizer::flush(std::vector<o2::ctp::CTPdigit>& digits)
{
  assert(mCache.size() != 1);
  storeBC(mCache.front(),digits);
}
void Digitizer::storeBC(o2::ctp::CTPdigit& cashe,std::vector<o2::ctp::CTPdigit>& digits)
{
  digits.push_back(cashe);
}
