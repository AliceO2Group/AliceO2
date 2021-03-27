// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CTPSimulation/Digitizer.h"
#include "TRandom.h"
#include <cassert>
#include "FairLogger.h"

using namespace o2::ctp;

ClassImp(Digitizer);

void Digitizer::process(CTPDigit digit, std::vector<o2::ctp::CTPDigit>& digits)
{
  //digit.mIntRecord = mIntRecord;
  // Dummy inputs and classes
  //TRandom rnd;
  //digit.mCTPInputMask = (rnd.Integer(0xffffffff));
  //digit.mCTPClassMask = (rnd.Integer(0xffffffff));
  mCache.push_back(digit);
}
void Digitizer::flush(std::vector<o2::ctp::CTPDigit>& digits)
{
  assert(mCache.size() != 1);
  storeBC(mCache.front(), digits);
}
void Digitizer::storeBC(const o2::ctp::CTPDigit& cashe, std::vector<o2::ctp::CTPDigit>& digits)
{
  digits.push_back(cashe);
}
void Digitizer::init()
{
  LOG(INFO) << " @@@ CTP Digitizer::init. Nothing done. " << std::endl;
}
