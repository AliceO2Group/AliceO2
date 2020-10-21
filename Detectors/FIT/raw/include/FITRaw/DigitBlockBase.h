// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file DigitBlockBase.h base class for processing RAW data into Digits
//
// Artur.Furs
// afurs@cern.ch

#ifndef ALICEO2_FIT_DIGITBLOCKBASE_H_
#define ALICEO2_FIT_DIGITBLOCKBASE_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <Rtypes.h>
#include <CommonDataFormat/InteractionRecord.h>

#include <gsl/span>
namespace o2
{
namespace fit
{
template <class DigitBlock>
class DigitBlockBase //:public DigitBlock
{
 public:
  DigitBlockBase(o2::InteractionRecord intRec)
  { /*static_cast<DigitBlock*>(this)->setIntRec(intRec);*/
  }
  DigitBlockBase() = default;
  DigitBlockBase(const DigitBlockBase& other) = default;
  ~DigitBlockBase() = default;
  template <class DataBlockType>
  void process(DataBlockType& dataBlock, int linkID)
  {
    static_cast<DigitBlock*>(this)->processDigits(dataBlock, linkID);
  }
  template <class... DigitType>
  void pop(std::vector<DigitType>&... vecDigits)
  {
    static_cast<DigitBlock*>(this)->getDigits(vecDigits...);
  }
};

} // namespace fit
} // namespace o2
#endif
