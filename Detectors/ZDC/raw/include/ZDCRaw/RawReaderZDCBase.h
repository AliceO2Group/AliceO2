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
//file RawReaderZDCBase.h Base class for RAW data reading
//

#ifndef ALICEO2_FIT_RAWREADERFT0BASE_H_
#define ALICEO2_FIT_RAWREADERFT0BASE_H_
#include <iostream>
#include <vector>
#include <Rtypes.h>
#include "ZDCRaw/RawReaderBase.h"

#include <boost/mpl/inherit.hpp>
#include <boost/mpl/vector.hpp>

#include <CommonDataFormat/InteractionRecord.h>
#include "Headers/RAWDataHeader.h"

#include <gsl/span>

using namespace o2::zdc;
namespace o2
{
namespace zdc
{
// Common raw reader for ZDC
class RawReaderZDCBase : public RawReaderBase
{
 public:
  RawReaderZDCBase() = default;
  ~RawReaderZDCBase() = default;
  //deserialize payload to raw data blocks and proccesss them to digits
  void process(int linkID, gsl::span<const uint8_t> payload)
  {
    if (0 <= linkID && linkID < 16) {
      //PM data processing
      processBinaryData(payload, linkID);
    } else {
      //put here code in case of bad rdh.linkID value
      LOG(INFO) << "WARNING! WRONG LINK ID! " << linkID;
      return;
    }
    //
  }
};
} // namespace zdc
} // namespace o2

#endif
