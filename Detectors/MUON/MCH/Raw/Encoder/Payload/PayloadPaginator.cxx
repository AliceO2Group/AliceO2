// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoderPayload/PayloadPaginator.h"
#include "DetectorsRaw/RawFileWriter.h"
#include "MCHRawEncoderPayload/DataBlock.h"

namespace o2::mch::raw
{

PayloadPaginator::PayloadPaginator(o2::raw::RawFileWriter& fw,
                                   const std::string outputFileName,
                                   Solar2FeeLinkMapper solar2feelink,
                                   bool userLogic) : mRawFileWriter(fw),
                                                     mOutputFileName{outputFileName},
                                                     mSolar2FeeLink{solar2feelink},
                                                     mExtraFeeIdMask{userLogic ? static_cast<uint16_t>(0x100) : static_cast<uint16_t>(0)}
{
}

void PayloadPaginator::operator()(gsl::span<const std::byte> buffer)
{
  std::set<DataBlockRef> dataBlockRefs;
  forEachDataBlockRef(
    buffer, [&](const DataBlockRef& ref) {
      dataBlockRefs.insert(ref);
    });

  for (auto r : dataBlockRefs) {
    auto& b = r.block;
    auto& h = b.header;
    auto feelink = mSolar2FeeLink(r.block.header.solarId).value();
    int endpoint = feelink.feeId() % 2;
    int cru = (feelink.feeId() - endpoint) / 2;
    auto feeId = feelink.feeId() | mExtraFeeIdMask;
    if (mFeeLinkIds.find(feelink) == mFeeLinkIds.end()) {
      mRawFileWriter.registerLink(feeId, cru, feelink.linkId(), endpoint, mOutputFileName);
      mFeeLinkIds.insert(feelink);
    }
    mRawFileWriter.addData(feeId, cru, feelink.linkId(), endpoint,
                           {h.bc, h.orbit},
                           gsl::span<char>(const_cast<char*>(reinterpret_cast<const char*>(&b.payload[0])),
                                           b.payload.size()));
  }
}

} // namespace o2::mch::raw
