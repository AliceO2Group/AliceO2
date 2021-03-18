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
#include "Framework/Logger.h"

namespace o2::mch::raw
{

PayloadPaginator::PayloadPaginator(o2::raw::RawFileWriter& fw,
                                   const std::string outputFileName,
                                   bool filePerLink,
                                   Solar2FeeLinkMapper solar2feelink,
                                   bool userLogic,
                                   bool chargeSumMode) : mRawFileWriter(fw),
                                                         mOutputFileName{outputFileName},
                                                         mFilePerLink(filePerLink),
                                                         mSolar2FeeLink{solar2feelink},
                                                         mExtraFeeIdMask{chargeSumMode ? static_cast<uint16_t>(0x100) : static_cast<uint16_t>(0)}
{
  if (userLogic) {
    mSolar2FeeLink = [solar2feelink](uint16_t solarId) -> std::optional<FeeLinkId> {
      static auto s2f = solar2feelink;
      auto f = s2f(solarId);
      if (!f.has_value()) {
        return std::nullopt;
      }
      return FeeLinkId(f->feeId(), 15);
    };
  }
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
      mRawFileWriter.registerLink(feeId, cru, feelink.linkId(), endpoint,
                                  mFilePerLink ? fmt::format("{:s}_feeid{:d}.raw", mOutputFileName, feeId) : fmt::format("{:s}.raw", mOutputFileName));
      mFeeLinkIds.insert(feelink);
    }
    mRawFileWriter.addData(feeId, cru, feelink.linkId(), endpoint,
                           {h.bc, h.orbit},
                           gsl::span<char>(const_cast<char*>(reinterpret_cast<const char*>(&b.payload[0])),
                                           b.payload.size()));
  }
}

std::vector<std::byte> paginate(gsl::span<const std::byte> buffer, bool userLogic,
                                bool chargeSumMode, const std::string& tmpfilename)
{
  fair::Logger::SetConsoleSeverity("nolog");
  o2::raw::RawFileWriter fw;

  fw.setVerbosity(1);
  fw.setDontFillEmptyHBF(true);

  Solar2FeeLinkMapper solar2feelink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();

  {
    PayloadPaginator p(fw, tmpfilename, false, solar2feelink, userLogic, chargeSumMode);
    p(buffer);
    fw.close();
  }

  std::ifstream in(tmpfilename, std::ifstream::binary);
  // get length of file:
  in.seekg(0, in.end);
  int length = in.tellg();
  in.seekg(0, in.beg);
  std::vector<std::byte> pages(length);

  // read data as a block:
  in.read(reinterpret_cast<char*>(&pages[0]), length);

  return pages;
}
} // namespace o2::mch::raw
