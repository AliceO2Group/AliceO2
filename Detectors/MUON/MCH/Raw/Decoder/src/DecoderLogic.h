// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_DECODER_DECODER_LOGIC_H
#define O2_MCH_RAW_DECODER_DECODER_LOGIC_H

#include <vector>
#include <array>
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MCHRawElecMap/DsElecId.h"
#include <cstdlib>
#include <iostream>
#include <optional>

namespace o2::mch::raw
{
class DecoderLogic
{
 public:
  DecoderLogic() = delete;
  DecoderLogic(DsElecId dsId, SampaChannelHandler sampaChannelHandler);

  bool headerIsComplete() const;
  bool moreDataAvailable() const;
  bool moreSampleToRead() const;
  bool moreWordsToRead() const;
  uint16_t data10(uint64_t value, size_t index) const;
  uint16_t pop10();
  void addHeaderPart(uint16_t a);
  template <typename CHARGESUM>
  void addSample(uint16_t sample);
  void completeHeader();
  void decrementClusterSize();
  void reset();
  void setClusterSize(uint16_t value);
  void setClusterTime(uint16_t value);
  void setData(uint64_t data);

  bool hasError() const;
  std::string errorMessage() const;

  DsElecId dsId() const;

  friend std::ostream& operator<<(std::ostream&, const DecoderLogic&);

 private:
  std::ostream& debugHeader() const;
  void sendCluster(const SampaCluster& sc) const;

 private:
  // mMasks used to access groups of 10 bits in a 50 bits range
  static constexpr std::array<uint64_t, 5> mMasks = {0x3FF, 0xFFC00, 0x3FF00000, 0xFFC0000000, 0x3FF0000000000};
  DsElecId mDsId{0, 0, 0};
  uint16_t mNof10BitWords{0};
  uint16_t mClusterSize{0};
  uint16_t mClusterTime{0};
  uint64_t mData{0};
  size_t mMaskIndex{mMasks.size()};
  std::vector<uint16_t> mSamples;
  std::vector<uint16_t> mHeaderParts;
  SampaChannelHandler mSampaChannelHandler;
  SampaHeader mSampaHeader;
  std::optional<std::string> mErrorMessage{std::nullopt};
};

} // namespace o2::mch::raw

#endif
