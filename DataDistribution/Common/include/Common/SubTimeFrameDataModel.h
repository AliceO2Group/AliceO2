// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_SUBTIMEFRAME_DATAMODEL_H_
#define ALICEO2_SUBTIMEFRAME_DATAMODEL_H_

#include "Common/Utilities.h"
#include "Common/DataModelUtils.h"
#include "Common/ReadoutDataModel.h"

#include <O2Device/O2Device.h>
#include <Headers/DataHeader.h>

#include <vector>
#include <map>
#include <unordered_set>
#include <stdexcept>

#include <functional>

namespace o2
{
namespace DataDistribution
{

using namespace o2::Base;
using namespace o2::header;

static constexpr o2::header::DataDescription gDataDescSubTimeFrame{ "DISTSUBTIMEFRAME" };

struct EquipmentIdentifier {
  DataDescription mDataDescription;
  DataOrigin mDataOrigin;
  DataHeader::SubSpecificationType mSubSpecification; /* uint64_t */

  EquipmentIdentifier() = delete;

  EquipmentIdentifier(const DataDescription& pDataDesc, const DataOrigin& pDataOrig, const DataHeader::SubSpecificationType& pSubSpec) noexcept
    : mDataDescription(pDataDesc),
      mDataOrigin(pDataOrig),
      mSubSpecification(pSubSpec)
  {
  }

  EquipmentIdentifier(const DataIdentifier& pDataId, const DataHeader::SubSpecificationType& pSubSpec) noexcept
    : mDataDescription(pDataId.dataDescription),
      mDataOrigin(pDataId.dataOrigin),
      mSubSpecification(pSubSpec)
  {
  }

  EquipmentIdentifier(const EquipmentIdentifier& pEid) noexcept
    : mDataDescription(pEid.mDataDescription),
      mDataOrigin(pEid.mDataOrigin),
      mSubSpecification(pEid.mSubSpecification)
  {
  }

  EquipmentIdentifier(const o2::header::DataHeader& pDh) noexcept
    : mDataDescription(pDh.dataDescription),
      mDataOrigin(pDh.dataOrigin),
      mSubSpecification(pDh.subSpecification)
  {
  }

  operator DataIdentifier() const noexcept
  {
    return DataIdentifier(mDataDescription, mDataOrigin);
  }

  bool operator<(const EquipmentIdentifier& other) const noexcept
  {
    if (mDataDescription < other.mDataDescription)
      return true;
    else if (mDataDescription == other.mDataDescription &&
             mSubSpecification < other.mSubSpecification)
      return true;
    else if (mDataDescription == other.mDataDescription &&
             mSubSpecification == other.mSubSpecification &&
             mSubSpecification < other.mSubSpecification)
      return true;
    else
      return false;
  }

  const std::string info() const
  {
    return std::string("DataDescription: ") + std::string(mDataDescription.str) +
           std::string(" DataOrigin: ") + std::string(mDataOrigin.str) +
           std::string(" SubSpecification: ") + std::to_string(mSubSpecification);
  }
};

struct HBFrameHeader : public BaseHeader {

  // Required to do the lookup
  static const o2::header::HeaderType sHeaderType;
  static const uint32_t sVersion = 1;

  uint32_t mHBFrameId;

  HBFrameHeader(uint32_t pId)
    : BaseHeader(sizeof(HBFrameHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion),
      mHBFrameId(pId)
  {
  }

  HBFrameHeader()
    : HBFrameHeader(0)
  {
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Visitor friends
////////////////////////////////////////////////////////////////////////////////
#define DECLARE_STF_FRIENDS                    \
  friend class SubTimeFrameReadoutBuilder;     \
  friend class InterleavedHdrDataSerializer;   \
  friend class InterleavedHdrDataDeserializer; \
  friend class DataIdentifierSplitter;         \
  friend class SubTimeFrameFileWriter;         \
  friend class SubTimeFrameFileReader;         \
  friend class StfDplAdapter;

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrame
////////////////////////////////////////////////////////////////////////////////
using TimeFrameIdType = std::uint64_t;
using SubTimeFrameIdType = TimeFrameIdType;
static constexpr TimeFrameIdType sInvalidTimeFrameId = TimeFrameIdType(-1);

class SubTimeFrame : public IDataModelObject
{
  DECLARE_STF_FRIENDS

  struct StfData {
    std::unique_ptr<FairMQMessage> mHeader;
    std::unique_ptr<FairMQMessage> mData;

    DataHeader getDataHeader()
    {
      DataHeader lDataHdr;
      std::memcpy(&lDataHdr, mHeader->GetData(), sizeof(DataHeader));
      return lDataHdr;
    }
  };

 public:
  SubTimeFrame(TimeFrameIdType pStfId);
  //SubTimeFrame() = default;
  ~SubTimeFrame() = default;
  // no copy
  SubTimeFrame(const SubTimeFrame&) = delete;
  SubTimeFrame& operator=(const SubTimeFrame&) = delete;
  // default move
  SubTimeFrame(SubTimeFrame&& a) = default;
  SubTimeFrame& operator=(SubTimeFrame&& a) = default;

  // adopt all data from a
  void mergeStf(std::unique_ptr<SubTimeFrame> pStf);

  std::uint64_t getDataSize() const;

  std::vector<EquipmentIdentifier> getEquipmentIdentifiers() const;

  struct Header {
    TimeFrameIdType mId = sInvalidTimeFrameId;
  };

  const Header& header() const { return mHeader; }

  ///
  /// Fields
  ///
  Header mHeader;

 protected:
  void accept(ISubTimeFrameVisitor& v) override { v.visit(*this); }
  void accept(ISubTimeFrameConstVisitor& v) const override { v.visit(*this); }

 private:
  using StfDataVector = std::vector<StfData>;
  using StfSubSpecMap = std::unordered_map<DataHeader::SubSpecificationType, StfDataVector>;
  using StfDataIdentMap = std::unordered_map<DataIdentifier, StfSubSpecMap>;

  StfDataIdentMap mData;

  ///
  /// helper methods
  inline void addStfData(const DataHeader& pDataHeader, StfData&& pStfData)
  {
    const DataIdentifier lDataId = pDataHeader.getDataIdentifier();

    auto& lDataVector = mData[lDataId][pDataHeader.subSpecification];

    lDataVector.reserve(512);
    lDataVector.emplace_back(std::move(pStfData));
  }

  inline void addStfData(StfData&& pStfData)
  {
    const DataHeader lDataHeader = pStfData.getDataHeader();
    addStfData(lDataHeader, std::move(pStfData));
  }
};
}
} /* o2::DataDistribution */

#endif /* ALICEO2_SUBTIMEFRAME_DATAMODEL_H_ */
