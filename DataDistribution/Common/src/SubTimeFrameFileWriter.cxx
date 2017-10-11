// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Common/SubTimeFrameFile.h"
#include "Common/SubTimeFrameFileWriter.h"

#include <iomanip>

#include <gsl/gsl_util>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileWriter
////////////////////////////////////////////////////////////////////////////////

SubTimeFrameFileWriter::SubTimeFrameFileWriter(const boost::filesystem::path& pFileName, bool pWriteInfo)
  : mWriteInfo(pWriteInfo)
{
  using ios = std::ios_base;

  // allocate and set the larger stream buffer
  mFileBuf = std::make_unique<char[]>(sBuffSize);
  mFile.rdbuf()->pubsetbuf(mFileBuf.get(), sBuffSize);
  mFile.clear();
  mFile.exceptions(std::fstream::failbit | std::fstream::badbit);
  // allocate and set the larger stream buffer (info file)
  if (mWriteInfo) {
    mInfoFileBuf = std::make_unique<char[]>(sBuffSize);
    mInfoFile.rdbuf()->pubsetbuf(mInfoFileBuf.get(), sBuffSize);
    mInfoFile.clear();
    mInfoFile.exceptions(std::fstream::failbit | std::fstream::badbit);
  }

  try {
    mFile.open(pFileName.string(), ios::binary | ios::trunc | ios::out | ios::ate);

    if (mWriteInfo) {
      auto lInfoFileName = pFileName.string();
      lInfoFileName += ".info";

      mInfoFile.open(lInfoFileName, ios::trunc | ios::out);

      mInfoFile << "TF_ID ";
      mInfoFile << "TF_OFFSET ";
      mInfoFile << "TF_SIZE ";
      mInfoFile << "ORIGIN ";
      mInfoFile << "DESCRIPTION ";
      mInfoFile << "SUBSPECIFICATION ";
      mInfoFile << "HEADER_OFFSET ";
      mInfoFile << "HEADER_SIZE ";
      mInfoFile << "DATA_OFFSET ";
      mInfoFile << "DATA_SIZE" << '\n';
    }
  } catch (std::ifstream::failure& eOpenErr) {
    LOG(ERROR) << "Failed to open/create TF file for writing. Error: " << eOpenErr.what();
    throw eOpenErr;
  }
}

SubTimeFrameFileWriter::~SubTimeFrameFileWriter()
{
  try {
    mFile.close();
    if (mWriteInfo) {
      mInfoFile.close();
    }
  } catch (std::ifstream::failure& eCloseErr) {
    LOG(ERROR) << "Closing TF file failed. Error: " << eCloseErr.what();
  } catch (...) {
    LOG(ERROR) << "Closing TF file failed.";
  }
}

void SubTimeFrameFileWriter::visit(const SubTimeFrame& pStf)
{
  assert(mStfData.empty() && mStfSize == 0);
  assert(mStfDataIndex.empty());

  // Write data in lexicographical order of DataIdentifier + subSpecification
  // for easier binary comparison
  std::vector<EquipmentIdentifier> lEquipIds = pStf.getEquipmentIdentifiers();
  std::sort(std::begin(lEquipIds), std::end(lEquipIds));

  //  sizes for different equipment identifiers
  std::map<DataIdentifier, std::uint64_t> lDataIdSize;

  for (const auto& lEquip : lEquipIds) {

    const auto& lEquipDataVec = pStf.mData.at(lEquip).at(lEquip.mSubSpecification);

    for (const auto& lData : lEquipDataVec) {
      // NOTE: get only pointers to <hdr, data> struct
      mStfData.emplace_back(&lData);
      // account the size
      const auto lHdrDataSize = lData.mHeader->GetSize() + lData.mData->GetSize();

      // total size
      mStfSize += lHdrDataSize;

      // calculate the size for the index
      lDataIdSize[lEquip] += lHdrDataSize;
    }
  }

  // build the index
  {
    std::uint64_t lCurrOff = 0;
    for (const auto& lId : lEquipIds) {
      const auto lIdSize = lDataIdSize[lId];
      assert(lIdSize > sizeof(DataHeader));
      mStfDataIndex.AddStfElement(lId, lCurrOff, lIdSize);
      lCurrOff += lIdSize;
    }
  }
}

template <
  typename pointer,
  typename std::enable_if<
    std::is_pointer<pointer>::value &&
    (std::is_void<std::remove_pointer_t<pointer>>::value ||
     std::is_standard_layout<std::remove_pointer_t<pointer>>::value)>::type*>
void SubTimeFrameFileWriter::buffered_write(const pointer p, std::streamsize pCount)
{
  using value_type = std::conditional_t<std::is_void<std::remove_pointer_t<pointer>>::value,
                                        char,
                                        std::remove_pointer_t<pointer>>;
  // make sure we're not doing a short write
  assert((pCount % sizeof(value_type) == 0) && "Performing short write?");

  const char* lPtr = reinterpret_cast<const char*>(p);
  // avoid the optimization if the write is large enough
  if (pCount >= sBuffSize) {
    mFile.write(lPtr, pCount);
  } else {
    // split the write to smaller chunks
    while (pCount > 0) {
      const auto lToWrite = std::min(pCount, sChunkSize);
      assert(lToWrite > 0 && lToWrite <= sChunkSize && lToWrite <= pCount);

      mFile.write(lPtr, lToWrite);
      lPtr += lToWrite;
      pCount -= lToWrite;
    }
  }
}

const std::uint64_t SubTimeFrameFileWriter::getSizeInFile() const
{
  return SubTimeFrameFileMeta::getSizeInFile() + mStfDataIndex.getSizeInFile() + mStfSize;
}

std::uint64_t SubTimeFrameFileWriter::write(const SubTimeFrame& pStf)
{
  // cleanup
  auto lCleanup = gsl::finally([this] {
    // make sure headers and chunk pointers don't linger
    mStfData.clear();
    mStfDataIndex.clear();
    mStfSize = 0;
  });

  if (!mFile.good()) {
    LOG(WARNING) << "Error while writing a TF to file. (bad stream state)";
    return std::uint64_t(0);
  }

  // collect all stf blocks
  pStf.accept(*this);

  // get file position
  const std::uint64_t lPrevSize = size();
  const std::uint64_t lStfSizeInFile = getSizeInFile();
  std::uint64_t lDataOffset = 0;

  SubTimeFrameFileMeta lStfFileMeta(lStfSizeInFile);

  try {
    // Write DataHeader + SubTimeFrameFileMeta
    mFile << lStfFileMeta;

    // Write DataHeader + SubTimeFrameFileDataIndex
    mFile << mStfDataIndex;

    lDataOffset = size(); // save for the info file

    for (const auto& lStfData : mStfData) {
      buffered_write(lStfData->mHeader->GetData(), lStfData->mHeader->GetSize());
      buffered_write(lStfData->mData->GetData(), lStfData->mData->GetSize());
    }

    // flush the buffer and check the state
    mFile.flush();

  } catch (const std::ios_base::failure& eFailExc) {
    LOG(ERROR) << "Writing to file failed. Error: " << eFailExc.what();
    return std::uint64_t(0);
  }

  assert((size() - lPrevSize == lStfSizeInFile) && "Calculated and written sizes differ");

  // sidecar
  if (mWriteInfo) {
    try {
      const auto l1StfId = pStf.header().mId;
      const auto l2StfFileOff = lPrevSize;
      const auto l3StfFileSize = lStfSizeInFile;

      for (const auto& lStfData : mStfData) {
        DataHeader lDH;
        std::memcpy(&lDH, lStfData->mHeader->GetData(), sizeof(DataHeader));

        const auto& l4DataOrigin = lDH.dataOrigin;
        const auto& l5DataDescription = lDH.dataDescription;
        const auto l6SubSpec = lDH.subSpecification;

        const auto l7HdrOff = lDataOffset;
        lDataOffset += lStfData->mHeader->GetSize();
        const auto l8HdrSize = lStfData->mHeader->GetSize();
        const auto l9DataOff = lDataOffset;
        lDataOffset += lStfData->mData->GetSize();
        const auto l10DataSize = lStfData->mData->GetSize();

        mInfoFile << l1StfId << sSidecarFieldSep;
        mInfoFile << l2StfFileOff << sSidecarFieldSep;
        mInfoFile << l3StfFileSize << sSidecarFieldSep;
        mInfoFile << l4DataOrigin.str << sSidecarFieldSep;
        mInfoFile << l5DataDescription.str << sSidecarFieldSep;
        mInfoFile << l6SubSpec << sSidecarFieldSep;
        mInfoFile << l7HdrOff << sSidecarFieldSep;
        mInfoFile << l8HdrSize << sSidecarFieldSep;
        mInfoFile << l9DataOff << sSidecarFieldSep;
        mInfoFile << l10DataSize << sSidecarRecordSep;
      }
      mInfoFile.flush();
    } catch (const std::ios_base::failure& eFailExc) {
      LOG(ERROR) << "Writing to file failed. Error: " << eFailExc.what();
      return std::uint64_t(0);
    }
  }

  return std::uint64_t(size() - lPrevSize);
}
}
} /* o2::DataDistribution */
