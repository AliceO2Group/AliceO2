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
#include "Common/SubTimeFrameFileReader.h"

#include <gsl/gsl_util>

namespace o2
{
namespace DataDistribution
{

////////////////////////////////////////////////////////////////////////////////
/// SubTimeFrameFileReader
////////////////////////////////////////////////////////////////////////////////

SubTimeFrameFileReader::SubTimeFrameFileReader(boost::filesystem::path& pFileName)
{
  using ios = std::ios_base;

  try {
    mFile.open(pFileName.string(), ios::binary | ios::in);
    mFile.exceptions(std::fstream::failbit | std::fstream::badbit);

    // get the file size
    mFile.seekg(0, std::ios_base::end);
    mFileSize = mFile.tellg();
    mFile.seekg(0, std::ios_base::beg);

  } catch (std::ifstream::failure& eOpenErr) {
    LOG(ERROR) << "Failed to open TF file for reading. Error: " << eOpenErr.what();
  }
}

SubTimeFrameFileReader::~SubTimeFrameFileReader()
{
  try {
    if (mFile.is_open())
      mFile.close();
  } catch (std::ifstream::failure& eCloseErr) {
    LOG(ERROR) << "Closing TF file failed. Error: " << eCloseErr.what();
  } catch (...) {
    LOG(ERROR) << "Closing TF file failed.";
  }
}

void SubTimeFrameFileReader::visit(SubTimeFrame& pStf)
{
  for (auto& lStfDataPair : mStfData) {
    pStf.addStfData(std::move(lStfDataPair));
  }
}

std::int64_t SubTimeFrameFileReader::getHeaderStackSize() // throws ios_base::failure
{
  std::int64_t lHdrStackSize = 0;

  const auto lStartPos = mFile.tellg();

  DataHeader lBaseHdr;

  do {
    buffered_read(&lBaseHdr, sizeof(BaseHeader));
    if (nullptr == BaseHeader::get(reinterpret_cast<o2::byte*>(&lBaseHdr))) {
      // error: expected a header here
      return -1;
    } else {
      lHdrStackSize += lBaseHdr.headerSize;
    }

    // skip the rest of the current header
    if (lBaseHdr.headerSize > sizeof(BaseHeader)) {
      mFile.ignore(lBaseHdr.headerSize - sizeof(BaseHeader));
    } else {
      // error: invalid header size value
      return -1;
    }
  } while (lBaseHdr.next() != nullptr);

  // we should not eof here,
  if (mFile.eof())
    return -1;

  // rewind the file to the start of Header stack
  mFile.seekg(lStartPos);

  return lHdrStackSize;
}

bool SubTimeFrameFileReader::read(SubTimeFrame& pStf, std::uint64_t pStfId, FairMQChannel& pDstChan)
{
  // If mFile is good, we're positioned to read a TF
  if (!mFile || mFile.eof()) {
    return false;
  }

  if (!mFile.good()) {
    LOG(WARNING) << "Error while reading a TF from file. (bad stream state)";
    return false;
  }

  // cleanup
  auto lCleanup = gsl::finally([this] {
    // make sure headers and chunk pointers don't linger
    mStfData.clear();
  });

  // record current position
  const auto lTfStartPosition = this->position();

  DataHeader lStfMetaDataHdr;
  SubTimeFrameFileMeta lStfFileMeta;

  try {
    // Write DataHeader + SubTimeFrameFileMeta
    buffered_read(&lStfMetaDataHdr, sizeof(DataHeader));
    buffered_read(&lStfFileMeta, sizeof(SubTimeFrameFileMeta));

  } catch (const std::ios_base::failure& eFailExc) {
    LOG(ERROR) << "Reading from file failed. Error: " << eFailExc.what();
    return false;
  }

  // verify we're actually reading the correct data in
  if (!(SubTimeFrameFileMeta::getDataHeader() == lStfMetaDataHdr)) {
    LOG(WARNING) << "Reading bad data: SubTimeFrame META header";
    mFile.close();
    return false;
  }

  // prepare to read the TF data
  const auto lStfSizeInFile = lStfFileMeta.mStfSizeInFile;
  if (lStfSizeInFile == (sizeof(DataHeader) + sizeof(SubTimeFrameFileMeta))) {
    LOG(WARNING) << "Reading an empty TF from file. Only meta information present";
    return false;
  }

  const auto lStfDataSize = lStfSizeInFile - (sizeof(DataHeader) + sizeof(SubTimeFrameFileMeta));

  // check there's enough data in the file
  if ((lTfStartPosition + lStfSizeInFile) > this->size()) {
    LOG(WARNING) << "Not enough data in file for this TF. Required: " << lStfSizeInFile
                 << ", available: " << (this->size() - lTfStartPosition);
    mFile.close();
    return false;
  }

  // read all data blocks and headers
  assert(mStfData.empty());
  try {

    std::int64_t lLeftToRead = lStfDataSize;

    // read <hdrStack + data> pairs
    while (lLeftToRead > 0) {

      // read the header stack
      const std::int64_t lHdrSize = getHeaderStackSize();
      if (lHdrSize < sizeof(DataHeader)) {
        // error while checking headers
        LOG(WARNING) << "Reading bad data: Header stack cannot be parsed";
        mFile.close();
        return false;
      }
      // allocate and read the Headers
      auto lHdrStackMsg = pDstChan.NewMessage(lHdrSize);
      if (!lHdrStackMsg) {
        LOG(WARNING) << "Out of memory: header message, allocation size: " << lHdrSize;
        mFile.close();
        return false;
      }
      buffered_read(lHdrStackMsg->GetData(), lHdrSize);

      // read the data
      DataHeader lDataHeader;
      std::memcpy(&lDataHeader, lHdrStackMsg->GetData(), sizeof(DataHeader));
      const std::uint64_t lDataSize = lDataHeader.payloadSize;

      auto lDataMsg = pDstChan.NewMessage(lDataSize);
      if (!lDataMsg) {
        LOG(WARNING) << "Out of memory: data message, allocation size: " << lDataSize;
        mFile.close();
        return false;
      }
      buffered_read(lDataMsg->GetData(), lDataSize);

      mStfData.emplace_back(
        SubTimeFrame::StfData{
          std::move(lHdrStackMsg),
          std::move(lDataMsg) });

      // update the counter
      lLeftToRead -= (lHdrSize + lDataSize);
    }

    if (lLeftToRead < 0) {
      LOG(DEBUG) << "Read more data than it is indicated in the META header!";
    }

  } catch (const std::ios_base::failure& eFailExc) {
    LOG(ERROR) << "Reading from file failed. Error: " << eFailExc.what();
    return false;
  }

  // build the SubtimeFrame
  pStf.accept(*this);

  LOG(INFO) << "FileReader: read TF size: " << lStfFileMeta.mStfSizeInFile
            << ", created on " << lStfFileMeta.getTimeString()
            << " (timestamp: " << lStfFileMeta.mWriteTimeMs << ")";

  return true;
}
}
} /* o2::DataDistribution */
