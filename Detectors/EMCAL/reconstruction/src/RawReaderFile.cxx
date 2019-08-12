// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <iostream>
#include <iomanip>

#include "EMCALReconstruction/RawReaderFile.h"

using namespace o2::emcal;

#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))

RawReaderFile::RawReaderFile(const std::string_view filename) : mInputFileName(filename),
                                                                mDataFile(),
                                                                mRawHeader()
{
  init();
}

RawReaderFile::~RawReaderFile()
{
  mDataFile.close();
}

void RawReaderFile::init()
{
  mDataFile.open(mInputFileName, std::ifstream::binary);
  if (!mDataFile.good())
    throw std::runtime_error("Unable to open or access file " + mInputFileName);
  // get length of file in bytes
  mDataFile.seekg(0, mDataFile.end);
  mFileSize = mDataFile.tellg();
  mDataFile.seekg(0, mDataFile.beg);
  // the file is supposed to contain N x 8kB packets. So the number of packets
  // can be determined by the file-size. Ideally, this is not required but the
  // information is derived directly from the header size and payload size.
  // *** to be adapted to header info ***
  mNumData = mFileSize / (8 * 1024);
}

void RawReaderFile::nextPage()
{
  if (mCurrentPosition >= mNumData)
    throw Error(Error::ErrorType_t::PAGE_NOTFOUND);
  auto start = mDataFile.tellg();
  readHeader();
  readPayload();
  mDataFile.seekg(int(start) + mRawHeader.getOffset());
  mCurrentPosition++;
}

void RawReaderFile::readPage(int page)
{
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  if (page >= mNumData)
    throw Error(Error::ErrorType_t::PAGE_NOTFOUND);
  mDataFile.seekg(page * 8192);
  auto start = mDataFile.tellg();
  readHeader();
  readPayload();
  mDataFile.seekg(int(start) + mRawHeader.getOffset());
  mCurrentPosition = page;
}

void RawReaderFile::readHeader()
{
  try {
    // assume the seek is at the header position
    mDataFile >> mRawHeader;
  } catch (...) {
    throw Error(Error::ErrorType_t::HEADER_DECODING);
  }
  mRawHeaderInitialized = true;
}

void RawReaderFile::readPayload()
{
  try {
    // assume the seek is at the payload position
    mRawBuffer.readFromStream(mDataFile, mRawHeader.getBlockSize() - sizeof(o2::emcal::RAWDataHeader));
  } catch (...) {
    throw Error(Error::ErrorType_t::PAYLOAD_DECODING);
  }
  mPayloadInitialized = true;
}

const RAWDataHeader& RawReaderFile::getRawHeader() const
{
  if (!mRawHeaderInitialized)
    throw Error(Error::ErrorType_t::HEADER_INVALID);
  return mRawHeader;
}

const RawBuffer& RawReaderFile::getRawBuffer() const
{
  if (!mPayloadInitialized)
    throw Error(Error::ErrorType_t::PAYLOAD_INVALID);
  return mRawBuffer;
}

void RawReaderFile::readFile(const std::string_view filename)
{
  RawReaderFile reader(filename);
  for (int ipage = 0; ipage < reader.getNumberOfPages(); ipage++) {
    reader.nextPage();
    std::cout << reader.getRawHeader();
  }
}