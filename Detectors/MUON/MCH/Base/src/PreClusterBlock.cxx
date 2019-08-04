// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/PreClusterBlock.h"

#include <cassert>

#include <FairMQLogger.h>

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
PreClusterBlock::PreClusterBlock(void* buffer, uint32_t size, bool write)
{
  /// constructor

  // initialization is the same as in the "reset" method
  reset(buffer, size, write);
}

//_________________________________________________________________________________________________
int PreClusterBlock::reset(void* buffer, uint32_t size, bool write)
{
  /// initialize the internal structure in write mode

  assert(buffer != nullptr);

  int status = 0;

  // store buffer
  mBuffer = buffer;

  // store buffer size
  mSize = size;

  // store write mode
  mWriteMode = write;

  // reset
  const uint32_t minSizeOfCluster = SSizeOfUShort + SSizeOfDigit;
  mSize4PreCluster = (size > minSizeOfCluster) ? size - minSizeOfCluster : 0;
  mSize4Digit = (size > SSizeOfDigit) ? size - SSizeOfDigit : 0;
  mLastNDigits = nullptr;
  mPreClusters.clear();

  if (size >= SSizeOfUShort) {

    // store number of precluster and increment buffer
    mNPreClusters = reinterpret_cast<uint16_t*>(mBuffer);
    mBuffer = (reinterpret_cast<uint16_t*>(mBuffer) + 1);
    mCurrentSize = SSizeOfUShort;

    if (mWriteMode) {
      // assign 0 clusters in write mode
      *mNPreClusters = 0;
    } else {
      // read buffer otherwise
      status = readBuffer();
    }

  } else {

    mCurrentSize = mSize + 1;
    mNPreClusters = nullptr;

    if (mWriteMode) {
      LOG(ERROR) << "The buffer is too small to store the data block.";
      status = -ENOBUFS;
    } else {
      LOG(ERROR) << "The buffer is too small to contain the data block.";
      status = -EILSEQ;
    }
  }

  return status;
}

//_________________________________________________________________________________________________
int PreClusterBlock::startPreCluster(const DigitStruct& digit)
{
  /// start a new precluster

  assert(mWriteMode);

  // could move back to constructor (or reset)
  if (mCurrentSize > mSize4PreCluster) {
    LOG(ERROR) << "The buffer is too small to store a new precluster.";
    mLastNDigits = nullptr;
    return -ENOBUFS;
  }

  // use current buffer position to store number of digits
  // and increment buffer
  mLastNDigits = reinterpret_cast<uint16_t*>(mBuffer);
  *mLastNDigits = 1;
  mBuffer = (reinterpret_cast<uint16_t*>(mBuffer) + 1);
  mCurrentSize += SSizeOfUShort;

  // store digit and increment buffer
  auto lastDigit = reinterpret_cast<DigitStruct*>(mBuffer);
  *lastDigit = digit;
  mBuffer = (reinterpret_cast<DigitStruct*>(mBuffer) + 1);
  mCurrentSize += SSizeOfDigit;

  // increment number of pre-clusters
  (*mNPreClusters) += 1;

  // insert in list
  PreClusterStruct preCluster = {*mLastNDigits, lastDigit};
  mPreClusters.push_back(preCluster);

  return 0;
}

//_________________________________________________________________________________________________
int PreClusterBlock::addDigit(const DigitStruct& digit)
{
  /// add a new digit to the current precluster

  assert(mWriteMode);
  if (mCurrentSize > mSize4Digit) {
    LOG(ERROR) << "The buffer is too small to store a new digit.";
    return -ENOBUFS;
  }

  if (!mLastNDigits) {
    LOG(ERROR) << "No precluster to attach the new digit to.";
    return -EPERM;
  }

  // increment number of digits
  *mLastNDigits += 1;

  // assign digit to the buffer and increment buffer
  *(reinterpret_cast<DigitStruct*>(mBuffer)) = digit;
  mBuffer = (reinterpret_cast<DigitStruct*>(mBuffer) + 1);
  mCurrentSize += SSizeOfDigit;

  // increment number of digits in the stored cluster
  if (!mPreClusters.empty()) {
    ++mPreClusters.back().nDigits;
  }

  return 0;
}

//_________________________________________________________________________________________________
int PreClusterBlock::readBuffer()
{
  /// read the buffer to fill the internal structure
  /// fNPreclus,fNdig,dig1, dig2 ... dign, fNdig, dig1 ... digN, ... last_precluster

  assert(!mWriteMode);

  // make sure mNPreClusters was assigned
  if (mNPreClusters == nullptr) {
    return -EILSEQ;
  }

  // initialize status
  int status = 0;

  // rolling pre-cluster
  PreClusterStruct preCluster = {};

  // loop over
  for (int i = 0; i < *mNPreClusters; ++i) {

    // store number of digits from buffer and increment
    uint16_t* nDigits = nullptr;
    if (mCurrentSize < mSize) {
      nDigits = reinterpret_cast<uint16_t*>(mBuffer);
      mBuffer = (reinterpret_cast<uint16_t*>(mBuffer) + 1);
      mCurrentSize += SSizeOfUShort;
    } else {
      LOG(ERROR) << "Cannot read the expected number of preclusters.";
      status = -EILSEQ;
      break;
    }

    // read the digits
    if (nDigits && (*nDigits) > 0 && mCurrentSize + (*nDigits) * SSizeOfDigit <= mSize) {

      auto digit = reinterpret_cast<DigitStruct*>(mBuffer);
      mBuffer = (reinterpret_cast<DigitStruct*>(mBuffer) + (*nDigits));
      mCurrentSize += (*nDigits) * SSizeOfDigit;

      // store
      preCluster.nDigits = *nDigits;
      preCluster.digits = digit;
      mPreClusters.push_back(preCluster);

    } else {

      if (*nDigits == 0) {
        LOG(ERROR) << "The precluster cannot contain 0 digit.";
      } else {
        LOG(ERROR) << "Cannot read the expected number of digits.";
      }

      status = -EILSEQ;
      break;
    }
  }

  // sanity check on read size
  if (mCurrentSize != mSize && status >= 0) {
    LOG(ERROR) << "The number of bytes read differs from the buffer size";
    status = -EILSEQ;
  }

  // reset in case of negative status
  if (status < 0) {
    mCurrentSize = mSize + 1;
    mNPreClusters = nullptr;
    mPreClusters.clear();
  }

  return status;
}

//_________________________________________________________________
std::ostream& operator<<(std::ostream& stream, const PreClusterStruct& cluster)
{
  stream << "{nDigits= " << cluster.nDigits;
  for (int i = 0; i < cluster.nDigits; ++i) {
    stream << ", digit[" << i << "]= " << cluster.digits[i];
  }
  stream << "}";

  return stream;
}

//_________________________________________________________________
std::ostream& operator<<(std::ostream& stream, const PreClusterBlock& clusterBlock)
{
  stream << "{fNClusters= " << clusterBlock.getNPreClusters() << std::endl;
  const auto& clusters(clusterBlock.getPreClusters());
  int i(0);
  for (const auto& cluster : clusters) {
    stream << "  cluster[" << i++ << "]= " << cluster << std::endl;
  }
  stream << "}";

  return stream;
}

} // namespace mch
} // namespace o2
