// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <EMCALBase/Mapper.h>
#include <EMCALReconstruction/Channel.h>
#include <EMCALReconstruction/RawErrorHandler.h>
#include <FairLogger.h>

using namespace o2::emcal;

RawErrorHandler::DDLErrorContainer::DDLErrorContainer() : mHwAddress()
{
  std::fill(mHwAddress.begin(), mHwAddress.end(), 0);
}

void RawErrorHandler::DDLErrorContainer::reset()
{
  std::fill(mHwAddress.begin(), mHwAddress.end(), 0);
}

RawErrorHandler::RawErrorClass::RawErrorClass(const char* message) : mMapper(nullptr), mDDLHandlers(), mDescription(message) {}

bool RawErrorHandler::RawErrorClass::book(unsigned int ddl, int hwaddress)
{
  if (ddl > mDDLHandlers.size()) {
    LOG(ERROR) << "DDL index out of range: " << ddl;
    return false;
  }
  auto bookStatus = mDDLHandlers[ddl].book(hwaddress);
  if (!bookStatus) {
    // Create error message (for the first time)
    if (!mMapper) {
      LOG(ERROR) << "New error channel, but no mapper. Cannot provide more information about channel";
    } else {
      auto fec = mMapper->getFEEForChannelInDDL(ddl, Channel::getFecIndexFromHwAddress(hwaddress), Channel::getBranchIndexFromHwAddress(hwaddress));
      LOG(ERROR) << "DDL " << ddl << ", FEC " << fec << ", address 0x" << std::hex << hwaddress << std::dec << ": " << mDescription;
    }
  }
  return bookStatus;
}

void RawErrorHandler::RawErrorClass::reset()
{
  for (auto& cont : mDDLHandlers)
    cont.reset();
}
void RawErrorHandler::RawErrorClass::stats()
{
  if (!mMapper) {
    LOG(ERROR) << "Mapper not available, cannot print stats";
    return;
  }
  bool initialzied = false;
  auto& description = mDescription;
  auto header = [description, &initialzied]() { if(!initialzied) { LOG(ERROR) << "Summary of raw errors: " << description; initialzied = true; } };
  for (int iddl = 0; iddl < mDDLHandlers.size(); iddl++) {
    auto ddlerrors = mDDLHandlers[iddl].getErrorCounters();
    for (int ihwadd = 0; ihwadd < ddlerrors.size(); ihwadd++) {
      if (ddlerrors[ihwadd]) {
        header();
        auto fec = mMapper->getFEEForChannelInDDL(iddl, Channel::getFecIndexFromHwAddress(ihwadd), Channel::getBranchIndexFromHwAddress(ihwadd));
        LOG(ERROR) << "DDL " << iddl << ", FEC " << fec << ", address 0x" << std::hex << ihwadd << std::dec << ": " << mDescription << ": " << ddlerrors[ihwadd] << " errors";
      }
    }
  }
}

RawErrorHandler::RawErrorHandler(int nclasses) : mMapper(nullptr), mErrorClasses(), mInitialized() { setNumberOfErrorClasses(nclasses); }

void RawErrorHandler::setNumberOfErrorClasses(int nclasses)
{
  std::vector<RawErrorClass> oldclasses;
  std::vector<bool> oldInitStatus;
  if (mErrorClasses.size() > 0 && nclasses > mErrorClasses.size()) {
    oldclasses = mErrorClasses;
    oldInitStatus = mInitialized;
  }
  mErrorClasses.resize(nclasses);
  mInitialized.resize(nclasses);
  int currentclass = 0;
  if (oldclasses.size()) {
    for (currentclass = 0; currentclass < oldclasses.size(); currentclass++) {
      mErrorClasses[currentclass] = oldclasses[currentclass];
      mInitialized[currentclass] = oldInitStatus[currentclass];
    }
  }
  while (currentclass < mErrorClasses.size()) {
    mInitialized[currentclass] = false;
    currentclass++;
  }
}

void RawErrorHandler::setMapper(MappingHandler* mapper)
{
  mMapper = mapper;
  for (int icls = 0; icls < mErrorClasses.size(); icls++) {
    if (mInitialized[icls]) {
      mErrorClasses[icls].setMapper(mMapper);
    }
  }
}

void RawErrorHandler::define(int errorcode, const char* message)
{
  if (errorcode > mErrorClasses.size()) {
    setNumberOfErrorClasses(errorcode);
  }
  if (mInitialized[errorcode]) {
    mErrorClasses[errorcode].setMessage(message);
  } else {
    mErrorClasses[errorcode] = RawErrorClass(message);
  }
  if (mMapper)
    mErrorClasses[errorcode].setMapper(mMapper);
}

bool RawErrorHandler::book(int errorcode, int ddl, int hwaddress)
{
  if (errorcode > mErrorClasses.size()) {
    LOG(ERROR) << "Error code " << errorcode << " out-of-range, max " << mErrorClasses.size();
    return false;
  }
  if (!mInitialized[errorcode]) {
    LOG(ERROR) << "Error code " << errorcode << " not initialized";
    return false;
  }
  return mErrorClasses[errorcode].book(ddl, hwaddress);
}

void RawErrorHandler::stat()
{
  for (int icls = 0; icls < mErrorClasses.size(); icls++) {
    if (mInitialized[icls]) {
      mErrorClasses[icls].stats();
    }
  }
}

void RawErrorHandler::clear()
{
  mErrorClasses.clear();
}

void RawErrorHandler::reset()
{
  for (int icls = 0; icls < mErrorClasses.size(); icls++) {
    if (mInitialized[icls]) {
      mErrorClasses[icls].reset();
    }
  }
}

RawErrorCodeHandler::RawErrorCodeHandler()
{
  int currentcode = 0;
  for (int ierror = 0; ierror < AltroDecoderError::getNumberOfErrorTypes(); ierror++) {
    mAltroIndices[ierror] = currentcode;
    currentcode++;
  }
  for (int ierror = 0; ierror < MinorAltroDecodingError::getNumberOfErrorTypes(); ierror++) {
    mMinorAltroIndices[ierror] = currentcode;
    currentcode++;
  }
  for (int ierror = 0; ierror < 2; ierror++) {
    mGainErrorIndices[ierror] = currentcode;
    currentcode++;
  }
}

int RawErrorCodeHandler::getGlobalCodeForAltroError(const AltroDecoderError::ErrorType_t& err) const
{
  return mAltroIndices[AltroDecoderError::errorTypeToInt(err)];
}

int RawErrorCodeHandler::getGlobalCodeForMinorAltroError(const MinorAltroDecodingError::ErrorType_t& err) const
{
  return mMinorAltroIndices[MinorAltroDecodingError::errorTypeToInt(err)];
}

int RawErrorCodeHandler::getGlobalCodeForGainError(int errorcode) const
{
  return mGainErrorIndices[errorcode];
}
