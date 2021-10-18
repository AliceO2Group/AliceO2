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
#ifndef ALICEO2_RAWERRORHANDLER_H
#define ALICEO2_RAWERRORHANDLER_H

#include <array>
#include <string>
#include <EMCALReconstruction/AltroDecoder.h>

namespace o2
{

namespace emcal
{

class MappingHandler;

class RawErrorCodeHandler
{
 public:
  RawErrorCodeHandler();
  ~RawErrorCodeHandler() = default;
  int getGlobalCodeForAltroError(const AltroDecoderError::ErrorType_t& err) const;
  int getGlobalCodeForMinorAltroError(const MinorAltroDecodingError::ErrorType_t& err) const;
  int getGlobalCodeForGainError(int errorcode) const;
  constexpr int getNumberOfErrorCodes() { return mAltroIndices.size() + mMinorAltroIndices.size() + mGainErrorIndices.size(); }

 private:
  std::array<int, AltroDecoderError::getNumberOfErrorTypes()> mAltroIndices;
  std::array<int, MinorAltroDecodingError::getNumberOfErrorTypes()> mMinorAltroIndices;
  std::array<int, 2> mGainErrorIndices;
};

class RawErrorHandler
{
 public:
  RawErrorHandler() = default;
  RawErrorHandler(int nclasses);
  ~RawErrorHandler() = default;

  void setNumberOfErrorClasses(int nclasses);
  void setMapper(MappingHandler* mapper);
  void define(int errorcode, const char* messasge);
  bool book(int errorcode, int ddl, int hwaddress);
  void clear();
  void stat();
  void reset();

 private:
  class DDLErrorContainer
  {
   public:
    DDLErrorContainer();
    ~DDLErrorContainer() = default;
    void reset();
    bool book(int hwaddress)
    {
      bool hasBefore = mHwAddress[hwaddress] == 0;
      mHwAddress[hwaddress]++;
      return hasBefore;
    }
    const std::array<int, 3279>& getErrorCounters() const { return mHwAddress; }

   private:
    std::array<int, 3279> mHwAddress;
  };

  class RawErrorClass
  {
   public:
    RawErrorClass() = default;
    RawErrorClass(const char* basemessage);
    ~RawErrorClass() = default;
    void setMapper(MappingHandler* mapper) { mMapper = mapper; }
    void setMessage(const char* message) { mDescription = message; }
    bool book(unsigned int ddl, int hwaddress);
    void reset();
    void stats();

   private:
    MappingHandler* mMapper = nullptr;
    std::array<DDLErrorContainer, 40> mDDLHandlers;
    std::string mDescription;
  };

  MappingHandler* mMapper = nullptr;
  std::vector<RawErrorClass> mErrorClasses;
  std::vector<bool> mInitialized;
};

} // namespace emcal

} // namespace o2

#endif