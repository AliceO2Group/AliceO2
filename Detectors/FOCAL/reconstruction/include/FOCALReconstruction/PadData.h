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
#ifndef ALICEO2_FOCAL_PADDATA_H
#define ALICEO2_FOCAL_PADDATA_H

#include <array>
#include <exception>
#include <string>
#include <vector>

#include <gsl/span>

#include "Rtypes.h"

#include "FOCALReconstruction/PadWord.h"

namespace o2::focal
{
class ASICData
{
 public:
  class IndexException : public std::exception
  {
   public:
    IndexException() = default;
    IndexException(int index, int maxindex) : std::exception(), mIndex(index), mMaxIndex(maxindex)
    {
      mMessage = "Invalid index " + std::to_string(mIndex) + ", max " + std::to_string(mMaxIndex);
    }
    ~IndexException() noexcept final = default;

    const char* what() const noexcept final
    {
      return mMessage.data();
    }
    int getIndex() const { return mIndex; }
    int getMaxIndex() const { return mMaxIndex; }

   private:
    int mIndex;
    int mMaxIndex;
    std::string mMessage;
  };

  static constexpr int NCHANNELS = 72;
  static constexpr int NHALVES = 2;

  ASICData() = default;
  ASICData(ASICHeader firstheader, ASICHeader secondheader);

  void setFirstHeader(ASICHeader header) { setHeader(header, 0); }
  void setSecondHeader(ASICHeader header) { setHeader(header, 1); }
  void setHeader(ASICHeader header, int index);

  void setChannel(ASICChannel data, int index);
  void setChannels(const gsl::span<const ASICChannel> channels);

  void setFirstCMN(ASICChannel data) { setCMN(data, 0); }
  void setSecondCMN(ASICChannel data) { setCMN(data, 1); }
  void setCMN(ASICChannel data, int index);
  void setCMNs(const gsl::span<const ASICChannel> channels);

  void setFirstCalib(ASICChannel data) { setCalib(data, 0); }
  void setSecondCalib(ASICChannel data) { setCalib(data, 1); }
  void setCalib(ASICChannel data, int index);
  void setCalibs(const gsl::span<const ASICChannel> channels);

  ASICHeader getFirstHeader() const { return getHeader(0); }
  ASICHeader getSecondHeader() const { return getHeader(1); }
  ASICHeader getHeader(int index) const;
  gsl::span<const ASICHeader> getHeaders() const;

  gsl::span<const ASICChannel> getChannels() const;
  ASICChannel getChannel(int index) const;

  ASICChannel getFirstCalib() const { return getCalib(0); }
  ASICChannel getSecondCalib() const { return getCalib(1); }
  ASICChannel getCalib(int index) const;
  gsl::span<const ASICChannel> getCalibs() const;

  ASICChannel getFirstCMN() const { return getCMN(0); }
  ASICChannel getSecondCMN() const { return getCMN(1); }
  ASICChannel getCMN(int index) const;
  gsl::span<const ASICChannel> getCMNs() const;

  void reset();

 private:
  std::array<ASICHeader, NHALVES> mHeaders;
  std::array<ASICChannel, NCHANNELS> mChannels;
  std::array<ASICChannel, NHALVES> mCalibChannels;
  std::array<ASICChannel, NHALVES> mCMNChannels;

  ClassDefNV(ASICData, 1);
};

class ASICContainer
{
 public:
  ASICContainer() = default;
  ~ASICContainer() = default;

  const ASICData& getASIC() const { return mASIC; }
  ASICData& getASIC() { return mASIC; }
  gsl::span<const TriggerWord> getTriggerWords() const;
  void appendTriggerWords(gsl::span<const TriggerWord> triggerwords);
  void appendTriggerWord(TriggerWord triggerword);
  void reset();

 private:
  ASICData mASIC;
  std::vector<TriggerWord> mTriggerData;

  ClassDefNV(ASICContainer, 1);
};

class PadData
{
 public:
  class IndexException : public std::exception
  {
   public:
    IndexException() = default;
    IndexException(int index, int maxindex) : std::exception(), mIndex(index), mMaxIndex(maxindex)
    {
      mMessage = "Invalid index " + std::to_string(mIndex) + ", max " + std::to_string(mMaxIndex);
    }
    ~IndexException() noexcept final = default;
    const char* what() const noexcept final
    {
      return mMessage.data();
    }

    int getIndex() const { return mIndex; }
    int getMaxIndex() const { return mMaxIndex; }

   private:
    int mIndex;
    int mMaxIndex;
    std::string mMessage;
  };
  static constexpr int NASICS = 20;

  PadData() = default;
  ~PadData() = default;

  const ASICContainer& operator[](int index) const { return getDataForASIC(index); }
  ASICContainer& operator[](int index) { return getDataForASIC(index); }

  const ASICContainer& getDataForASIC(int index) const;
  ASICContainer& getDataForASIC(int index);
  void reset();

 private:
  std::array<ASICContainer, NASICS> mASICs;

  ClassDefNV(PadData, 1);
};

} // namespace o2::focal
#endif // ALICEO2_FOCAL_PADDATA_H