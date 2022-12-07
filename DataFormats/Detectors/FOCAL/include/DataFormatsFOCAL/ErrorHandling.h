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
#ifndef ALICEO2_FOCAL_ERRORHANDLING_H
#define ALICEO2_FOCAL_ERRORHANDLING_H
#include <exception>
#include <iosfwd>
#include <string>

namespace o2::focal
{
class IndexExceptionEvent : public std::exception
{
 public:
  enum class IndexType_t {
    PAD_LAYER,
    PAD_NHALVES,
    PAD_CHANNEL,
    PIXEL_LAYER
  };
  IndexExceptionEvent(unsigned int index, unsigned int maxindex, IndexType_t source);
  ~IndexExceptionEvent() noexcept final = default;

  const char* what() const noexcept final;

  unsigned int getIndex() const noexcept { return mIndex; }
  unsigned int getMaxIndex() const noexcept { return mMaxIndex; }
  IndexType_t getSource() const noexcept { return mSource; }

 private:
  unsigned int mIndex;
  unsigned int mMaxIndex;
  IndexType_t mSource;
  std::string mMessageBuffer;
};

std::ostream& operator<<(std::ostream& in, const IndexExceptionEvent& error);

} // namespace o2::focal

#endif