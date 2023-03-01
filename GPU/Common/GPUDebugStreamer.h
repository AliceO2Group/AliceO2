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

/// \file GPUDebugStreamer.h

#ifndef GPUDEBUGSTREAMER_H
#define GPUDEBUGSTREAMER_H

#include "GPUCommonDef.h"
#if defined(GPUCA_HAVE_O2HEADERS)
#include "CommonUtils/DebugStreamer.h"
#else  // GPUCA_HAVE_O2HEADERS

namespace o2
{
namespace utils
{

/// struct defining the flags which can be used to check if a certain debug streamer is used
enum StreamFlags {
  streamdEdx = 1 << 0,          ///< stream corrections and cluster properties used for the dE/dx
  streamDigitFolding = 1 << 1,  ///< stream ion tail and saturatio information
  streamDigits = 1 << 2,        ///< stream digit information
  streamFastTransform = 1 << 3, ///< stream tpc fast transform
  streamITCorr = 1 << 4,        ///< stream ion tail correction information
  streamDistortionsSC = 1 << 5, ///< stream distortions applied in the TPC space-charge class (used for example in the tpc digitizer)
};

class DebugStreamer
{
 public:
  /// empty for GPU
  template <typename... Args>
  GPUd() void setStreamer(Args... args){};

  /// always false for GPU
  GPUd() static bool checkStream(const StreamFlags) { return false; }

  class StreamerDummy
  {
   public:
    GPUd() int data() const { return 0; };

    template <typename Type>
    GPUd() StreamerDummy& operator<<(Type)
    {
      return *this;
    }
  };

  GPUd() StreamerDummy getStreamer(const int id = 0) const { return StreamerDummy{}; };

  template <typename Type>
  GPUd() StreamerDummy getUniqueTreeName(Type, const int id = 0) const
  {
    return StreamerDummy{};
  }

  GPUd() void flush() const {};
};

} // namespace utils
} // namespace o2
#endif // !GPUCA_HAVE_O2HEADERS

#endif
