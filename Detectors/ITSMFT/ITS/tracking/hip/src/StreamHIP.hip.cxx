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
///
/// \file StreamHIP.hip.cxx
/// \brief
///

#include "ITStrackingHIP/StreamHIP.h"

namespace o2
{
namespace its
{
namespace gpu
{

StreamHIP::StreamHIP()
{
  (void)hipStreamCreateWithFlags(&mStream, hipStreamNonBlocking);
}

StreamHIP::~StreamHIP() // NOLINT: clang-tidy doesn't understand hip macro magic, and thinks this is trivial
{
  (void)hipStreamDestroy(mStream);
}

const hipStream_t& StreamHIP::get() const
{
  return mStream;
}

} // namespace gpu
} // namespace its
} // namespace o2
