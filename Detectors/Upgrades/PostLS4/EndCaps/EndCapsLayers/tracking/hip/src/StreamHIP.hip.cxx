// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
namespace GPU
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

} // namespace GPU
} // namespace its
} // namespace o2
