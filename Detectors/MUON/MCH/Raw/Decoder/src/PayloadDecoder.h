// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_PAYLOAD_DECODER_H
#define O2_MCH_RAW_PAYLOAD_DECODER_H

#include "Headers/RAWDataHeader.h"
#include "MCHRawDecoder/SampaChannelHandler.h"
#include "MCHRawDecoder/PageDecoder.h"
#include <map>
#include <cstdlib>
#include <gsl/span>

namespace o2
{
namespace mch
{
namespace raw
{
bool hasOrbitJump(uint32_t orb1, uint32_t orb2)
{
  return std::abs(static_cast<long int>(orb1 - orb2)) > 1;
}

using Payload = Page;

/// @brief Decoder for MCH  Raw Data Format.

template <typename T>
class PayloadDecoder
{
 public:
  /// Constructs a decoder
  /// \param channelHandler the handler that will be called for each
  /// piece of sampa data (a SampaCluster, i.e. a part of a time window)
  PayloadDecoder(SampaChannelHandler channelHandler);

  /// decode the buffer (=payload only)
  /// \return the number of bytes used from the buffer
  size_t process(uint32_t orbit, Payload payload);

 private:
  uint32_t mOrbit;
  SampaChannelHandler mChannelHandler;
};

template <typename T>
PayloadDecoder<T>::PayloadDecoder(SampaChannelHandler channelHandler)
  : mChannelHandler(channelHandler)
{
}

template <typename T>
size_t PayloadDecoder<T>::process(uint32_t orbit, Payload payload)
{
  if (hasOrbitJump(orbit, mOrbit)) {
    static_cast<T*>(this)->reset();
  }
  mOrbit = orbit;
  return static_cast<T*>(this)->append(payload);
}

} // namespace raw
} // namespace mch
} // namespace o2

#endif
