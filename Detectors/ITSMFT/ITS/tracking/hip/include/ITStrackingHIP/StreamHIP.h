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
/// \file StreamHIP.h
/// \brief
///

#ifndef O2_ITS_TRACKING_INCLUDE_STREAM_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_STREAM_HIP_H_

#include <hip/hip_runtime_api.h>

namespace o2
{
namespace its
{
namespace GPU
{

class StreamHIP final
{

 public:
  StreamHIP();
  ~StreamHIP();

  StreamHIP(const StreamHIP&) = delete;
  StreamHIP& operator=(const StreamHIP&) = delete;

  const hipStream_t& get() const;

 private:
  hipStream_t mStream;
};
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_STREAM_HIP_H_ */
