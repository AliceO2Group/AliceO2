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
namespace gpu
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
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKING_INCLUDE_STREAM_HIP_H_ */
