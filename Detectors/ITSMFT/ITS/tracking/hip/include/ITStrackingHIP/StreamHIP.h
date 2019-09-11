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
/// \file Stream.h
/// \brief
///

#ifndef TRAKINGITSU_INCLUDE_GPU_STREAM_H_
#define TRAKINGITSU_INCLUDE_GPU_STREAM_H_

#include "ITStracking/Definitions.h"

namespace o2
{
namespace its
{
namespace GPU
{

class Stream final
{

 public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  const GPUStream& get() const;

 private:
  GPUStream mStream;
};
} // namespace GPU
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_STREAM_H_ */
