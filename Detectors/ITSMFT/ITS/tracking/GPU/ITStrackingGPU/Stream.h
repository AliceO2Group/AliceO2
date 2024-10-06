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
/// \file Stream.h
/// \brief
///

#ifndef ITSTRACKINGGPU_STREAM_H_
#define ITSTRACKINGGPU_STREAM_H_

#include "ITStracking/Definitions.h"

namespace o2
{
namespace its
{
namespace gpu
{

class Stream final
{

 public:
  Stream();
  ~Stream();

  [[nodiscard]] const GPUStream& get() const;

 private:
  GPUStream mStream;
};
} // namespace gpu
} // namespace its
} // namespace o2

#endif /* TRAKINGITSU_INCLUDE_GPU_STREAM_H_ */
