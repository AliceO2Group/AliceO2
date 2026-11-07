// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDisplayTypes.h
/// \author David Rohr

#ifndef GPUDISPLAYTYPES_H
#define GPUDISPLAYTYPES_H

namespace o2::gpu
{
template <class T>
struct GPUDisplayCopyableAtomic : std::atomic<T> { // std::atomic deletes the copy, here it copies the value
  using std::atomic<T>::atomic;
  GPUDisplayCopyableAtomic(const GPUDisplayCopyableAtomic& o) : std::atomic<T>(o.load()) {}
  GPUDisplayCopyableAtomic& operator=(const GPUDisplayCopyableAtomic& o)
  {
    this->store(o.load());
    return *this;
  }
};
} // namespace o2::gpu

#endif
