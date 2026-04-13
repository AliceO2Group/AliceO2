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

#include "Framework/InputSpan.h"

template class std::function<o2::framework::DataRef(size_t, o2::framework::DataRefIndices)>;
template class std::function<o2::framework::DataRefIndices(size_t, o2::framework::DataRefIndices)>;

namespace o2::framework
{
InputSpan::InputSpan(std::function<size_t(size_t)> nofPartsGetter,
                     std::function<int(size_t)> refCountGetter,
                     std::function<DataRef(size_t, DataRefIndices)> indicesGetter,
                     std::function<DataRefIndices(size_t, DataRefIndices)> nextIndicesGetter,
                     size_t size)
  : mNofPartsGetter{nofPartsGetter}, mRefCountGetter(refCountGetter), mIndicesGetter{std::move(indicesGetter)}, mNextIndicesGetter{std::move(nextIndicesGetter)}, mSize{size}
{
}

} // namespace o2::framework
