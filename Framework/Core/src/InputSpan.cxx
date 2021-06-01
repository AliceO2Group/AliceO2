// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/InputSpan.h"

template class std::function<o2::framework::DataRef(size_t)>;
template class std::function<o2::framework::DataRef(size_t, size_t)>;

namespace o2::framework
{
InputSpan::InputSpan(std::function<DataRef(size_t)> getter, size_t size)
  : mGetter{}, mNofPartsGetter{}, mSize{size}
{
  mGetter = [getter](size_t index, size_t) -> DataRef {
    return getter(index);
  };
}

InputSpan::InputSpan(std::function<DataRef(size_t, size_t)> getter, size_t size)
  : mGetter{getter}, mNofPartsGetter{}, mSize{size}
{
}

InputSpan::InputSpan(std::function<DataRef(size_t, size_t)> getter, std::function<size_t(size_t)> nofPartsGetter, size_t size)
  : mGetter{getter}, mNofPartsGetter{nofPartsGetter}, mSize{size}
{
}

} // namespace o2::framework
