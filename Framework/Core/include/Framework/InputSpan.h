// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTSPAN_H
#define FRAMEWORK_INPUTSPAN_H

namespace o2
{
namespace framework
{

/// Mapping helper between the store of all inputs being processed and the
/// actual inputs to be processed in a given go.
/// In general this will use an helper which returns
/// `FairMQMessages->GetData()` from the Message cache, but in principle
/// the mechanism should be flexible enough to
class InputSpan
{
 public:
  /// @a getter is the mapping between an element of the span referred by
  /// index and the buffer associated.
  /// @a size is the number of elements in the span.
  InputSpan(std::function<const char*(size_t)> getter, size_t size)
    : mGetter{getter},
      mSize{size}
  {
  }

  /// @a i-th element of the InputSpan
  char const* get(size_t i) const
  {
    return mGetter(i);
  }

  /// Number of elements in the InputSpan
  size_t size() const
  {
    return mSize;
  }

 private:
  std::function<char const*(size_t)> mGetter;
  size_t mSize;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_INPUTSSPAN_H
