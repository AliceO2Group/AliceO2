// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/InputSpec.h"
#include <fairmq/FairMQMessage.h>
#include <cassert>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/builder.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type_traits.h>
#include <arrow/status.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace o2::framework
{

InputRecord::InputRecord(std::vector<InputRoute> const& inputsSchema,
                         InputSpan& span)
  : mInputsSchema{inputsSchema},
    mSpan{span}
{
}

int InputRecord::getPos(const char* binding) const
{
  auto inputIndex = 0;
  for (size_t i = 0; i < mInputsSchema.size(); ++i) {
    auto& route = mInputsSchema[i];
    if (route.timeslice != 0) {
      continue;
    }
    if (route.matcher.binding == binding) {
      return inputIndex;
    }
    ++inputIndex;
  }
  return -1;
}

int InputRecord::getPos(std::string const& binding) const
{
  return this->getPos(binding.c_str());
}

DataRef InputRecord::getByPos(int pos, int part) const
{
  if (pos >= mSpan.size() || pos < 0) {
    throw runtime_error_f("Unknown message requested at position %d", pos);
  }
  if (part > 0 && part >= getNofParts(pos)) {
    throw runtime_error_f("Invalid message part index at %d:%d", pos, part);
  }
  if (pos >= mInputsSchema.size()) {
    throw runtime_error_f("Unknown schema at position %d", pos);
  }
  auto ref = mSpan.get(pos, part);
  ref.spec = &mInputsSchema[pos].matcher;
  return ref;
}

size_t InputRecord::getNofParts(int pos) const
{
  if (pos < 0 || pos >= mSpan.size()) {
    return 0;
  }
  return mSpan.getNofParts(pos);
}
size_t InputRecord::size() const
{
  return mSpan.size();
}

bool InputRecord::isValid(char const* s) const
{
  DataRef ref = get(s);
  if (ref.header == nullptr) {
    return false;
  }
  return true;
}

bool InputRecord::isValid(int s) const
{
  if (s >= size()) {
    return false;
  }
  DataRef ref = getByPos(s);
  if (ref.header == nullptr) {
    return false;
  }
  return true;
}

size_t InputRecord::countValidInputs() const
{
  size_t count = 0;
  for (auto&& _ : *this) {
    ++count;
  }
  return count;
}

} // namespace o2::framework
