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
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/InputSpec.h"
#include "Framework/ObjectCache.h"
#include "Framework/CallbackService.h"
#include <fairmq/Message.h>
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
                         InputSpan& span,
                         ServiceRegistryRef registry)
  : mRegistry{registry},
    mInputsSchema{inputsSchema},
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

InputRecord::InputPos InputRecord::getPos(std::vector<InputRoute> const& schema, ConcreteDataMatcher concrete)
{
  size_t inputIndex = 0;
  for (const auto& route : schema) {
    if (route.timeslice != 0) {
      continue;
    }
    if (DataSpecUtils::match(route.matcher, concrete)) {
      return {inputIndex};
    }
    ++inputIndex;
  }
  return InputPos{InputPos::INVALID};
}

int InputRecord::getPos(std::string const& binding) const
{
  return this->getPos(binding.c_str());
}

DataRef InputRecord::getByPos(int pos, int part) const
{
  return InputRecord::getByPos(mInputsSchema, mSpan, pos, part);
}

DataRef InputRecord::getByPos(std::vector<InputRoute> const& schema, InputSpan const& span, int pos, int part)
{
  if (pos >= (int)span.size() || pos < 0) {
    throw runtime_error_f("Unknown message requested at position %d", pos);
  }
  if (part > 0 && part >= (int)span.getNofParts(pos)) {
    throw runtime_error_f("Invalid message part index at %d:%d", pos, part);
  }
  if (pos >= (int)schema.size()) {
    throw runtime_error_f("Unknown schema at position %d", pos);
  }
  auto ref = span.get(pos, part);
  auto inputIndex = 0;
  auto schemaIndex = 0;
  for (size_t i = 0; i < schema.size(); ++i) {
    schemaIndex = i;
    auto& route = schema[i];
    if (route.timeslice != 0) {
      continue;
    }
    if (inputIndex == pos) {
      break;
    }
    ++inputIndex;
  }
  ref.spec = &schema[schemaIndex].matcher;
  return ref;
}

DataRef InputRecord::getFirstValid(bool throwOnFailure) const
{
  for (size_t i = 0; i < size(); i++) {
    auto ref = mSpan.get(i);
    if (ref.header != nullptr) {
      ref.spec = &mInputsSchema[i].matcher;
      return ref;
    }
  }
  if (throwOnFailure) {
    throw runtime_error_f("No valid input found out of total ", size());
  }
  return {};
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
