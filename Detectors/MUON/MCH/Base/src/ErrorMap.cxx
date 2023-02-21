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

#include "MCHBase/ErrorMap.h"

#include <utility>

namespace o2::mch
{

uint64_t encode(uint32_t a, uint32_t b)
{
  uint64_t r = a;
  r = (r << 32) & 0xFFFFFFFF00000000 | b;
  return r;
}

std::pair<uint32_t, uint32_t> decode(uint64_t x)
{
  uint32_t a = static_cast<uint32_t>((x & 0xFFFFFFFF00000000) >> 32);
  uint32_t b = static_cast<uint32_t>(x & 0xFFFFFFFF);
  return std::make_pair(a, b);
}

void ErrorMap::add(ErrorType errorType, uint32_t id0, uint32_t id1, uint64_t n)
{
  auto [itError, isNew] = mErrors[errorType].emplace(encode(id0, id1), Error{errorType, id0, id1, n});
  if (!isNew) {
    itError->second.count += n;
  }
}

void ErrorMap::add(Error error)
{
  auto [itError, isNew] = mErrors[error.type].emplace(encode(error.id0, error.id1), error);
  if (!isNew) {
    itError->second.count += error.count;
  }
}

void ErrorMap::add(gsl::span<const Error> errors)
{
  for (auto error : errors) {
    add(error);
  }
}

void ErrorMap::add(const ErrorMap& errors)
{
  errors.forEach([this](Error error) {
    add(error);
  });
}

uint64_t ErrorMap::getNumberOfErrors() const
{
  uint64_t n{0};
  forEach([&n](Error error) {
    n += error.count;
  });
  return n;
}

uint64_t ErrorMap::getNumberOfErrors(ErrorType type) const
{
  uint64_t n{0};
  forEach(type, [&n](Error error) {
    n += error.count;
  });
  return n;
}

uint64_t ErrorMap::getNumberOfErrors(ErrorGroup group) const
{
  uint64_t n{0};
  forEach(group, [&n](Error error) {
    n += error.count;
  });
  return n;
}

void ErrorMap::forEach(ErrorFunction f) const
{
  for (const auto& typeErrors : mErrors) {
    for (auto error : typeErrors.second) {
      f(error.second);
    }
  }
}

void ErrorMap::forEach(ErrorType type, ErrorFunction f) const
{
  for (const auto& [thisType, errors] : mErrors) {
    if (thisType == type) {
      for (auto error : errors) {
        f(error.second);
      }
    }
  }
}

void ErrorMap::forEach(ErrorGroup group, ErrorFunction f) const
{
  for (const auto& [thisType, errors] : mErrors) {
    if (errorGroup(thisType) == group) {
      for (auto error : errors) {
        f(error.second);
      }
    }
  }
}

} // namespace o2::mch
