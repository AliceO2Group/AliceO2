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
#include "Framework/Variant.h"
#include "Framework/VariantPropertyTreeHelpers.h"
#include "Framework/VariantJSONHelpers.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>

namespace o2::framework
{
std::ostream& operator<<(std::ostream& oss, Variant const& val)
{
  switch (val.type()) {
    case VariantType::Int:
      oss << val.get<int>();
      break;
    case VariantType::Int8:
      oss << (int)val.get<int8_t>();
      break;
    case VariantType::Int16:
      oss << (int)val.get<int16_t>();
      break;
    case VariantType::UInt8:
      oss << (int)val.get<uint8_t>();
      break;
    case VariantType::UInt16:
      oss << (int)val.get<uint16_t>();
      break;
    case VariantType::UInt32:
      oss << val.get<uint32_t>();
      break;
    case VariantType::UInt64:
      oss << val.get<uint64_t>();
      break;
    case VariantType::Int64:
      oss << val.get<int64_t>();
      break;
    case VariantType::Float:
      oss << val.get<float>();
      break;
    case VariantType::Double:
      oss << val.get<double>();
      break;
    case VariantType::String:
      oss << val.get<const char*>();
      break;
    case VariantType::Bool:
      oss << val.get<bool>();
      break;
    case VariantType::ArrayInt:
    case VariantType::ArrayFloat:
    case VariantType::ArrayDouble:
    case VariantType::ArrayBool:
    case VariantType::ArrayString:
    case VariantType::Array2DInt:
    case VariantType::Array2DFloat:
    case VariantType::Array2DDouble:
      VariantJSONHelpers::write(oss, val);
      break;
    case VariantType::Empty:
      break;
    case VariantType::Dict:
      oss << "{}";
      break;
    default:
      oss << "undefined";
      break;
  };
  return oss;
}

std::string Variant::asString() const
{
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

namespace
{
/// Helper visitor for Variant
template <typename F>
bool visitStoredObject(VariantType type, F&& f)
{
  switch (type) {
    case VariantType::ArrayString:
      f.template operator()<std::vector<std::string>>();
      return true;
    case VariantType::Array2DInt:
      f.template operator()<Array2D<int>>();
      return true;
    case VariantType::Array2DFloat:
      f.template operator()<Array2D<float>>();
      return true;
    case VariantType::Array2DDouble:
      f.template operator()<Array2D<double>>();
      return true;
    case VariantType::LabeledArrayInt:
      f.template operator()<LabeledArray<int>>();
      return true;
    case VariantType::LabeledArrayFloat:
      f.template operator()<LabeledArray<float>>();
      return true;
    case VariantType::LabeledArrayDouble:
      f.template operator()<LabeledArray<double>>();
      return true;
    case VariantType::LabeledArrayString:
      f.template operator()<LabeledArray<std::string>>();
      return true;
    default:
      return false;
  }
}

/// Types for which the storage keeps a pointer to a manually allocated buffer
bool holdsMallocedPointer(VariantType type)
{
  switch (type) {
    case VariantType::String:
    case VariantType::ArrayInt:
    case VariantType::ArrayFloat:
    case VariantType::ArrayDouble:
    case VariantType::ArrayBool:
      return true;
    default:
      return false;
  }
}

template <typename T>
T* copyBuffer(T const* values, size_t size)
{
  if (values == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T*>(std::memcpy(std::malloc(size * sizeof(T)), values, size * sizeof(T)));
}
} // namespace

Variant::Variant(VariantType type) : mType{type}
{
  // Make sure that destroying a Variant created without a value is always safe
  // by creating a default stored object upfront
  if (!visitStoredObject(mType, [this]<typename T>() { new (&mStore) T{}; })) {
    std::memset(&mStore, 0, sizeof(mStore));
  }
}

void Variant::copyStore(Variant const& other)
{
  // Proper objects are simply copied
  if (visitStoredObject(mType, [this, &other]<typename T>() { new (&mStore) T(*reinterpret_cast<T const*>(&other.mStore)); })) {
    return;
  }
  // Manually allocated buffers have to be managed
  switch (mType) {
    case VariantType::String: {
      auto const* value = *reinterpret_cast<char const* const*>(&other.mStore);
      *reinterpret_cast<char**>(&mStore) = value != nullptr ? strdup(value) : nullptr;
      return;
    }
    case VariantType::ArrayInt:
      *reinterpret_cast<int**>(&mStore) = copyBuffer(*reinterpret_cast<int* const*>(&other.mStore), mSize);
      return;
    case VariantType::ArrayFloat:
      *reinterpret_cast<float**>(&mStore) = copyBuffer(*reinterpret_cast<float* const*>(&other.mStore), mSize);
      return;
    case VariantType::ArrayDouble:
      *reinterpret_cast<double**>(&mStore) = copyBuffer(*reinterpret_cast<double* const*>(&other.mStore), mSize);
      return;
    case VariantType::ArrayBool:
      *reinterpret_cast<bool**>(&mStore) = copyBuffer(*reinterpret_cast<bool* const*>(&other.mStore), mSize);
      return;
    default:
      // Trivially copyable content
      mStore = other.mStore;
  }
}

void Variant::moveStore(Variant& other) noexcept
{
  // Correct move for objects, leaving proper "moved from" state
  if (visitStoredObject(mType, [this, &other]<typename T>() { new (&mStore) T(std::move(*reinterpret_cast<T*>(&other.mStore))); })) {
    return;
  }
  mStore = other.mStore;
  // Buffers have to change their owner
  if (holdsMallocedPointer(mType)) {
    *reinterpret_cast<void**>(&other.mStore) = nullptr;
  }
}

void Variant::destroyStore() noexcept
{
  // destroy objects
  if (visitStoredObject(mType, [this]<typename T>() { std::destroy_at(reinterpret_cast<T*>(&mStore)); })) {
    return;
  }
  // deallocate buffers
  if (holdsMallocedPointer(mType)) {
    free(*reinterpret_cast<void**>(&mStore));
  }
}

Variant::Variant(const Variant& other) : mType(other.mType), mSize(other.mSize)
{
  copyStore(other);
}

Variant::Variant(Variant&& other) noexcept : mType(other.mType), mSize(other.mSize)
{
  moveStore(other);
}

Variant::~Variant()
{
  destroyStore();
}

Variant& Variant::operator=(const Variant& other)
{
  if (this != &other) {
    // Copy first, so that a throwing copy leaves this Variant untouched
    Variant copy(other);
    *this = std::move(copy);
  }
  return *this;
}

Variant& Variant::operator=(Variant&& other) noexcept
{
  if (this != &other) {
    destroyStore();
    mType = other.mType;
    mSize = other.mSize;
    moveStore(other);
  }
  return *this;
}

std::pair<std::vector<std::string>, std::vector<std::string>> extractLabels(boost::property_tree::ptree const& tree)
{
  std::vector<std::string> labels_rows;
  std::vector<std::string> labels_cols;
  auto lrc = tree.get_child_optional(labels_rows_str);
  if (lrc) {
    labels_rows = basicVectorFromBranch<std::string>(lrc.value());
  }
  auto lcc = tree.get_child_optional(labels_cols_str);
  if (lcc) {
    labels_cols = basicVectorFromBranch<std::string>(lcc.value());
  }
  return std::make_pair(labels_rows, labels_cols);
}

} // namespace o2::framework
