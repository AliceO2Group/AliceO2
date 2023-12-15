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
#include <iostream>
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

Variant::Variant(const Variant& other) : mType(other.mType)
{
  // In case this is an array we need to duplicate it to avoid
  // double deletion.
  switch (mType) {
    case variant_trait_v<const char*>:
      mSize = other.mSize;
      variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
      return;
    case variant_trait_v<int*>:
      mSize = other.mSize;
      variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
      return;
    case variant_trait_v<float*>:
      mSize = other.mSize;
      variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
      return;
    case variant_trait_v<double*>:
      mSize = other.mSize;
      variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
      return;
    case variant_trait_v<bool*>:
      mSize = other.mSize;
      variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
      return;
    case variant_trait_v<std::string*>:
      mSize = other.mSize;
      variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
      return;
    default:
      mStore = other.mStore;
      mSize = other.mSize;
  }
}

Variant::Variant(Variant&& other) noexcept : mType(other.mType)
{
  mStore = other.mStore;
  mSize = other.mSize;
  switch (mType) {
    case variant_trait_v<const char*>:
      *reinterpret_cast<char**>(&(other.mStore)) = nullptr;
      return;
    case variant_trait_v<int*>:
      *reinterpret_cast<int**>(&(other.mStore)) = nullptr;
      return;
    case variant_trait_v<float*>:
      *reinterpret_cast<float**>(&(other.mStore)) = nullptr;
      return;
    case variant_trait_v<double*>:
      *reinterpret_cast<double**>(&(other.mStore)) = nullptr;
      return;
    case variant_trait_v<bool*>:
      *reinterpret_cast<bool**>(&(other.mStore)) = nullptr;
      return;
    case variant_trait_v<std::string*>:
      *reinterpret_cast<std::string**>(&(other.mStore)) = nullptr;
    default:
      return;
  }
}

Variant::~Variant()
{
  // In case we allocated an array, we
  // should delete it.
  switch (mType) {
    case variant_trait_v<const char*>:
    case variant_trait_v<int*>:
    case variant_trait_v<float*>:
    case variant_trait_v<double*>:
    case variant_trait_v<bool*>:
    case variant_trait_v<std::string*>:
      if (reinterpret_cast<void**>(&mStore) != nullptr) {
        free(*reinterpret_cast<void**>(&mStore));
      }
      return;
    default:
      return;
  }
}

Variant& Variant::operator=(const Variant& other)
{
  mSize = other.mSize;
  mType = other.mType;
  switch (mType) {
    case variant_trait_v<const char*>:
      variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
      return *this;
    case variant_trait_v<int*>:
      variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
      return *this;
    case variant_trait_v<float*>:
      variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
      return *this;
    case variant_trait_v<double*>:
      variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
      return *this;
    case variant_trait_v<bool*>:
      variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
      return *this;
    case variant_trait_v<std::string*>:
      variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
      return *this;
    default:
      mStore = other.mStore;
      return *this;
  }
}

Variant& Variant::operator=(Variant&& other) noexcept
{
  mSize = other.mSize;
  mType = other.mType;
  switch (mType) {
    case variant_trait_v<const char*>:
      variant_helper<storage_t, const char*>::set(&mStore, other.get<const char*>());
      *reinterpret_cast<char**>(&(other.mStore)) = nullptr;
      return *this;
    case variant_trait_v<int*>:
      variant_helper<storage_t, int*>::set(&mStore, other.get<int*>(), mSize);
      *reinterpret_cast<int**>(&(other.mStore)) = nullptr;
      return *this;
    case variant_trait_v<float*>:
      variant_helper<storage_t, float*>::set(&mStore, other.get<float*>(), mSize);
      *reinterpret_cast<float**>(&(other.mStore)) = nullptr;
      return *this;
    case variant_trait_v<double*>:
      variant_helper<storage_t, double*>::set(&mStore, other.get<double*>(), mSize);
      *reinterpret_cast<double**>(&(other.mStore)) = nullptr;
      return *this;
    case variant_trait_v<bool*>:
      variant_helper<storage_t, bool*>::set(&mStore, other.get<bool*>(), mSize);
      *reinterpret_cast<bool**>(&(other.mStore)) = nullptr;
      return *this;
    case variant_trait_v<std::string*>:
      variant_helper<storage_t, std::string*>::set(&mStore, other.get<std::string*>(), mSize);
      *reinterpret_cast<std::string**>(&(other.mStore)) = nullptr;
      return *this;
    default:
      mStore = other.mStore;
      return *this;
  }
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
