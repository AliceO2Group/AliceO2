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

#ifndef O2_FIT_DCS_BASE_CONFIG_READER_H
#define O2_FIT_DCS_BASE_CONFIG_READER_H

#include <rapidjson/document.h>
#include <rapidjson/schema.h>
#include <gsl/span>
#include "DetectorsCalibration/Utils.h"

namespace o2::fit
{
class FITDCSBaseConfigReader
{
 public:
  virtual ~FITDCSBaseConfigReader() = default;
  template <typename T, int Size>
  void parseJsonArray(const rapidjson::Value& node, const char* childName, T (&array)[Size])
  {
    if (node.HasMember(childName) == false) {
      throw std::runtime_error(std::string("Failed to find node of name ") + childName);
    }
    const auto& childNode = node[childName];
    if (childNode.IsArray() == false) {
      throw std::runtime_error(std::format("Node {} is not an array!", childName));
    }
    auto jsonArray = childNode.GetArray();
    if (jsonArray.Size() != Size) {
      throw std::runtime_error(std::format("Array {}. Expected array of size {}, parsed array of size {}", childName, Size, jsonArray.Size()));
    }
    for (int idx = 0; idx < Size; idx++) {
      const auto& node = jsonArray[idx];
      if constexpr (std::is_same_v<T, bool>) {
        if (!node.IsBool()) {
          throw std::runtime_error(std::format("{} is not a bool array", childName));
        }
        array[idx] = node.GetBool();
      } else if constexpr (std::is_floating_point_v<T>) {
        if (!node.IsNumber()) {
          throw std::runtime_error(std::format("{} is not an floating point array", childName));
        }
        array[idx] = node.GetFloat();
      } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
        if (!node.IsUint()) {
          throw std::runtime_error(std::format("{} is not an unsigned integer array", childName));
        }
        array[idx] = static_cast<T>(node.GetUint());
      } else if constexpr (std::is_integral_v<T>) {
        if (!node.IsInt()) {
          throw std::runtime_error(std::format("{} is not an integer array", childName));
        }
        array[idx] = static_cast<T>(node.GetInt());
      } else {
        static_assert(std::is_same_v<T, void>, "Unsupported type");
      }
    }
  }

  void setFilename(const std::string& filename)
  {
    mFilename = filename;
  }

  const std::string& getFilename() const
  {
    return mFilename;
  }

  bool matchFilename(const std::string& filename)
  {
    return mFilename == filename;
  }

  bool validateSchema(const rapidjson::Document& docs, std::string& errorMessage);

 protected:
  void loadSchema(std::string_view schema);
  rapidjson::Document parseJsonBuffer(gsl::span<const char> buffer, bool throwOnInvalidSchema = true);

 private:
  std::string mFilename;
  std::unique_ptr<rapidjson::SchemaDocument> mSchema;
};
} // namespace o2::fit
#endif