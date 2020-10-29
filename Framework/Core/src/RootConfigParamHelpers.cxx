// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/RootConfigParamHelpers.h"
#include "Framework/ConfigParamSpec.h"
#include <TClass.h>
#include <TDataMember.h>
#include <TDataType.h>
#include <TEnum.h>
#include <TEnumConstant.h>
#include <TIterator.h>
#include <TList.h>
#include <iostream>
#include <sstream>
#include <boost/property_tree/ptree.hpp>
#include <functional>
#include <cassert>

using namespace o2::framework;

namespace
{
bool isString(TDataMember const& dm)
{
  return strcmp(dm.GetTrueTypeName(), "string") == 0;
}

// a generic looper of data members of a TClass; calling a callback
// reused in various functions below
void loopOverMembers(TClass* cl, void* obj,
                     std::function<void(const TDataMember*, int, int)>&& callback)
{
  auto memberlist = cl->GetListOfDataMembers();
  for (int i = 0; i < memberlist->GetEntries(); ++i) {
    auto dm = (TDataMember*)memberlist->At(i);

    auto isValidComplex = [dm]() {
      return isString(*dm) || dm->IsEnum();
    };

    // filter out static members for now
    if (dm->Property() & kIsStatic) {
      continue;
    }

    if (dm->IsaPointer()) {
      continue;
    }
    if (!dm->IsBasic() && !isValidComplex()) {
      continue;
    }

    const auto dim = dm->GetArrayDim();
    // we support very simple vectored data in 1D for now
    if (dim > 1) {
      continue;
    }

    const auto size = (dim == 1) ? dm->GetMaxIndex(dim - 1) : 1; // size of array (1 if scalar)
    for (int index = 0; index < size; ++index) {
      callback(dm, index, size);
    }
  }
}

// construct name (in dependence on vector or scalar data and index)
std::string getName(const TDataMember* dm, int index, int size)
{
  std::stringstream namestream;
  namestream << dm->GetName();
  if (size > 1) {
    namestream << "[" << index << "]";
  }
  return namestream.str();
}

void ptreeToMember(boost::property_tree::ptree const& value,
                   char const* tname,
                   TDataMember const* dm,
                   void* ptr)
{
  auto dt = dm->GetDataType();
  if (dt != nullptr) {
    switch (dt->GetType()) {
      case kChar_t: {
        *(char*)ptr = value.get_value<char>();
        return;
      }
      case kUChar_t: {
        *(unsigned char*)ptr = value.get_value<unsigned char>();
        return;
      }
      case kShort_t: {
        *(short*)ptr = value.get_value<short>();
        return;
      }
      case kUShort_t: {
        *(unsigned short*)ptr = value.get_value<unsigned short>();
        return;
      }
      case kInt_t: {
        *(int*)ptr = value.get_value<int>();
        return;
      }
      case kUInt_t: {
        *(unsigned int*)ptr = value.get_value<unsigned int>();
        return;
      }
      case kLong_t: {
        *(long*)ptr = value.get_value<long>();
        return;
      }
      case kULong_t: {
        *(unsigned long*)ptr = value.get_value<unsigned long>();
        return;
      }
      case kFloat_t: {
        *(float*)ptr = value.get_value<float>();
        return;
      }
      case kDouble_t: {
        *(double*)ptr = value.get_value<double>();
        return;
      }
      case kDouble32_t: {
        *(double*)ptr = value.get_value<double>();
        return;
      }
      case kBool_t: {
        *(bool*)ptr = value.get_value<bool>();
        return;
      }
      case kLong64_t: {
        *(int64_t*)ptr = value.get_value<int64_t>();
        return;
      }
      case kULong64_t: {
        *(uint64_t*)ptr = value.get_value<uint64_t>();
        return;
      }
      default: {
        break;
      }
    }
  }
  // if we get here none of the above worked
  if (strcmp(tname, "string") == 0 || strcmp(tname, "std::string")) {
    *(std::string*)ptr = value.get_value<std::string>();
  }
  throw std::runtime_error("Unable to override value");
}

// Convert a DataMember to a ConfigParamSpec
ConfigParamSpec memberToConfigParamSpec(const char* tname, TDataMember const* dm, void* ptr)
{
  auto dt = dm->GetDataType();
  if (dt != nullptr) {
    switch (dt->GetType()) {
      case kChar_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(char*)ptr, {"No help"}};
      }
      case kUChar_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(unsigned char*)ptr, {"No help"}};
      }
      case kShort_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(short*)ptr, {"No help"}};
      }
      case kUShort_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(unsigned short*)ptr, {"No help"}};
      }
      case kInt_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(int*)ptr, {"No help"}};
      }
      case kUInt_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(unsigned int*)ptr, {"No help"}};
      }
      case kLong_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(long*)ptr, {"No help"}};
      }
      case kULong_t: {
        return ConfigParamSpec{tname, VariantType::Int, *(unsigned long*)ptr, {"No help"}};
      }
      case kFloat_t: {
        return ConfigParamSpec{tname, VariantType::Float, *(float*)ptr, {"No help"}};
      }
      case kDouble_t: {
        return ConfigParamSpec{tname, VariantType::Double, *(double*)ptr, {"No help"}};
      }
      case kDouble32_t: {
        return ConfigParamSpec{tname, VariantType::Double, *(double*)ptr, {"No help"}};
      }
      case kBool_t: {
        return ConfigParamSpec{tname, VariantType::Bool, *(bool*)ptr, {"No help"}};
      }
      case kLong64_t: {
        return ConfigParamSpec{tname, VariantType::Int64, *(int64_t*)ptr, {"No help"}};
      }
      case kULong64_t: {
        return ConfigParamSpec{tname, VariantType::Int64, *(uint64_t*)ptr, {"No help"}};
      }
      default: {
        break;
      }
    }
  }
  // if we get here none of the above worked
  if (strcmp(tname, "string") == 0 || strcmp(tname, "std::string")) {
    return ConfigParamSpec{tname, VariantType::String, *(std::string*)ptr, {"No help"}};
  }
  throw std::runtime_error("Cannot use " + std::string(tname));
}
} // namespace

namespace o2::framework
{

std::vector<ConfigParamSpec>
  RootConfigParamHelpers::asConfigParamSpecsImpl(std::string const& mainKey, TClass* cl, void* obj)
{
  std::vector<ConfigParamSpec> specs;

  auto toDataMember = [&mainKey, &specs, obj](const TDataMember* dm, int index, int size) {
    auto dt = dm->GetDataType();
    auto TS = dt ? dt->Size() : 0;
    char* ptr = ((char*)obj) + dm->GetOffset() + index * TS;
    const std::string name = mainKey + "." + getName(dm, index, size);

    specs.push_back(memberToConfigParamSpec(name.c_str(), dm, ptr));
  };

  loopOverMembers(cl, obj, toDataMember);
  return specs;
}

/// Given a TClass, fill the object in obj as if it was member of the former,
/// using the values in the ptree to override, where appropriate.
void RootConfigParamHelpers::fillFromPtree(TClass* cl, void* obj, boost::property_tree::ptree const& pt)
{
  auto toDataMember = [obj, &pt](const TDataMember* dm, int index, int size) {
    auto dt = dm->GetDataType();
    auto TS = dt ? dt->Size() : 0;
    char* ptr = ((char*)obj) + dm->GetOffset() + index * TS;
    const std::string name = getName(dm, index, size);
    auto it = pt.get_child_optional(name);
    if (!it) {
      return;
    }
    ptreeToMember(*it, dm->GetTrueTypeName(), dm, ptr);
  };

  loopOverMembers(cl, obj, toDataMember);
}

} // namespace o2::framework
