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

//first version 8/2018, Sandro Wenzel

#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonUtils/ConfigurableParam.h"
#include <TClass.h>
#include <TDataMember.h>
#include <TDataType.h>
#include <TEnum.h>
#include <TEnumConstant.h>
#include <TIterator.h>
#include <TList.h>
#include <iostream>
#include <sstream>
#include <fairlogger/Logger.h>
#include <boost/property_tree/ptree.hpp>
#include <functional>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::conf;

// ----------------------------------------------------------------------

std::string ParamDataMember::toString(std::string const& prefix, bool showProv) const
{
  std::string nil = "<null>";

  std::ostringstream out;
  out << prefix << "." << name << " : " << value;

  if (showProv) {
    std::string prov = (provenance.compare("") == 0 ? nil : provenance);
    out << "\t\t[ " + prov + " ]";
  }
  return out.str();
}

std::ostream& operator<<(std::ostream& out, const ParamDataMember& pdm)
{
  out << pdm.toString("", false) << "\n";
  return out;
}

// ----------------------------------------------------------------------

bool isString(TDataMember const& dm)
{
  return strcmp(dm.GetTrueTypeName(), "string") == 0;
}

// ----------------------------------------------------------------------

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
      LOG(warning) << "Pointer types not supported in ConfigurableParams: " << dm->GetFullTypeName() << " " << dm->GetName();
      continue;
    }
    if (!dm->IsBasic() && !isValidComplex()) {
      LOG(warning) << "Generic complex types not supported in ConfigurableParams: " << dm->GetFullTypeName() << " " << dm->GetName();
      continue;
    }

    const auto dim = dm->GetArrayDim();
    // we support very simple vectored data in 1D for now
    if (dim > 1) {
      LOG(warning) << "We support at most 1 dimensional arrays in ConfigurableParams: " << dm->GetFullTypeName() << " " << dm->GetName();
      continue;
    }

    const auto size = (dim == 1) ? dm->GetMaxIndex(dim - 1) : 1; // size of array (1 if scalar)
    for (int index = 0; index < size; ++index) {
      callback(dm, index, size);
    }
  }
}

// ----------------------------------------------------------------------

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

// ----------------------------------------------------------------------
size_t getSizeOfUnderlyingType(const TDataMember& dm)
{
  auto dt = dm.GetDataType();
  if (dt) {
    // if basic built-in type supported by ROOT
    return dt->Size();
  } else {
    // for now only catch std::string as other supported type
    auto tname = dm.GetFullTypeName();
    if (strcmp(tname, "string") == 0 || strcmp(tname, "std::string")) {
      return sizeof(std::string);
    }
    LOG(error) << "ENCOUNTERED AN UNSUPPORTED TYPE " << tname << "IN A CONFIGURABLE PARAMETER";
  }
  return 0;
}

// ----------------------------------------------------------------------

std::string asString(TDataMember const& dm, char* pointer)
{
  // first check if this is a basic data type, in which case
  // we let ROOT do the work
  if (auto dt = dm.GetDataType()) {
    // we put the numeric interpration for char / unsigned char
    // instead of the string one
    if (dt->GetType() == EDataType::kChar_t) {
      auto c = (char)(*pointer);
      return std::to_string((int)c).c_str();
    } else if (dt->GetType() == EDataType::kUChar_t) {
      auto u = (unsigned char)(*pointer);
      return std::to_string((unsigned int)u).c_str();
    }

    auto val = dt->AsString(pointer);

    // For enums we grab the string value of the member
    // and use that instead if its int value
    if (dm.IsEnum()) {
      const auto enumtype = TEnum::GetEnum(dm.GetTypeName());
      assert(enumtype != nullptr);
      const auto constantlist = enumtype->GetConstants();
      assert(constantlist != nullptr);
      if (enumtype) {
        for (int i = 0; i < constantlist->GetEntries(); ++i) {
          const auto e = (TEnumConstant*)(constantlist->At(i));
          if (val == std::to_string((int)e->GetValue())) {
            return std::string(e->GetName());
          }
        }
      }
    }

    return std::string(val);
  }

  // if data member is a std::string just return
  else if (isString(dm)) {
    return ((std::string*)pointer)->c_str();
  }
  // potentially other cases to be added here

  LOG(error) << "COULD NOT REPRESENT AS STRING";
  return nullptr;
}

// ----------------------------------------------------------------------

std::vector<ParamDataMember>* _ParamHelper::getDataMembersImpl(std::string const& mainkey, TClass* cl, void* obj,
                                                               std::map<std::string, ConfigurableParam::EParamProvenance> const* provmap)
{
  std::vector<ParamDataMember>* members = new std::vector<ParamDataMember>;

  auto toDataMember = [&members, obj, mainkey, provmap](const TDataMember* dm, int index, int size) {
    auto TS = getSizeOfUnderlyingType(*dm);
    char* pointer = ((char*)obj) + dm->GetOffset() + index * TS;
    const std::string name = getName(dm, index, size);
    auto value = asString(*dm, pointer);

    std::string prov = "";
    auto iter = provmap->find(mainkey + "." + name);
    if (iter != provmap->end()) {
      prov = ConfigurableParam::toString(iter->second);
    }
    ParamDataMember member{name, value, prov};
    members->push_back(member);
  };

  loopOverMembers(cl, obj, toDataMember);
  return members;
}

// ----------------------------------------------------------------------

// a function converting a string representing a type to the type_info
// because unfortunately typeid(double) != typeid("double")
// but we can take the TDataType (if it exists) as a hint in order to
// minimize string comparisons
std::type_info const& nameToTypeInfo(const char* tname, TDataType const* dt)
{
  if (dt) {
    switch (dt->GetType()) {
      case kChar_t: {
        return typeid(char);
      }
      case kUChar_t: {
        return typeid(unsigned char);
      }
      case kShort_t: {
        return typeid(short);
      }
      case kUShort_t: {
        return typeid(unsigned short);
      }
      case kInt_t: {
        return typeid(int);
      }
      case kUInt_t: {
        return typeid(unsigned int);
      }
      case kLong_t: {
        return typeid(long);
      }
      case kULong_t: {
        return typeid(unsigned long);
      }
      case kFloat_t: {
        return typeid(float);
      }
      case kDouble_t: {
        return typeid(double);
      }
      case kDouble32_t: {
        return typeid(double);
      }
      case kBool_t: {
        return typeid(bool);
      }
      case kLong64_t: {
        return typeid(long long);
      }
      case kULong64_t: {
        return typeid(unsigned long long);
      }
      default: {
        break;
      }
    }
  }
  // if we get here none of the above worked
  if (strcmp(tname, "string") == 0 || strcmp(tname, "std::string")) {
    return typeid(std::string);
  }
  LOG(error) << "ENCOUNTERED AN UNSUPPORTED TYPE " << tname << "IN A CONFIGURABLE PARAMETER";
  return typeid("ERROR");
}

// ----------------------------------------------------------------------

void _ParamHelper::fillKeyValuesImpl(std::string const& mainkey, TClass* cl, void* obj, boost::property_tree::ptree* tree,
                                     std::map<std::string, std::pair<std::type_info const&, void*>>* keytostoragemap,
                                     EnumRegistry* enumRegistry)
{
  boost::property_tree::ptree localtree;
  auto fillMap = [obj, &mainkey, &localtree, &keytostoragemap, &enumRegistry](const TDataMember* dm, int index, int size) {
    const auto name = getName(dm, index, size);
    auto dt = dm->GetDataType();
    auto TS = getSizeOfUnderlyingType(*dm);
    char* pointer = ((char*)obj) + dm->GetOffset() + index * TS;
    localtree.put(name, asString(*dm, pointer));

    auto key = mainkey + "." + name;

    // If it's an enum, we need to store separately all the legal
    // values so that we can map to them from the command line
    if (dm->IsEnum()) {
      enumRegistry->add(key, dm);
    }

    using mapped_t = std::pair<std::type_info const&, void*>;
    auto& ti = nameToTypeInfo(dm->GetTrueTypeName(), dt);
    keytostoragemap->insert(std::pair<std::string, mapped_t>(key, mapped_t(ti, pointer)));
  };
  loopOverMembers(cl, obj, fillMap);
  tree->add_child(mainkey, localtree);
}

// ----------------------------------------------------------------------

void _ParamHelper::printMembersImpl(std::string const& mainkey, std::vector<ParamDataMember> const* members, bool showProv, bool useLogger)
{

  _ParamHelper::outputMembersImpl(std::cout, mainkey, members, showProv, useLogger);
}

void _ParamHelper::outputMembersImpl(std::ostream& out, std::string const& mainkey, std::vector<ParamDataMember> const* members, bool showProv, bool useLogger)
{
  if (members == nullptr) {
    return;
  }

  for (auto& member : *members) {
    if (useLogger) {
      LOG(info) << member.toString(mainkey, showProv);
    } else {
      out << member.toString(mainkey, showProv) << "\n";
    }
  }
}

// ----------------------------------------------------------------------

bool isMemblockDifferent(char const* block1, char const* block2, int sizeinbytes)
{
  // loop over thing in elements of bytes
  for (int i = 0; i < sizeinbytes / sizeof(char); ++i) {
    if (block1[i] != block2[i]) {
      return false;
    }
  }
  return true;
}

// ----------------------------------------------------------------------

void _ParamHelper::assignmentImpl(std::string const& mainkey, TClass* cl, void* to, void* from,
                                  std::map<std::string, ConfigurableParam::EParamProvenance>* provmap)
{
  auto assignifchanged = [to, from, &mainkey, provmap](const TDataMember* dm, int index, int size) {
    const auto name = getName(dm, index, size);
    auto dt = dm->GetDataType();
    auto TS = getSizeOfUnderlyingType(*dm);
    char* pointerto = ((char*)to) + dm->GetOffset() + index * TS;
    char* pointerfrom = ((char*)from) + dm->GetOffset() + index * TS;

    // lambda to update the provenance
    auto updateProv = [&mainkey, name, provmap]() {
      auto key = mainkey + "." + name;
      auto iter = provmap->find(key);
      if (iter != provmap->end()) {
        iter->second = ConfigurableParam::EParamProvenance::kCCDB; // TODO: change to "current STATE"??
      } else {
        LOG(warn) << "KEY " << key << " NOT FOUND WHILE UPDATING PARAMETER PROVENANCE";
      }
    };

    // TODO: this could dispatch to the same method used in ConfigurableParam::setValue
    // but will be slower

    // test if a complicated case
    if (isString(*dm)) {
      std::string& target = *(std::string*)pointerto;
      std::string const& origin = *(std::string*)pointerfrom;
      if (target.compare(origin) != 0) {
        updateProv();
        target = origin;
      }
      return;
    }

    //
    if (!isMemblockDifferent(pointerto, pointerfrom, TS)) {
      updateProv();
      // actually copy
      std::memcpy(pointerto, pointerfrom, getSizeOfUnderlyingType(*dm));
    }
  };
  loopOverMembers(cl, to, assignifchanged);
}

// ----------------------------------------------------------------------

void _ParamHelper::syncCCDBandRegistry(const std::string& mainkey, TClass* cl, void* to, void* from,
                                       std::map<std::string, ConfigurableParam::EParamProvenance>* provmap)
{
  auto sync = [to, from, &mainkey, provmap](const TDataMember* dm, int index, int size) {
    const auto name = getName(dm, index, size);
    auto dt = dm->GetDataType();
    auto TS = getSizeOfUnderlyingType(*dm);
    char* pointerto = ((char*)to) + dm->GetOffset() + index * TS;
    char* pointerfrom = ((char*)from) + dm->GetOffset() + index * TS;

    // check current provenance
    auto key = mainkey + "." + name;
    auto proviter = provmap->find(key);
    bool isRT = proviter != provmap->end() && proviter->second == ConfigurableParam::EParamProvenance::kRT;
    if (isRT) {
      return;
    }
    // lambda to update the provenance
    auto updateProv = [&proviter]() {
      proviter->second = ConfigurableParam::EParamProvenance::kCCDB;
    };

    // test if a complicated case
    if (isString(*dm)) {
      std::string& target = *(std::string*)pointerto;
      std::string const& origin = *(std::string*)pointerfrom;
      // if (target.compare(origin) != 0) {
      updateProv();
      target = origin;
      // }
      return;
    }

    //
    // if (!isMemblockDifferent(pointerto, pointerfrom, TS)) {
    updateProv();
    // actually copy
    std::memcpy(pointerto, pointerfrom, getSizeOfUnderlyingType(*dm));
    //  }
  };
  loopOverMembers(cl, to, sync);
}

// ----------------------------------------------------------------------

void _ParamHelper::printWarning(std::type_info const& tinfo)
{
  LOG(warning) << "Registered parameter class with name " << tinfo.name()
               << " has no ROOT dictionary and will not be available in the configurable parameter system";
}
