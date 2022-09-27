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

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/StringUtils.h"
#include "CommonUtils/KeyValParam.h"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <array>
#include <limits>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <iostream>
#include <string>
#include <FairLogger.h>
#include <typeinfo>
#include "TDataMember.h"
#include "TDataType.h"
#include "TFile.h"
#include "TEnum.h"
#include "TEnumConstant.h"
#include <filesystem>

namespace o2
{
namespace conf
{
std::vector<ConfigurableParam*>* ConfigurableParam::sRegisteredParamClasses = nullptr;
boost::property_tree::ptree* ConfigurableParam::sPtree = nullptr;
std::map<std::string, std::pair<std::type_info const&, void*>>* ConfigurableParam::sKeyToStorageMap = nullptr;
std::map<std::string, ConfigurableParam::EParamProvenance>* ConfigurableParam::sValueProvenanceMap = nullptr;
std::string ConfigurableParam::sInputDir = "";
std::string ConfigurableParam::sOutputDir = "";
EnumRegistry* ConfigurableParam::sEnumRegistry = nullptr;

bool ConfigurableParam::sIsFullyInitialized = false;
bool ConfigurableParam::sRegisterMode = true;

// ------------------------------------------------------------------

std::ostream& operator<<(std::ostream& out, ConfigurableParam const& param)
{
  param.output(out);
  return out;
}

// Does the given key exist in the boost property tree?
bool keyInTree(boost::property_tree::ptree* pt, const std::string& key)
{
  if (key.size() == 0 || pt == nullptr) {
    return false;
  }
  bool reply = false;
  try {
    reply = pt->get_optional<std::string>(key).is_initialized();
  } catch (std::exception const& e) {
    LOG(error) << "ConfigurableParam: Exception when checking for key " << key << " : " << e.what();
  }
  return reply;
}

// ------------------------------------------------------------------

void EnumRegistry::add(const std::string& key, const TDataMember* dm)
{
  if (!dm->IsEnum() || this->contains(key)) {
    return;
  }

  EnumLegalValues legalVals;
  auto enumtype = TEnum::GetEnum(dm->GetTypeName());
  assert(enumtype != nullptr);
  auto constantlist = enumtype->GetConstants();
  assert(constantlist != nullptr);
  if (enumtype) {
    for (int i = 0; i < constantlist->GetEntries(); ++i) {
      auto e = (TEnumConstant*)(constantlist->At(i));
      std::pair<std::string, int> val(e->GetName(), (int)e->GetValue());
      legalVals.vvalues.push_back(val);
    }
  }

  // The other method of fetching enum constants from TDataMember->GetOptions
  // stopped working with ROOT6-18-0:

  // auto opts = dm->GetOptions();
  // for (int i = 0; i < opts->GetEntries(); ++i) {
  //   auto opt = (TOptionListItem*)opts->At(i);
  //   std::pair<std::string, int> val(opt->fOptName, (int)opt->fValue);
  //   legalVals.vvalues.push_back(val);
  //   LOG(info) << "Adding legal value " << val.first << " " << val.second;
  // }

  auto entry = std::pair<std::string, EnumLegalValues>(key, legalVals);
  this->entries.insert(entry);
}

std::string EnumRegistry::toString() const
{
  std::string out = "";
  for (auto& entry : entries) {
    out.append(entry.first + " => ");
    out.append(entry.second.toString());
    out.append("\n");
  }

  LOG(info) << out;
  return out;
}

std::string EnumLegalValues::toString() const
{
  std::string out = "";

  for (auto& value : vvalues) {
    out.append("[");
    out.append(value.first);
    out.append(" | ");
    out.append(std::to_string(value.second));
    out.append("] ");
  }

  return out;
}

// getIntValue takes a string value which is supposed to be
// a legal enum value and tries to cast it to an int.
// If it succeeds, and if the int value is legal, it is returned.
// If it fails, and if it is a legal string enum value, we look up
// and return the equivalent int value. In any case, if it is not
// a legal value we return -1 to indicate this fact.
int EnumLegalValues::getIntValue(const std::string& value) const
{
  try {
    int val = boost::lexical_cast<int>(value);
    if (isLegal(val)) {
      return val;
    }
  } catch (const boost::bad_lexical_cast& e) {
    if (isLegal(value)) {
      for (auto& pair : vvalues) {
        if (pair.first == value) {
          return pair.second;
        }
      }
    }
  }

  return -1;
}

// -----------------------------------------------------------------

void ConfigurableParam::writeINI(std::string const& filename, std::string const& keyOnly)
{
  if (sOutputDir == "/dev/null") {
    LOG(debug) << "ignoring writing of ini file " << filename;
    return;
  }
  auto outfilename = o2::utils::Str::concat_string(sOutputDir, filename);
  initPropertyTree();     // update the boost tree before writing
  if (!keyOnly.empty()) { // write ini for selected key only
    try {
      boost::property_tree::ptree kTree;
      kTree.add_child(keyOnly, sPtree->get_child(keyOnly));
      boost::property_tree::write_ini(outfilename, kTree);
    } catch (const boost::property_tree::ptree_bad_path& err) {
      LOG(fatal) << "non-existing key " << keyOnly << " provided to writeINI";
    }
  } else {
    boost::property_tree::write_ini(outfilename, *sPtree);
  }
}

// ------------------------------------------------------------------

bool ConfigurableParam::configFileExists(std::string const& filepath)
{
  return std::filesystem::exists(o2::utils::Str::concat_string(sInputDir, filepath));
}

// ------------------------------------------------------------------

boost::property_tree::ptree ConfigurableParam::readConfigFile(std::string const& filepath)
{
  auto inpfilename = o2::utils::Str::concat_string(sInputDir, filepath);
  if (!std::filesystem::exists(inpfilename)) {
    LOG(fatal) << inpfilename << " : config file does not exist!";
  }

  boost::property_tree::ptree pt;

  if (boost::iends_with(inpfilename, ".ini")) {
    pt = readINI(inpfilename);
  } else if (boost::iends_with(inpfilename, ".json")) {
    pt = readJSON(inpfilename);
  } else {
    LOG(fatal) << "Configuration file must have either .ini or .json extension";
  }

  return pt;
}

// ------------------------------------------------------------------

boost::property_tree::ptree ConfigurableParam::readINI(std::string const& filepath)
{
  boost::property_tree::ptree pt;
  try {
    boost::property_tree::read_ini(filepath, pt);
  } catch (const boost::property_tree::ptree_error& e) {
    LOG(fatal) << "Failed to read INI config file " << filepath << " (" << e.what() << ")";
  } catch (...) {
    LOG(fatal) << "Unknown error when reading INI config file ";
  }

  return pt;
}

// ------------------------------------------------------------------

boost::property_tree::ptree ConfigurableParam::readJSON(std::string const& filepath)
{
  boost::property_tree::ptree pt;

  try {
    boost::property_tree::read_json(filepath, pt);
  } catch (const boost::property_tree::ptree_error& e) {
    LOG(fatal) << "Failed to read JSON config file " << filepath << " (" << e.what() << ")";
  }

  return pt;
}

// ------------------------------------------------------------------

void ConfigurableParam::writeJSON(std::string const& filename, std::string const& keyOnly)
{
  if (sOutputDir == "/dev/null") {
    LOG(info) << "ignoring writing of json file " << filename;
    return;
  }
  initPropertyTree();     // update the boost tree before writing
  auto outfilename = o2::utils::Str::concat_string(sOutputDir, filename);
  if (!keyOnly.empty()) { // write ini for selected key only
    try {
      boost::property_tree::ptree kTree;
      kTree.add_child(keyOnly, sPtree->get_child(keyOnly));
      boost::property_tree::write_json(outfilename, kTree);
    } catch (const boost::property_tree::ptree_bad_path& err) {
      LOG(fatal) << "non-existing key " << keyOnly << " provided to writeJSON";
    }
  } else {
    boost::property_tree::write_json(outfilename, *sPtree);
  }
}

// ------------------------------------------------------------------

void ConfigurableParam::initPropertyTree()
{
  sPtree->clear();
  for (auto p : *sRegisteredParamClasses) {
    p->putKeyValues(sPtree);
  }
}

// ------------------------------------------------------------------

void ConfigurableParam::printAllKeyValuePairs(bool useLogger)
{
  if (!sIsFullyInitialized) {
    initialize();
  }
  std::cout << "####\n";
  for (auto p : *sRegisteredParamClasses) {
    p->printKeyValues(true, useLogger);
  }
  std::cout << "----\n";
}

// ------------------------------------------------------------------

ConfigurableParam::EParamProvenance ConfigurableParam::getProvenance(const std::string& key)
{
  if (!sIsFullyInitialized) {
    initialize();
  }
  auto iter = sValueProvenanceMap->find(key);
  if (iter == sValueProvenanceMap->end()) {
    throw std::runtime_error(fmt::format("provenace of unknown {:s} parameter is requested", key));
  }
  return iter->second;
}

// ------------------------------------------------------------------

// evidently this could be a local file or an OCDB server
// ... we need to generalize this ... but ok for demonstration purposes
void ConfigurableParam::toCCDB(std::string filename)
{
  if (!sIsFullyInitialized) {
    initialize();
  }
  TFile file(filename.c_str(), "RECREATE");
  for (auto p : *sRegisteredParamClasses) {
    p->serializeTo(&file);
  }
  file.Close();
}

// ------------------------------------------------------------------

void ConfigurableParam::fromCCDB(std::string filename)
{
  if (!sIsFullyInitialized) {
    initialize();
  }
  TFile file(filename.c_str(), "READ");
  for (auto p : *sRegisteredParamClasses) {
    p->initFrom(&file);
  }
  file.Close();
}

// ------------------------------------------------------------------

ConfigurableParam::ConfigurableParam()
{
  if (sRegisteredParamClasses == nullptr) {
    sRegisteredParamClasses = new std::vector<ConfigurableParam*>;
  }
  if (sPtree == nullptr) {
    sPtree = new boost::property_tree::ptree;
  }
  if (sKeyToStorageMap == nullptr) {
    sKeyToStorageMap = new std::map<std::string, std::pair<std::type_info const&, void*>>;
  }
  if (sValueProvenanceMap == nullptr) {
    sValueProvenanceMap = new std::map<std::string, ConfigurableParam::EParamProvenance>;
  }

  if (sEnumRegistry == nullptr) {
    sEnumRegistry = new EnumRegistry();
  }

  if (sRegisterMode == true) {
    sRegisteredParamClasses->push_back(this);
  }
}

// ------------------------------------------------------------------

void ConfigurableParam::initialize()
{
  initPropertyTree();
  // initialize the provenance map
  // initially the values come from code
  for (auto& key : *sKeyToStorageMap) {
    sValueProvenanceMap->insert(std::pair<std::string, ConfigurableParam::EParamProvenance>(key.first, kCODE));
  }
  sIsFullyInitialized = true;
}

// ------------------------------------------------------------------

void ConfigurableParam::printAllRegisteredParamNames()
{
  for (auto p : *sRegisteredParamClasses) {
    std::cout << p->getName() << "\n";
  }
}

// ------------------------------------------------------------------

// Update the storage map of params from the given configuration file.
// It can be in JSON or INI format.
// If nonempty comma-separated paramsList is provided, only those params will
// be updated, absence of data for any of requested params will lead to fatal
// If unchangedOnly is true, then only those parameters whose provenance is kCODE will be updated
// (to allow prefernce of run-time settings)
void ConfigurableParam::updateFromFile(std::string const& configFile, std::string const& paramsList, bool unchangedOnly)
{
  if (!sIsFullyInitialized) {
    initialize();
  }

  auto cfgfile = o2::utils::Str::trim_copy(configFile);

  if (cfgfile.length() == 0) {
    return;
  }

  boost::property_tree::ptree pt = readConfigFile(cfgfile);

  std::vector<std::pair<std::string, std::string>> keyValPairs;
  auto request = o2::utils::Str::tokenize(paramsList, ',', true);
  std::unordered_map<std::string, int> requestMap;
  for (const auto& par : request) {
    if (!par.empty()) {
      requestMap[par] = 0;
    }
  }

  try {
    for (auto& section : pt) {
      std::string mainKey = section.first;
      if (requestMap.size()) {
        if (requestMap.find(mainKey) == requestMap.end()) {
          continue; // if something was requested, ignore everything else
        } else {
          requestMap[mainKey] = 1;
        }
      }
      for (auto& subKey : section.second) {
        auto name = subKey.first;
        auto value = subKey.second.get_value<std::string>();
        std::string key = mainKey + "." + name;
        if (!unchangedOnly || getProvenance(key) == kCODE) {
          std::pair<std::string, std::string> pair = std::make_pair(key, o2::utils::Str::trim_copy(value));
          keyValPairs.push_back(pair);
        }
      }
    }
  } catch (std::exception const& error) {
    LOG(error) << "Error while updating params " << error.what();
  } catch (...) {
    LOG(error) << "Unknown while updating params ";
  }

  // make sure all requested params were retrieved
  for (const auto& req : requestMap) {
    if (req.second == 0) {
      throw std::runtime_error(fmt::format("Param {:s} was not found in {:s}", req.first, configFile));
    }
  }

  try {
    setValues(keyValPairs);
  } catch (std::exception const& error) {
    LOG(error) << "Error while setting values " << error.what();
  }
}

// ------------------------------------------------------------------
// ------------------------------------------------------------------

void ConfigurableParam::updateFromString(std::string const& configString)
{
  if (!sIsFullyInitialized) {
    initialize();
  }

  auto cfgStr = o2::utils::Str::trim_copy(configString);
  if (cfgStr.length() == 0) {
    return;
  }

  // Take a vector of strings with elements of form a=b, and
  // return a vector of pairs with each pair of form <a, b>
  auto toKeyValPairs = [](std::vector<std::string>& tokens) {
    std::vector<std::pair<std::string, std::string>> pairs;

    for (auto& token : tokens) {
      auto keyval = o2::utils::Str::tokenize(token, '=');
      if (keyval.size() != 2) {
        LOG(fatal) << "Illegal command-line key/value string: " << token;
        continue;
      }

      std::pair<std::string, std::string> pair = std::make_pair(keyval[0], o2::utils::Str::trim_copy(keyval[1]));
      pairs.push_back(pair);
    }

    return pairs;
  };

  // Simple check that the string starts/ends with an open square bracket
  // Find the maximum index of a given key with array value.
  // We store string keys for arrays as a[0]...a[size_of_array]
  /*
  auto maxIndex = [](std::string baseName) {
    bool isFound = true;
    int index = -1;
    do {
      index++;
      std::string key = baseName + "[" + std::to_string(index) + "]";
      isFound = keyInTree(sPtree, key);
    } while (isFound);

    return index;
  };
*/

  // ---- end of helper functions --------------------

  // Command-line string is a ;-separated list of key=value params
  auto params = o2::utils::Str::tokenize(configString, ';', true);

  // Now split each key=value string into its std::pair<key, value> parts
  auto keyValues = toKeyValPairs(params);

  setValues(keyValues);

  const auto& kv = o2::conf::KeyValParam::Instance();
  if (getProvenance("keyval.input_dir") != kCODE) {
    sInputDir = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(kv.input_dir));
  }
  if (getProvenance("keyval.output_dir") != kCODE) {
    if (kv.output_dir == "/dev/null") {
      sOutputDir = kv.output_dir;
    } else {
      sOutputDir = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(kv.output_dir));
    }
  }
}

// setValues takes a vector of pairs where each pair is a key and value
// to be set in the storage map
void ConfigurableParam::setValues(std::vector<std::pair<std::string, std::string>> const& keyValues)
{
  auto isArray = [](std::string& el) {
    return el.size() > 0 && (el.at(0) == '[') && (el.at(el.size() - 1) == ']');
  };

  // Take a vector of param key/value pairs
  // and update the storage map for each of them by calling setValue.
  // 1. For string/scalar types this is simple.
  // 2. For array values we need to iterate over each array element
  // and call setValue on the element, using an appropriately constructed key.
  // 3. For enum types we check for the existence of the key in the enum registry
  // and also confirm that the value is in the list of legal values
  for (auto& keyValue : keyValues) {
    std::string key = keyValue.first;
    std::string value = o2::utils::Str::trim_copy(keyValue.second);

    if (!keyInTree(sPtree, key)) {
      LOG(fatal) << "Inexistant ConfigurableParam key: " << key;
    }

    if (sEnumRegistry->contains(key)) {
      setEnumValue(key, value);
    } else if (isArray(value)) {
      setArrayValue(key, value);
    } else {
      assert(sKeyToStorageMap->find(key) != sKeyToStorageMap->end());

      // If the value is given as a boolean true|false, change to 1|0 int equivalent
      if (value == "true") {
        value = "1";
      } else if (value == "false") {
        value = "0";
      }

      // TODO: this will trap complex types like maps and structs.
      // These need to be broken into their own cases, so that we only
      // get strings and scalars here.
      setValue(key, value);
    }
  }
}

void ConfigurableParam::setArrayValue(const std::string& key, const std::string& value)
{
  // We remove the lead/trailing square bracket
  // value.erase(0, 1).pop_back();
  auto elems = o2::utils::Str::tokenize(value.substr(1, value.length() - 2), ',', true);

  // TODO:
  // 1. Should not assume each array element is a scalar/string. We may need to recurse.
  // 2. Should not assume each array element - even if not complex - is correctly written. Validate.
  // 3. Validation should include finding same types as in provided defaults.
  for (int i = 0; i < elems.size(); ++i) {
    std::string indexKey = key + "[" + std::to_string(i) + "]";
    setValue(indexKey, elems[i]);
  }
}

void ConfigurableParam::setEnumValue(const std::string& key, const std::string& value)
{
  int val = (*sEnumRegistry)[key]->getIntValue(value);
  if (val == -1) {
    LOG(fatal) << "Illegal value "
               << value << " for enum " << key
               << ". Legal string|int values:\n"
               << (*sEnumRegistry)[key]->toString() << std::endl;
  }

  setValue(key, std::to_string(val));
}

void unsupp() { std::cerr << "currently unsupported\n"; }

template <typename T>
bool isMemblockDifferent(void const* block1, void const* block2)
{
  // loop over thing in elements of bytes
  for (int i = 0; i < sizeof(T) / sizeof(char); ++i) {
    if (((char*)block1)[i] != ((char*)block2)[i]) {
      return true;
    }
  }
  return false;
}

// copies data from one place to other and returns
// true of data was actually changed
template <typename T>
ConfigurableParam::EParamUpdateStatus Copy(void const* addr, void* targetaddr)
{
  if (isMemblockDifferent<T>(addr, targetaddr)) {
    std::memcpy(targetaddr, addr, sizeof(T));
    return ConfigurableParam::EParamUpdateStatus::Changed;
  }
  return ConfigurableParam::EParamUpdateStatus::Unchanged;
}

ConfigurableParam::EParamUpdateStatus ConfigurableParam::updateThroughStorageMap(std::string mainkey, std::string subkey, std::type_info const& tinfo,
                                                                                 void* addr)
{
  // check if key_exists
  auto key = mainkey + "." + subkey;
  auto iter = sKeyToStorageMap->find(key);
  if (iter == sKeyToStorageMap->end()) {
    LOG(warn) << "Cannot update parameter " << key << " not found";
    return ConfigurableParam::EParamUpdateStatus::Failed;
  }

  // the type we need to convert to
  int type = TDataType::GetType(tinfo);

  // check that type matches
  if (iter->second.first != tinfo) {
    LOG(warn) << "Types do not match; cannot update value";
    return ConfigurableParam::EParamUpdateStatus::Failed;
  }

  auto targetaddress = iter->second.second;
  switch (type) {
    case kChar_t: {
      return Copy<char>(addr, targetaddress);
      break;
    }
    case kUChar_t: {
      return Copy<unsigned char>(addr, targetaddress);
      break;
    }
    case kShort_t: {
      return Copy<short>(addr, targetaddress);
      break;
    }
    case kUShort_t: {
      return Copy<unsigned short>(addr, targetaddress);
      break;
    }
    case kInt_t: {
      return Copy<int>(addr, targetaddress);
      break;
    }
    case kUInt_t: {
      return Copy<unsigned int>(addr, targetaddress);
      break;
    }
    case kLong_t: {
      return Copy<long>(addr, targetaddress);
      break;
    }
    case kULong_t: {
      return Copy<unsigned long>(addr, targetaddress);
      break;
    }
    case kFloat_t: {
      return Copy<float>(addr, targetaddress);
      break;
    }
    case kDouble_t: {
      return Copy<double>(addr, targetaddress);
      break;
    }
    case kDouble32_t: {
      return Copy<double>(addr, targetaddress);
      break;
    }
    case kchar: {
      unsupp();
      break;
    }
    case kBool_t: {
      return Copy<bool>(addr, targetaddress);
      break;
    }
    case kLong64_t: {
      return Copy<long long>(addr, targetaddress);
      break;
    }
    case kULong64_t: {
      return Copy<unsigned long long>(addr, targetaddress);
      break;
    }
    case kOther_t: {
      unsupp();
      break;
    }
    case kNoType_t: {
      unsupp();
      break;
    }
    case kFloat16_t: {
      unsupp();
      break;
    }
    case kCounter: {
      unsupp();
      break;
    }
    case kCharStar: {
      return Copy<char*>(addr, targetaddress);
      break;
    }
    case kBits: {
      unsupp();
      break;
    }
    case kVoid_t: {
      unsupp();
      break;
    }
    case kDataTypeAliasUnsigned_t: {
      unsupp();
      break;
    }
    /*
    case kDataTypeAliasSignedChar_t: {
      unsupp();
      break;
    }
    case kNumDataTypes: {
      unsupp();
      break;
  }*/
    default: {
      unsupp();
      break;
    }
  }
  return ConfigurableParam::EParamUpdateStatus::Failed;
}

template <typename T>
ConfigurableParam::EParamUpdateStatus ConvertAndCopy(std::string const& valuestring, void* targetaddr)
{
  auto addr = boost::lexical_cast<T>(valuestring);
  if (isMemblockDifferent<T>(targetaddr, (void*)&addr)) {
    std::memcpy(targetaddr, (void*)&addr, sizeof(T));
    return ConfigurableParam::EParamUpdateStatus::Changed;
  }
  return ConfigurableParam::EParamUpdateStatus::Unchanged;
}

// special version for std::string
template <>
ConfigurableParam::EParamUpdateStatus ConvertAndCopy<std::string>(std::string const& valuestring, void* targetaddr)
{
  std::string& target = *((std::string*)targetaddr);
  if (target.compare(valuestring) != 0) {
    // the targetaddr is a std::string to which we can simply assign
    // and all the magic will happen internally
    target = valuestring;
    return ConfigurableParam::EParamUpdateStatus::Changed;
  }
  return ConfigurableParam::EParamUpdateStatus::Unchanged;
}
// special version for char and unsigned char since we are interested in the numeric
// meaning of char as an 8-bit integer (boost lexical cast is assigning the string as a character i// nterpretation
template <>
ConfigurableParam::EParamUpdateStatus ConvertAndCopy<char>(std::string const& valuestring, void* targetaddr)
{
  int intvalue = boost::lexical_cast<int>(valuestring);
  if (intvalue > std::numeric_limits<char>::max() || intvalue < std::numeric_limits<char>::min()) {
    LOG(error) << "Cannot assign " << valuestring << " to a char variable";
    return ConfigurableParam::EParamUpdateStatus::Failed;
  }
  char addr = intvalue;
  if (isMemblockDifferent<char>(targetaddr, (void*)&addr)) {
    std::memcpy(targetaddr, (void*)&addr, sizeof(char));
    return ConfigurableParam::EParamUpdateStatus::Changed;
  }
  return ConfigurableParam::EParamUpdateStatus::Unchanged;
}

template <>
ConfigurableParam::EParamUpdateStatus ConvertAndCopy<unsigned char>(std::string const& valuestring, void* targetaddr)
{
  unsigned int intvalue = boost::lexical_cast<int>(valuestring);
  if (intvalue > std::numeric_limits<unsigned char>::max() || intvalue < std::numeric_limits<unsigned char>::min()) {
    LOG(error) << "Cannot assign " << valuestring << " to an unsigned char variable";
    return ConfigurableParam::EParamUpdateStatus::Failed;
  }
  unsigned char addr = intvalue;
  if (isMemblockDifferent<unsigned char>(targetaddr, (void*)&addr)) {
    std::memcpy(targetaddr, (void*)&addr, sizeof(unsigned char));
    return ConfigurableParam::EParamUpdateStatus::Changed;
  }
  return ConfigurableParam::EParamUpdateStatus::Unchanged;
}

ConfigurableParam::EParamUpdateStatus ConfigurableParam::updateThroughStorageMapWithConversion(std::string const& key, std::string const& valuestring)
{
  // check if key_exists
  auto iter = sKeyToStorageMap->find(key);
  if (iter == sKeyToStorageMap->end()) {
    LOG(warn) << "Cannot update parameter " << key << " (parameter not found) ";
    return ConfigurableParam::EParamUpdateStatus::Failed;
  }

  auto targetaddress = iter->second.second;

  // treat some special cases first:
  // the type is actually a std::string
  if (iter->second.first == typeid(std::string)) {
    return ConvertAndCopy<std::string>(valuestring, targetaddress);
  }

  // the type (aka ROOT::EDataType which the type identification in the map) we need to convert to
  int targettype = TDataType::GetType(iter->second.first);

  switch (targettype) {
    case kChar_t: {
      return ConvertAndCopy<char>(valuestring, targetaddress);
      break;
    }
    case kUChar_t: {
      return ConvertAndCopy<unsigned char>(valuestring, targetaddress);
      break;
    }
    case kShort_t: {
      return ConvertAndCopy<short>(valuestring, targetaddress);
      break;
    }
    case kUShort_t: {
      return ConvertAndCopy<unsigned short>(valuestring, targetaddress);
      break;
    }
    case kInt_t: {
      return ConvertAndCopy<int>(valuestring, targetaddress);
      break;
    }
    case kUInt_t: {
      return ConvertAndCopy<unsigned int>(valuestring, targetaddress);
      break;
    }
    case kLong_t: {
      return ConvertAndCopy<long>(valuestring, targetaddress);
      break;
    }
    case kULong_t: {
      return ConvertAndCopy<unsigned long>(valuestring, targetaddress);
      break;
    }
    case kFloat_t: {
      return ConvertAndCopy<float>(valuestring, targetaddress);
      break;
    }
    case kDouble_t: {
      return ConvertAndCopy<double>(valuestring, targetaddress);
      break;
    }
    case kDouble32_t: {
      return ConvertAndCopy<double>(valuestring, targetaddress);
      break;
    }
    case kchar: {
      unsupp();
      break;
    }
    case kBool_t: {
      return ConvertAndCopy<bool>(valuestring, targetaddress);
      break;
    }
    case kLong64_t: {
      return ConvertAndCopy<long long>(valuestring, targetaddress);
      break;
    }
    case kULong64_t: {
      return ConvertAndCopy<unsigned long long>(valuestring, targetaddress);
      break;
    }
    case kOther_t: {
      unsupp();
      break;
    }
    case kNoType_t: {
      unsupp();
      break;
    }
    case kFloat16_t: {
      unsupp();
      break;
    }
    case kCounter: {
      unsupp();
      break;
    }
    case kCharStar: {
      unsupp();
      // return ConvertAndCopy<char*>(valuestring, targetaddress);
      break;
    }
    case kBits: {
      unsupp();
      break;
    }
    case kVoid_t: {
      unsupp();
      break;
    }
    case kDataTypeAliasUnsigned_t: {
      unsupp();
      break;
    }
    /*
    case kDataTypeAliasSignedChar_t: {
      unsupp();
      break;
    }
    case kNumDataTypes: {
      unsupp();
      break;
  }*/
    default: {
      unsupp();
      break;
    }
  }
  return ConfigurableParam::EParamUpdateStatus::Failed;
}

} // namespace conf
} // namespace o2
