// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//first version 8/2018, Sandro Wenzel

#include "SimConfig/ConfigurableParam.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <FairLogger.h>
#include <typeinfo>
#include <cassert>
#include "TDataType.h"
#include "TFile.h"

namespace o2
{
namespace conf
{

std::vector<ConfigurableParam*>* ConfigurableParam::sRegisteredParamClasses = nullptr;
boost::property_tree::ptree* ConfigurableParam::sPtree = nullptr;
std::map<std::string, std::pair<int, void*>>* ConfigurableParam::sKeyToStorageMap = nullptr;
std::map<std::string, ConfigurableParam::EParamProvenance>* ConfigurableParam::sValueProvenanceMap = nullptr;

bool ConfigurableParam::sIsFullyInitialized = false;
bool ConfigurableParam::sRegisterMode = true;

void ConfigurableParam::writeINI(std::string filename)
{
  initPropertyTree(); // update the boost tree before writing
  boost::property_tree::write_ini(filename, *sPtree);
}

void ConfigurableParam::writeJSON(std::string filename)
{
  initPropertyTree(); // update the boost tree before writing
  boost::property_tree::write_json(filename, *sPtree);
}

void ConfigurableParam::initPropertyTree()
{
  sPtree->clear();
  for (auto p : *sRegisteredParamClasses) {
    p->putKeyValues(sPtree);
  }
}

void ConfigurableParam::printAllKeyValuePairs()
{
  std::cout << "####\n";
  for (auto p : *sRegisteredParamClasses) {
    p->printKeyValues(true);
  }
  std::cout << "----\n";
}

// evidently this could be a local file or an OCDB server
// ... we need to generalize this ... but ok for demonstration purposes
void ConfigurableParam::toCCDB(std::string filename)
{
  TFile file(filename.c_str(), "RECREATE");
  for (auto p : *sRegisteredParamClasses) {
    p->serializeTo(&file);
  }
  file.Close();
}

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

ConfigurableParam::ConfigurableParam()
{
  if (sRegisteredParamClasses == nullptr) {
    sRegisteredParamClasses = new std::vector<ConfigurableParam*>;
  }
  if (sPtree == nullptr) {
    sPtree = new boost::property_tree::ptree;
  }
  if (sKeyToStorageMap == nullptr) {
    sKeyToStorageMap = new std::map<std::string, std::pair<int, void*>>;
  }
  if (sValueProvenanceMap == nullptr) {
    sValueProvenanceMap = new std::map<std::string, ConfigurableParam::EParamProvenance>;
  }
  if (sRegisterMode == true) {
    sRegisteredParamClasses->push_back(this);
  }
}

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

void ConfigurableParam::printAllRegisteredParamNames()
{
  for (auto p : *sRegisteredParamClasses) {
    std::cout << p->getName() << "\n";
  }
}

void ConfigurableParam::updateFromString(std::string configstring)
{
  if (!sIsFullyInitialized) {
    initialize();
  }
  using Tokenizer = boost::tokenizer<boost::char_separator<char>>;
  boost::char_separator<char> tokensep{ "," };
  boost::char_separator<char> keyvaluesep{ "=" };
  Tokenizer tok{ configstring, tokensep };
  for (const auto& t : tok) {
    Tokenizer keyvaluetokenizer{ t, keyvaluesep };
    std::string extractedkey;
    std::string extractedvalue;
    int counter = 0;
    // TODO: make sure format is correct with a regular expression
    for (const auto& s : keyvaluetokenizer) {
      if (counter == 1) {
        extractedvalue = s;
      }
      if (counter == 0) {
        extractedkey = s;
      }
      counter++;
    }
    // here we have key and value
    // ... check whether such a key exists
    auto optional = sPtree->get_optional<std::string>(extractedkey);
    if (optional.is_initialized()) {
      LOG(INFO) << "FOUND KEY ... and the current value is " << optional.get();

      assert(sKeyToStorageMap->find(extractedkey) != sKeyToStorageMap->end());

      setValue(extractedkey, extractedvalue);
    } else {
      LOG(FATAL) << "Configuration key " << extractedkey << " not valid ... (abort)";
      continue;
    }
  }
}

void unsupp() { std::cerr << "currently unsupported\n"; }

template <typename T>
bool isMemblockDifferent(void const* block1, void const* block2)
{
  // loop over thing in elements of bytes
  for (int i = 0; i < sizeof(T) / sizeof(char); ++i) {
    if (((char*)block1)[i] != ((char*)block2)[i]) {
      return false;
    }
  }
  return true;
}

// copies data from one place to other and returns
// true of data was actually changed
template <typename T>
bool Copy(void const* addr, void* targetaddr)
{
  if (isMemblockDifferent<T>(addr, targetaddr)) {
    std::memcpy(targetaddr, addr, sizeof(T));
    return true;
  }
  return false;
}

bool ConfigurableParam::updateThroughStorageMap(std::string mainkey, std::string subkey, std::type_info const& tinfo,
                                                void* addr)
{
  // check if key_exists
  auto key = mainkey + "." + subkey;
  auto iter = sKeyToStorageMap->find(key);
  if (iter == sKeyToStorageMap->end()) {
    LOG(WARN) << "Cannot update parameter " << key << " not found";
    return false;
  }

  // the type we need to convert to
  int type = TDataType::GetType(tinfo);

  // check that type matches
  if (iter->second.first != type) {
    LOG(WARN) << "Types do not match; cannot update value";
    return false;
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
  return false;
}

template <typename T>
bool ConvertAndCopy(std::string const& valuestring, void* targetaddr)
{
  auto addr = boost::lexical_cast<T>(valuestring);
  if (!isMemblockDifferent<T>(targetaddr, (void*)&addr)) {
    std::memcpy(targetaddr, (void*)&addr, sizeof(T));
    return true;
  }
  return false;
}

bool ConfigurableParam::updateThroughStorageMapWithConversion(std::string key, std::string valuestring)
{
  // check if key_exists
  auto iter = sKeyToStorageMap->find(key);
  if (iter == sKeyToStorageMap->end()) {
    LOG(WARN) << "Cannot update parameter" << key << " (parameter not found) ";
    return false;
  }

  // the type (aka ROOT::EDataType which the type identification in the map) we need to convert to
  int targettype = iter->second.first;

  auto targetaddress = iter->second.second;
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
  return false;
}

} // namespace conf
} // namespace o2
