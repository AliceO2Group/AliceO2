// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DCSConfigObject.h
/// \bried Data format to store DCS configurations

#include <vector>
#include <string>
#include <iostream>
#include <iterator>

#include <TString.h>

namespace o2
{
namespace dcs
{

  typedef std::vector<char> DCSconfigObject_t;

template <typename T>
  inline void addConfigItem(DCSconfigObject_t* configVector, std::string key, const T value) {
  std::string keyValue = key + ":" + std::to_string(value) + ";";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(*configVector));
 }

// explicit specialization for std::string 
template<>
   inline void addConfigItem(DCSconfigObject_t* configVector, std::string key, const std::string value) {
   std::string keyValue = key + ":" + value + ";";
   std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(*configVector));
 }
  
// explicit specialization for char
template<>
   inline void addConfigItem(DCSconfigObject_t* configVector, std::string key, const char value) {
   std::string keyValue = key + ":" + value + ";";
   std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(*configVector));
 }

// explicit specialization for char*
template<>
   inline void addConfigItem(DCSconfigObject_t* configVector, std::string key, const char* value) {
   std::string keyValue = key + ":" + value + ";";
   std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(*configVector));
 }

// explicit specialization for TString
template<>
   inline void addConfigItem(DCSconfigObject_t* configVector, std::string key, const TString value) {
  std::string keyValue = key + ":" + value.Data() + ";";
  std::copy(keyValue.begin(), keyValue.end(), std::back_inserter(*configVector));
 } 

  inline void printDCSConfig(const DCSconfigObject_t& configVector) {
    for (size_t i = 0; i < configVector.size(); ++i) {
      //      LOG(INFO) << i << " --> " << configVector[i];
      std::cout << i << " --> " << configVector[i] << std::endl;
    }
  }
  
} // end dcs
} // end o2
    
