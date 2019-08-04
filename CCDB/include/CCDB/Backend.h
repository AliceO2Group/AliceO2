// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Backend.h
/// \brief Definition of the Backend class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#ifndef ALICE_O2_BACKEND_H_
#define ALICE_O2_BACKEND_H_

#include <string>
#include <vector>

#if !(defined(__CLING__) || defined(__ROOTCLING__))
#include <FairMQDevice.h>
#endif

// Google protocol buffers headers
#include <google/protobuf/stubs/common.h>

namespace o2
{
namespace ccdb
{

class Condition;

class Backend
{
 public:
  virtual ~Backend() = default;

  /// Pack
  virtual void Pack(const std::string& path, const std::string& key, std::string*& messageString) = 0;

  /// UnPack
#if !(defined(__CLING__) || defined(__ROOTCLING__))
  virtual Condition* UnPack(std::unique_ptr<FairMQMessage> msg) = 0;
#endif

  /// Serializes a key (and optionally value) to an std::string using Protocol Buffers
  void Serialize(std::string*& messageString, const std::string& key, const std::string& operationType,
                 const std::string& dataSource, const std::string& object = std::string());
};
} // namespace ccdb
} // namespace o2
#endif
