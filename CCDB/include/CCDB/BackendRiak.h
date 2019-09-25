// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BackendRiak.h
/// \brief Definition of the BackendRiak class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#ifndef ALICE_O2_BACKENDRIAK_H_
#define ALICE_O2_BACKENDRIAK_H_

#include "CCDB/Backend.h"

namespace o2
{
namespace ccdb
{

class BackendRiak : public Backend
{

 private:
  /// Deserializes a message and stores the value to an std::string using Protocol Buffers
  void Deserialize(const std::string& messageString, std::string& object);

  /// Compresses uncompressed_string to compressed_string using zlib
  void Compress(const std::string& uncompressed_string, std::string& compressed_string);

  /// Decompresses compressed_string to uncompressed_string using zlib
  void Decompress(std::string& uncompressed_string, const std::string& compressed_string);

 public:
  BackendRiak();
  ~BackendRiak() override = default;

  /// Compresses and serializes an object prior to transmission to server
  void Pack(const std::string& path, const std::string& key, std::string*& messageString) override;

  /// Deserializes and uncompresses an incoming message from the CCDB server
#if !(defined(__CLING__) || defined(__ROOTCLING__))
  Condition* UnPack(std::unique_ptr<FairMQMessage> msg) override;
#endif
};
} // namespace ccdb
} // namespace o2
#endif
