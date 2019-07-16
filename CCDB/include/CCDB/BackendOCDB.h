// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file BackendOCDB.h
/// \brief Definition of the BackendOCDB class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#ifndef ALICE_O2_BACKENDOCDB_H_
#define ALICE_O2_BACKENDOCDB_H_

#include "CCDB/Backend.h"

#include <iosfwd>
#include <memory>

namespace o2
{
namespace ccdb
{

class BackendOCDB : public Backend
{

 private:
 public:
  BackendOCDB();
  ~BackendOCDB() override = default;

  /// Prepares an object before transmission to CCDB server
  void Pack(const std::string& path, const std::string& key, std::string*& messageString) override;

  /// Parses an incoming message from the CCDB server and prints the metadata of the included object
#if !(defined(__CLING__) || defined(__ROOTCLING__))
  Condition* UnPack(std::unique_ptr<FairMQMessage> msg) override;
#endif
};
}
}
#endif
