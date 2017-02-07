/// \file BackendOCDB.h
/// \brief Definition of the BackendOCDB class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#ifndef ALICE_O2_BACKENDOCDB_H_
#define ALICE_O2_BACKENDOCDB_H_

#include "CCDB/Backend.h"

#include <iostream>
#include <memory>

namespace AliceO2 {
namespace CDB {

class BackendOCDB : public Backend {

private:
public:
  BackendOCDB();
  virtual ~BackendOCDB(){};

  /// Prepares an object before transmission to CCDB server
  void Pack(const std::string& path, const std::string& key, std::string*& messageString);

  /// Parses an incoming message from the CCDB server and prints the metadata of the included object
  void UnPack(std::unique_ptr<FairMQMessage> msg);
};
}
}
#endif
