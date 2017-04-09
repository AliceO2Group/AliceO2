/// \file Backend.h
/// \brief Definition of the Backend class
/// \author Charis Kouzinopoulos <charalampos.kouzinopoulos@cern.ch>

#ifndef ALICE_O2_BACKEND_H_
#define ALICE_O2_BACKEND_H_

#include <string>
#include <vector>

#include <FairMQDevice.h>

// Google protocol buffers headers
#include <google/protobuf/stubs/common.h>
#include "request.pb.h"

namespace o2 {
namespace CDB {

class Backend {
public:
  virtual ~Backend()= default;

  /// Pack
  virtual void Pack(const std::string& path, const std::string& key, std::string*& messageString) = 0;

  /// UnPack
  virtual void UnPack(std::unique_ptr<FairMQMessage> msg) = 0;

  /// Serializes a key (and optionally value) to an std::string using Protocol Buffers
  void Serialize(std::string*& messageString, const std::string& key, const std::string& operationType,
                 const std::string& dataSource, const std::string& object = std::string());
};
}
}
#endif
