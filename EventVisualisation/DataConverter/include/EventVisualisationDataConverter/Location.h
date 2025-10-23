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

///
/// \file    Location.h
/// \author  Julian Myrcha
///

#ifndef O2EVE_LOCATION_H
#define O2EVE_LOCATION_H

#include <string>
#include <fstream>
#include <iosfwd>

namespace o2::event_visualisation
{
struct LocationParams {
  std::string fileName;
  int port = -1;
  int timeout = 100;
  std::string host = "localhost";
  bool toFile = true;
  bool toSocket = true;
};
class Location
{
  std::ofstream* mOut;
  int mClientSocket;
  bool mToFile;
  bool mToSocket;
  std::string mFileName;
  int mPort;
  int mTimeout;
  std::string mHostName;

 public:
  explicit Location(const LocationParams& params)
  {
    this->mFileName = params.fileName;
    this->mToFile = !params.fileName.empty() && params.toFile;
    this->mToSocket = params.port != -1 && params.toSocket;
    this->mOut = nullptr;
    this->mPort = params.port;
    this->mHostName = params.host;
    this->mClientSocket = -1;
    this->mTimeout = params.timeout;
  }
  ~Location()
  {
    close();
  }
  void open();
  void close();
  void write(char* buf, std::streamsize size);
  [[nodiscard]] std::string fileName() const { return this->mFileName; }
  [[nodiscard]] std::string hostName() const { return this->mHostName; }
  [[nodiscard]] int port() const { return this->mPort; }
};
} // namespace o2::event_visualisation

#endif // O2EVE_LOCATION_H
