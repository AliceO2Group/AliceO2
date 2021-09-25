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

/// @brief meta data of the file produced by O2

#ifndef _ALICEO2_FILE_METADATA_H
#define _ALICEO2_FILE_METADATA_H

#include <string>

namespace o2
{
namespace dataformats
{

struct FileMetaData {
  std::string LHCPeriod{};   // 1, LHC data taking period + detector name, in case of individual detector data stream, required
  std::string lurl{};        // 3, the local EPN path to the CTF or calibration file, required
  std::string type{};        // 4, CTF or calibration; default is CTF, optional
  std::string guid{};        // 7, default is auto-generated, optional
  std::string surl{};        // 8, the remote storage path where we store the data file, optional
  std::string curl{};        // 9, the Grid catalogue path, optional
  std::string md5{};         //10, default the checksum of the lurl file; only filled after a successful transfer, if needed, optional
  std::string xxhash{};      //11, default calculated from the lurl file, only filled after a successful transfer, if needed, optional
  std::string seName{};      //12, default is taken from the configuration file
  std::string seioDaemons{}; //13, default is taken from the configuration file
  std::string priority{};    //14, low or high; default is low
  long run{};                // 2, run number, required
  long ctime{};              // 5, default the timestamp of the lurl file, optional
  size_t size{};             // 6, default the size of the lurl file, optional

  bool fillFileData(const std::string& fname);
  std::string asString() const;
};

std::ostream& operator<<(std::ostream& stream, const FileMetaData& m);

} // namespace dataformats
} // namespace o2

#endif
