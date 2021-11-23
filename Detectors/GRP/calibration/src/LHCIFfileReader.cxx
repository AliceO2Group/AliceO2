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

#include "GRPCalibration/LHCIFfileReader.h"
#include <fstream>

namespace o2
{
namespace grp
{

void LHCIFfileReader::loadLHCIFfile(const std::string& fileName)
{
  // load the LHC IF file into a string
  if (o2::utils::Str::pathExists(fileName)) {
    std::string expandedFileName = o2::utils::Str::getFullPath(fileName);
    std::ifstream ifs(expandedFileName.c_str());
    if (ifs) {
      ifs.seekg(0, std::ios::end);
      const auto size = ifs.tellg();
      mFileBuffStr.resize(size);
      ifs.seekg(0);
      ifs.read(&mFileBuffStr[0], size);
      ifs.close();
    }
  } else {
    LOG(error) << "File " << fileName << " does not exist!";
  }
}

//_________________________________________________________________

void LHCIFfileReader::loadLHCIFfile(gsl::span<const char> configBuf)
{
  // load the LHC IF file into a string from a buffer
  mFileBuffStr.assign(configBuf.data());
}

//_________________________________________________________________

} // end namespace grp
} // end namespace o2
