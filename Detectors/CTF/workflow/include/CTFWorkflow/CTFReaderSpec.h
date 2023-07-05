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

/// @file   CTFReaderSpec.h

#ifndef O2_CTFREADER_SPEC
#define O2_CTFREADER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <string>

namespace o2
{
namespace ctf
{
struct CTFReaderInp {
  std::string inpdata{};
  o2::detectors::DetID::mask_t detMask = o2::detectors::DetID::FullMask;
  std::string copyCmd{};
  std::string tffileRegex{};
  std::string remoteRegex{};
  std::string metricChannel{};
  std::string fileIRFrames{};
  std::vector<int> ctfIDs{};
  bool skipSkimmedOutTF = false;
  bool allowMissingDetectors = false;
  bool checkTFLimitBeforeReading = false;
  bool sup0xccdb = false;
  int maxFileCache = 1;
  int64_t delay_us = 0;
  int maxLoops = 0;
  int maxTFs = -1;
  unsigned int subspec = 0;
  unsigned int decSSpecEMC = 0;
  int tfRateLimit = -999;
  size_t minSHM = 0;
};

/// create a processor spec
framework::DataProcessorSpec getCTFReaderSpec(const o2::ctf::CTFReaderInp& inp);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFREADER_SPEC */
