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

/// create a processor spec
framework::DataProcessorSpec getCTFReaderSpec(o2::detectors::DetID::mask_t dets, const std::string& inp, int loop, int delayMUS);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFREADER_SPEC */
