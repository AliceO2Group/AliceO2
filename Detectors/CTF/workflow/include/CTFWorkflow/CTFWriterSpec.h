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

/// @file   CTFWriterSpec.h

#ifndef O2_CTFWRITER_SPEC
#define O2_CTFWRITER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace ctf
{

/// create a processor spec
framework::DataProcessorSpec getCTFWriterSpec(o2::detectors::DetID::mask_t dets, uint64_t run, const std::string& outType, int verbosity, int reportInterval);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFWRITER_SPEC */
