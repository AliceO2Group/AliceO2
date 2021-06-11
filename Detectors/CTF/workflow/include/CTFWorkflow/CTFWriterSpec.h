// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
framework::DataProcessorSpec getCTFWriterSpec(o2::detectors::DetID::mask_t dets, uint64_t run, bool doCTF = true,
                                              bool doDict = false, bool dictPerDet = false, size_t smn = 0, size_t szmx = 0);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFWRITER_SPEC */
