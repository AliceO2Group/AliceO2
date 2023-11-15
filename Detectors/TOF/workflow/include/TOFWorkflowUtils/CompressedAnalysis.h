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

/// @file   CompressedAnalysisTask.h
/// @author Roberto Preghenella
/// @since  2020-09-04
/// @brief  TOF compressed data analysis base class

#ifndef O2_TOF_COMPRESSEDANALYSIS
#define O2_TOF_COMPRESSEDANALYSIS

#include "Headers/RAWDataHeader.h"
#include "DataFormatsTOF/CompressedDataFormat.h"
#include "TOFReconstruction/DecoderBase.h"

using namespace o2::tof::compressed;

namespace o2
{
namespace tof
{

class CompressedAnalysis : public DecoderBaseT<o2::header::RAWDataHeader>
{

 public:
  CompressedAnalysis() = default;
  ~CompressedAnalysis() override = default;

  virtual bool initialize() = 0;
  virtual bool finalize() = 0;

 private:
};

} // namespace tof
} // namespace o2

#endif /* O2_TOF_COMPRESSEDANALYSIS */
