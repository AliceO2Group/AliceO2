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

/// @file   FilteredTFWriterSpec.h

#ifndef O2_FILTERED_TF_WRITER_H
#define O2_FILTERED_TF_WRITER_H

#include "Framework/DataProcessorSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsGlobalTracking/FilteredRecoTF.h"

using namespace o2::framework;

namespace o2::filtering
{

framework::DataProcessorSpec getFilteredTFWriterSpec();

} // namespace o2::filtering

#endif /* O2_FILTERED_TF_WRITER_H */
