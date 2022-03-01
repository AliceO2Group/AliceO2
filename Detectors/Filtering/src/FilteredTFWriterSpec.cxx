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

/// @file   FilteredTFWriterSpec.cxx

#include "FilteredTFWriterSpec.h"
#include "DataFormatsGlobalTracking/FilteredRecoTF.h"

namespace o2::filtering
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getFilteredTFWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;
  // Spectators for logging
  auto logger = [](const o2::dataformats::FilteredRecoTF& tf) {
    LOG(debug) << "writing filtered TF: " << tf.header.asString();
  };
  return MakeRootTreeWriterSpec("filterer-reco-tf-writer",
                                "o2_filtered_tf.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Filtered reconstructed TF"},
                                BranchDefinition<o2::dataformats::FilteredRecoTF>{InputSpec{"ftf", "GLO", "FILTERED_RECO_TF", 0}, "FTF", logger})();
}

} // end namespace o2::filtering
