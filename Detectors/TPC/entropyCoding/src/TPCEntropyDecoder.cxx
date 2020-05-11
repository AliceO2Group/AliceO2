// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCEntropyDecoder.cxx
/// @author Michael Lettrich
/// @since  Apr 30, 2020
/// @brief

#include "TPCEntropyCoding/TPCEntropyDecoder.h"

namespace o2
{
namespace tpc
{

std::unique_ptr<EncodedClusters> TPCEntropyDecoder::fromTree(TTree& tree)
{
  auto ec = std::make_unique<o2::tpc::EncodedClusters>();

  // link adresses
  assert(tree.GetBranch("Header"));
  tree.SetBranchAddress("Header", &ec->header);

  assert(tree.GetBranch("Counters"));
  tree.SetBranchAddress("Counters", &ec->counters);

  assert(tree.GetBranch("Metadata"));
  tree.SetBranchAddress("Metadata", &ec->metadata);

  for (size_t i = 0; i < ec->NUM_ARRAYS; ++i) {
    const auto name = ec->NAMES[i];
    const auto dictName = std::string(name) + "Dict";

    assert(tree.GetBranch(name));
    tree.SetBranchAddress(name, &ec->buffers[i]);
    assert(tree.GetBranch(dictName.c_str()));

    tree.SetBranchAddress(dictName.c_str(), &ec->dicts[i]);
  }

  // poplulate tree
  assert(tree.GetEntries() == 1);
  tree.GetEntry(0);

  LOG(INFO) << "finished reading in EncodedClusters";

  return std::move(ec);
}

std::unique_ptr<CompressedClusters> TPCEntropyDecoder::initCompressedClusters(const EncodedClusters& ec)
{
  auto cc = std::make_unique<CompressedClusters>();

  // set counters
  assert(ec.counters);
  const auto& counters = *ec.counters;

  cc->nTracks = counters.nTracks;
  cc->nAttachedClusters = counters.nAttachedClusters;
  cc->nUnattachedClusters = counters.nUnattachedClusters;
  cc->nAttachedClustersReduced = counters.nAttachedClustersReduced;
  cc->nSliceRows = counters.nSliceRows;

  LOG(INFO) << "finished initialization of CompressedClusters";
  return std::move(cc);
}

} // namespace tpc
} // namespace o2
