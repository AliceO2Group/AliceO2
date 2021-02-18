// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file IOUtils.h
/// \brief Load pulled clusters, for a given read-out-frame, in a dedicated container
///

#ifndef O2_MFT_IOUTILS_H_
#define O2_MFT_IOUTILS_H_

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>
#include <gsl/gsl>

#include "MFTTracking/Tracker.h"
#include "MFTTracking/ROframe.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ReconstructionDataFormats/BaseCluster.h"

namespace o2
{

class MCCompLabel;

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace itsmft
{
class Cluster;
class CompClusterExt;
class TopologyDictionary;
} // namespace itsmft

namespace mft
{

namespace ioutils
{
constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

int loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& events, gsl::span<const itsmft::CompClusterExt> clusters,
                    gsl::span<const unsigned char>::iterator& pattIt, const itsmft::TopologyDictionary& dict,
                    const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr, const o2::mft::Tracker* tracker = nullptr);

} // namespace ioutils
} // namespace mft
} // namespace o2

#endif /* O2_MFT_IOUTILS_H_ */
