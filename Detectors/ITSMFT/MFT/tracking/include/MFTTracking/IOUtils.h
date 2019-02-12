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

#include "MFTTracking/ROframe.h"

namespace o2
{

class MCCompLabel;

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace ITSMFT
{
class Cluster;
}

namespace MFT
{

namespace IOUtils
{
Int_t loadROFrameData(std::uint32_t roFrame, ROframe& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                      const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
void loadEventData(ROframe& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                   const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
} // namespace IOUtils
} // namespace MFT
} // namespace o2

#endif /* O2_MFT_IOUTILS_H_ */
