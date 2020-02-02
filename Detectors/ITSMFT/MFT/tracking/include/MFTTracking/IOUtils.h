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
#include "DataFormatsITSMFT/ROFRecord.h"

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
}

namespace mft
{

namespace ioutils
{
Int_t loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& event, const std::vector<itsmft::Cluster>* clusters, const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);
void loadEventData(ROframe& event, const std::vector<itsmft::Cluster>* clusters,
                   const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);
} // namespace IOUtils
} // namespace mft
} // namespace o2

#endif /* O2_MFT_IOUTILS_H_ */
