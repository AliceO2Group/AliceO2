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
#include <gsl/gsl>
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace its
{
class TimeFrame;
}
namespace itsmft
{
class ROFRecord;
} // namespace itsmft

namespace dataformats
{
class MCCompLabel;
} // namespace dataformats
namespace its3
{
class TopologyDictionary;
class CompClusterExt;

namespace ioutils
{
int loadROFrameDataITS3(its::TimeFrame* tf,
                        gsl::span<o2::itsmft::ROFRecord> rofs,
                        gsl::span<const its3::CompClusterExt> clusters,
                        gsl::span<const unsigned char>::iterator& pattIt,
                        const its3::TopologyDictionary* dict,
                        const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);
}
} // namespace its3
} // namespace o2