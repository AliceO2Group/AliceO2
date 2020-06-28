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
/// \brief
///

#ifndef TRACKINGEC0__INCLUDE_EVENTLOADER_H_
#define TRACKINGEC0__INCLUDE_EVENTLOADER_H_

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsEndCaps/TopologyDictionary.h"
#include "EC0tracking/Configuration.h"
#include "EC0tracking/ROframe.h"
#include "EC0tracking/Label.h"
#include "EC0tracking/Road.h"
#include "EndCapsBase/SegmentationAlpide.h"
#include "ReconstructionDataFormats/BaseCluster.h"

namespace o2
{

class MCCompLabel;

namespace dataformats
{
template <typename T>
class MCTruthContainer;
}

namespace endcaps
{
//class o2::itsmft::Cluster;
//class o2::itsmft::CompClusterExt;
class TopologyDictionary;
} // namespace endcaps

namespace ecl
{

namespace ioutils
{
constexpr float DefClusErrorRow = o2::endcaps::SegmentationAlpide::PitchRow * 0.5;
constexpr float DefClusErrorCol = o2::endcaps::SegmentationAlpide::PitchCol * 0.5;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

void loadConfigurations(const std::string&);
std::vector<ROframe> loadEventData(const std::string&);
void loadEventData(ROframe& events, gsl::span<const itsmft::Cluster> clusters,
                   const o2::dataformats::MCTruthContainer<MCCompLabel>* clsLabels = nullptr);
void loadEventData(ROframe& events, gsl::span<const o2::itsmft::CompClusterExt> clusters,
                   gsl::span<const unsigned char>::iterator& pattIt, const o2::endcaps::TopologyDictionary& dict,
                   const dataformats::MCTruthContainer<MCCompLabel>* clsLabels = nullptr);
int loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& events, gsl::span<const itsmft::Cluster> clusters,
                    const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
int loadROFrameData(const o2::itsmft::ROFRecord& rof, ROframe& events, gsl::span<const itsmft::CompClusterExt> clusters,
                    gsl::span<const unsigned char>::iterator& pattIt, const o2::endcaps::TopologyDictionary& dict,
                    const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
void generateSimpleData(ROframe& event, const int phiDivs, const int zDivs);

void convertCompactClusters(gsl::span<const itsmft::CompClusterExt> clusters,
                            gsl::span<const unsigned char>::iterator& pattIt,
                            std::vector<o2::BaseCluster<float>>& output,
                            const o2::endcaps::TopologyDictionary& dict);

std::vector<std::unordered_map<int, Label>> loadLabels(const int, const std::string&);
void writeRoadsReport(std::ofstream&, std::ofstream&, std::ofstream&, const std::vector<std::vector<Road>>&,
                      const std::unordered_map<int, Label>&);
} // namespace ioutils
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_EVENTLOADER_H_ */
