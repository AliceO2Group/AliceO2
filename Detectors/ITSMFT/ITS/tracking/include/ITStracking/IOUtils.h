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

#ifndef TRACKINGITSU_INCLUDE_EVENTLOADER_H_
#define TRACKINGITSU_INCLUDE_EVENTLOADER_H_

#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include "ITStracking/Configuration.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/json.h"
#include "ITStracking/Label.h"
#include "ITStracking/Road.h"

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

namespace ITS
{

void to_json(nlohmann::json& j, const TrackingParameters& par);
void from_json(const nlohmann::json& j, TrackingParameters& par);
void to_json(nlohmann::json& j, const MemoryParameters& par);
void from_json(const nlohmann::json& j, MemoryParameters& par);
void to_json(nlohmann::json& j, const IndexTableParameters& par);
void from_json(const nlohmann::json& j, IndexTableParameters& par);

namespace IOUtils
{
void loadConfigurations(const std::string&);
std::vector<ROframe> loadEventData(const std::string&);
void loadEventData(ROframe& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                   const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
int loadROFrameData(std::uint32_t roFrame, ROframe& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                    const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
std::vector<std::unordered_map<int, Label>> loadLabels(const int, const std::string&);
void writeRoadsReport(std::ofstream&, std::ofstream&, std::ofstream&, const std::vector<std::vector<Road>>&,
                      const std::unordered_map<int, Label>&);
} // namespace IOUtils
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_EVENTLOADER_H_ */
