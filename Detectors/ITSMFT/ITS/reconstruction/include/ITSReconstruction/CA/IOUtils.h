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

#include "ITSReconstruction/CA/Configuration.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/json.h"
#include "ITSReconstruction/CA/Label.h"
#include "ITSReconstruction/CA/Road.h"

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
namespace CA
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
std::vector<Event> loadEventData(const std::string&);
void loadEventData(Event& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                   const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
int loadROFrameData(std::uint32_t roFrame, Event& events, const std::vector<ITSMFT::Cluster>* mClustersArray,
                    const dataformats::MCTruthContainer<MCCompLabel>* mClsLabels = nullptr);
std::vector<std::unordered_map<int, Label>> loadLabels(const int, const std::string&);
void writeRoadsReport(std::ofstream&, std::ofstream&, std::ofstream&, const std::vector<std::vector<Road>>&,
                      const std::unordered_map<int, Label>&);
} // namespace IOUtils
} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_EVENTLOADER_H_ */
