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

/// \file   MIDSimulation/PreClusterLabeler.h
/// \brief  PreClusterLabeler for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 April 2019
#ifndef O2_MID_PRECLUSTERLABELER_H
#define O2_MID_PRECLUSTERLABELER_H

#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "MIDClustering/PreCluster.h"
#include "DataFormatsMID/MCLabel.h"

namespace o2
{
namespace mid
{
class PreClusterLabeler
{
 public:
  void process(gsl::span<const PreCluster> preClusters, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, gsl::span<const ROFRecord> rofRecordsPC, gsl::span<const ROFRecord> rofRecordsData);

  const o2::dataformats::MCTruthContainer<MCCompLabel>& getContainer() { return mMCContainer; }

 private:
  bool isDuplicated(size_t idx, const MCLabel& label) const;
  bool addLabel(size_t idx, const MCLabel& label);

  o2::dataformats::MCTruthContainer<MCCompLabel> mMCContainer{}; ///< Labels
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERLABELER_H */
