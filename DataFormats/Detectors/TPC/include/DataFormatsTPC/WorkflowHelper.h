// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file WorkflowHelper.h
/// @brief Helper class to obtain TPC clusters / digits / labels from DPL
/// @author David Rohr

#ifndef WORKFLOWHELPER_H
#define WORKFLOWHELPER_H

#include <memory>
#include "Framework/ProcessingContext.h"
#include "Framework/DataRefUtils.h"
#include <Framework/InputRecord.h>
#include "Framework/InputRecordWalker.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/Digit.h"
#include "DataFormatsTPC/TPCSectorHeader.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"

// NOTE: The DataFormatsTPC package does not have all required dependencies for these includes.
// The users of these headers should add them by themselves, in order to avoid making
// DataFormatsTPC depend on the Framework.

namespace o2
{
namespace tpc
{
namespace internal
{
struct InputRef {
  o2::framework::DataRef data;
  o2::framework::DataRef labels;
};
struct getWorkflowTPCInput_ret_internal {
  std::map<int, InputRef> inputrefs;
  std::vector<o2::dataformats::ConstMCLabelContainerView> mcInputs;
  std::vector<gsl::span<const char>> inputs;
  std::array<int, constants::MAXSECTOR> inputDigitsMCIndex;
  std::vector<o2::dataformats::ConstMCLabelContainerView> inputDigitsMC;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clustersMCBuffer;
};
struct getWorkflowTPCInput_ret {
  getWorkflowTPCInput_ret_internal internal;
  std::array<gsl::span<const o2::tpc::Digit>, constants::MAXSECTOR> inputDigits;
  std::array<const o2::dataformats::ConstMCLabelContainerView*, constants::MAXSECTOR> inputDigitsMCPtrs;
  ClusterNativeAccess clusterIndex;
};
} // namespace internal

static auto getWorkflowTPCInput(o2::framework::ProcessingContext& pc, int verbosity = 0, bool do_mcLabels = false, bool do_clusters = true, unsigned long tpcSectorMask = 0xFFFFFFFFF, bool do_digits = false)
{
  auto retVal = std::make_unique<internal::getWorkflowTPCInput_ret>();

  if (do_clusters && do_digits) {
    throw std::invalid_argument("Currently cannot process both clusters and digits");
  }

  if (do_mcLabels) {
    std::vector<o2::framework::InputSpec> filter = {
      {"check", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "DIGITSMCTR"}, o2::framework::Lifetime::Timeframe},
      {"check", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CLNATIVEMCLBL"}, o2::framework::Lifetime::Timeframe},
    };
    unsigned long recvMask = 0;
    for (auto const& ref : o2::framework::InputRecordWalker(pc.inputs(), filter)) {
      auto const* sectorHeader = o2::framework::DataRefUtils::getHeader<TPCSectorHeader*>(ref);
      if (sectorHeader == nullptr) {
        // FIXME: think about error policy
        LOG(ERROR) << "sector header missing on header stack";
        return retVal;
      }
      const int sector = sectorHeader->sector();
      if (sector < 0) {
        continue;
      }
      if (recvMask & sectorHeader->sectorBits) {
        throw std::runtime_error("can only have one MC data set per sector");
      }
      recvMask |= sectorHeader->sectorBits;
      retVal->internal.inputrefs[sector].labels = ref;
      if (do_digits) {
        retVal->internal.inputDigitsMCIndex[sector] = retVal->internal.inputDigitsMC.size();
        retVal->internal.inputDigitsMC.emplace_back(o2::dataformats::ConstMCLabelContainerView(pc.inputs().get<gsl::span<char>>(ref)));
      }
    }
    if (recvMask != tpcSectorMask) {
      throw std::runtime_error("Incomplete set of MC labels received");
    }
    if (do_digits) {
      for (unsigned int i = 0; i < constants::MAXSECTOR; i++) {
        if (verbosity >= 1) {
          LOG(INFO) << "GOT MC LABELS FOR SECTOR " << i << " -> " << retVal->internal.inputDigitsMC[retVal->internal.inputDigitsMCIndex[i]].getNElements();
        }
        retVal->inputDigitsMCPtrs[i] = &retVal->internal.inputDigitsMC[retVal->internal.inputDigitsMCIndex[i]];
      }
    }
  }

  if (do_clusters || do_digits) {
    std::vector<o2::framework::InputSpec> filter = {
      {"check", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "DIGITS"}, o2::framework::Lifetime::Timeframe},
      {"check", o2::framework::ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, "CLUSTERNATIVE"}, o2::framework::Lifetime::Timeframe},
    };
    unsigned long recvMask = 0;
    for (auto const& ref : o2::framework::InputRecordWalker(pc.inputs(), filter)) {
      auto const* sectorHeader = o2::framework::DataRefUtils::getHeader<TPCSectorHeader*>(ref);
      if (sectorHeader == nullptr) {
        throw std::runtime_error("sector header missing on header stack");
      }
      const int sector = sectorHeader->sector();
      if (sector < 0) {
        continue;
      }
      if (recvMask & sectorHeader->sectorBits) {
        throw std::runtime_error("can only have one cluster data set per sector");
      }
      recvMask |= sectorHeader->sectorBits;
      retVal->internal.inputrefs[sector].data = ref;
      if (do_digits) {
        retVal->inputDigits[sector] = pc.inputs().get<gsl::span<o2::tpc::Digit>>(ref);
        if (verbosity >= 1) {
          LOG(INFO) << "GOT DIGITS SPAN FOR SECTOR " << sector << " -> " << retVal->inputDigits[sector].size();
        }
      }
    }
    if (recvMask != tpcSectorMask) {
      throw std::runtime_error("Incomplete set of clusters/digits received");
    }

    for (auto const& refentry : retVal->internal.inputrefs) {
      auto& sector = refentry.first;
      auto& ref = refentry.second.data;
      if (do_clusters) {
        if (ref.payload == nullptr) {
          // skip zero-length message
          continue;
        }
        if (refentry.second.labels.header != nullptr && refentry.second.labels.payload != nullptr) {
          retVal->internal.mcInputs.emplace_back(o2::dataformats::ConstMCLabelContainerView(pc.inputs().get<gsl::span<char>>(refentry.second.labels)));
        }
        retVal->internal.inputs.emplace_back(gsl::span(ref.payload, o2::framework::DataRefUtils::getPayloadSize(ref)));
      }
      if (verbosity > 1) {
        LOG(INFO) << "received " << *(ref.spec) << ", size " << o2::framework::DataRefUtils::getPayloadSize(ref) << " for sector " << sector;
      }
    }
  }

  if (do_clusters) {
    memset(&retVal->clusterIndex, 0, sizeof(retVal->clusterIndex));
    ClusterNativeHelper::Reader::fillIndex(retVal->clusterIndex, retVal->internal.clusterBuffer, retVal->internal.clustersMCBuffer, retVal->internal.inputs, retVal->internal.mcInputs, [&tpcSectorMask](auto& index) { return tpcSectorMask & (1ul << index); });
  }

  return retVal;
}

} // namespace tpc
} // namespace o2
#endif // WORKFLOWHELPER_H
