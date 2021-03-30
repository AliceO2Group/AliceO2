// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCAverageGroupIDCSpec.h
/// \brief TPC merging and averaging of IDCs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 16, 2021

#ifndef O2_TPCAVERAGEGROUPIDCSPEC_H
#define O2_TPCAVERAGEGROUPIDCSPEC_H

#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/IDCAverageGroup.h"
#include "TPCWorkflow/TPCIntegrateIDCSpec.h"
#include "TPCCalibration/IDCGroupingParameter.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCAverageGroupIDCDevice : public o2::framework::Task
{
 public:
  TPCAverageGroupIDCDevice(const int lane, const std::vector<uint32_t>& crus, const bool debug = false)
    : mLane{lane}, mCRUs{crus}, mDebug{debug}
  {
    auto& paramIDCGroup = ParameterIDCGroup::Instance();
    for (const auto& cru : mCRUs) {
      const CRU cruTmp(cru);
      const unsigned int reg = cruTmp.region();
      mIDCs.emplace(cru, IDCAverageGroup(paramIDCGroup.GroupPads[reg], paramIDCGroup.GroupRows[reg], paramIDCGroup.GroupLastRowsThreshold[reg], paramIDCGroup.GroupLastPadsThreshold[reg], reg, cruTmp.sector()));
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(info, "averaging and grouping IDCs for one TF for CRUs {} to {} using {} threads", mCRUs.front(), mCRUs.back(), mIDCs.begin()->second.getNThreads());

    for (int i = 0; i < mCRUs.size(); ++i) {
      const DataRef ref = pc.inputs().getByPos(i);
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification >> 7;
      mIDCs[cru].setIDCs(pc.inputs().get<std::vector<float>>(ref));
      mIDCs[cru].processIDCs();

      // send the output for one CRU for one TF
      sendOutput(pc.outputs(), cru);
    }

    if (mDebug) {
      TFile fOut(fmt::format("IDCGroup_{}_tf_{}.root", mLane, o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->tfCounter).data(), "RECREATE");
      for (int i = 0; i < mCRUs.size(); ++i) {
        const DataRef ref = pc.inputs().getByPos(i);
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        fOut.WriteObject(&mIDCs[cru], fmt::format("CRU_{}", cru).data());
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  /// return data description for IDC Group
  static constexpr header::DataDescription getDataDescriptionIDCGroup() { return header::DataDescription{"IDCGROUP"}; }

  /// return data description for 1D IDCs
  static constexpr header::DataDescription getDataDescription1DIDC() { return header::DataDescription{"1DIDC"}; }

 private:
  const int mLane{};                                         ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                       ///< CRUs to process in this instance
  const bool mDebug{};                                       ///< dump IDCs to tree for debugging
  std::unordered_map<unsigned int, IDCAverageGroup> mIDCs{}; ///< object for averaging and grouping of the IDCs

  void sendOutput(DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDCGroup(), subSpec, Lifetime::Timeframe}, mIDCs[cru].getIDCGroup().getData());

    // calculate 1D-IDCs = sum over 2D-IDCs as a function of the integration interval
    const std::vector<float> vec1DIDCs = mIDCs[cru].getIDCGroup().get1DIDCs();
    output.snapshot(Output{gDataOriginTPC, getDataDescription1DIDC(), subSpec, Lifetime::Timeframe}, vec1DIDCs);
  }
};

DataProcessorSpec getTPCAverageGroupIDCSpec(const int ilane, const std::vector<uint32_t>& crus, const bool debug = false)
{
  std::vector<OutputSpec> outputSpecs;
  std::vector<InputSpec> inputSpecs;
  outputSpecs.reserve(crus.size());
  inputSpecs.reserve(crus.size());

  for (const auto& cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    inputSpecs.emplace_back(InputSpec{"idcs", gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), subSpec, Lifetime::Timeframe});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescriptionIDCGroup(), subSpec});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCAverageGroupIDCDevice::getDataDescription1DIDC(), subSpec});
  }

  const auto id = fmt::format("tpc-averagegroup-idc-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCAverageGroupIDCDevice>(ilane, crus, debug)},
  }; // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
