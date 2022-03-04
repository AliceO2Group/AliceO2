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

/// \file TPCFLPIDCSpec.h
/// \brief TPC device for processing on FLPs
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Apr 16, 2021

#ifndef O2_TPCFLPIDCSPEC_H
#define O2_TPCFLPIDCSPEC_H

#include <vector>
#include <deque>
#include <algorithm>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/IDCAverageGroup.h"
#include "TPCWorkflow/TPCIntegrateIDCSpec.h"
#include "TPCCalibration/IDCGroupingParameter.h"
#include "Framework/CCDBParamSpec.h"
#include "TPCCalibration/IDCContainer.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCWorkflow/ProcessingHelpers.h"

#include "TKey.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

template <class Type>
struct IDCFLPIDCStruct;

/// dummy class for template specialization
class TPCFLPIDCDeviceGroup;
class TPCFLPIDCDeviceNoGroup;

template <>
struct IDCFLPIDCStruct<TPCFLPIDCDeviceGroup> {
  /// \return returns grouped and averaged IDC values
  const auto& getData(const unsigned int cru) { return mIDCs[cru].getIDCGroup().getData(); }

  std::unordered_map<unsigned int, IDCAverageGroup<IDCAverageGroupCRU>> mIDCs{}; ///< object for averaging and grouping of the IDCs
};

template <>
struct IDCFLPIDCStruct<TPCFLPIDCDeviceNoGroup> {
  /// \return returns IDC values
  const auto& getData(const unsigned int cru) { return mIDCs[cru]; }

  std::unordered_map<unsigned int, std::vector<float>> mIDCs{}; ///< object for storing the IDCs
};

template <class Type>
class TPCFLPIDCDevice : public o2::framework::Task
{
 public:
  TPCFLPIDCDevice(const int lane, const std::vector<uint32_t>& crus, const unsigned int rangeIDC, const bool debug = false, const bool loadFromFile = false, const bool loadStatusMap = false)
    : mLane{lane}, mCRUs{crus}, mRangeIDC{rangeIDC}, mDebug{debug}, mLoadFromFile{loadFromFile}, mLoadPadMapCCDB{loadStatusMap}, mOneDIDCs(rangeIDC)
  {
    if constexpr (std::is_same_v<Type, TPCFLPIDCDeviceGroup>) {
      auto& paramIDCGroup = ParameterIDCGroup::Instance();
      for (const auto& cru : mCRUs) {
        const CRU cruTmp(cru);
        const unsigned int reg = cruTmp.region();
        mIDCStruct.mIDCs.emplace(cru, IDCAverageGroup<IDCAverageGroupCRU>(paramIDCGroup.groupPads[reg], paramIDCGroup.groupRows[reg], paramIDCGroup.groupLastRowsThreshold[reg], paramIDCGroup.groupLastPadsThreshold[reg], paramIDCGroup.groupPadsSectorEdges, cru));
      }
    }

    for (const auto& cru : mCRUs) {
      mBuffer1DIDCs.emplace(cru, std::deque<std::vector<float>>(std::ceil(static_cast<float>(mRangeIDC) / mMinIDCsPerTF), std::vector<float>(mMinIDCsPerTF, 0)));
    }
  }

  void init(o2::framework::InitContext& ic) final
  {
    if constexpr (std::is_same_v<Type, TPCFLPIDCDeviceGroup>) {
      if (mLoadFromFile) {
        const char* fName = "IDCGroup.root";
        TFile fInp(fName, "READ");
        for (TObject* keyAsObj : *fInp.GetListOfKeys()) {
          const auto key = dynamic_cast<TKey*>(keyAsObj);
          const char* name = key->GetName();
          LOGP(info, "Key name: {} Type: {}", name, key->GetClassName());

          if (std::strcmp(o2::tpc::IDCAverageGroup<IDCAverageGroupCRU>::Class()->GetName(), key->GetClassName()) != 0) {
            LOGP(info, "skipping object. wrong class.");
            continue;
          }
          IDCAverageGroup<IDCAverageGroupCRU>* idcavg = (IDCAverageGroup<IDCAverageGroupCRU>*)fInp.Get(name);
          unsigned int cru = idcavg->getSector() * Mapper::NREGIONS + idcavg->getRegion();
          // check cru
          if (std::find(mCRUs.begin(), mCRUs.end(), cru) != mCRUs.end()) {
            mIDCStruct.mIDCs[cru].processIDCs(mPadFlagsMap);
            mIDCStruct.mIDCs[cru].setFromFile(fName, name);
          }
          delete idcavg;
        }
      }
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    if constexpr (std::is_same_v<Type, TPCFLPIDCDeviceGroup>) {
      LOGP(info, "averaging and grouping IDCs for TF {} for CRUs {} to {} using {} threads", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back(), mIDCStruct.mIDCs.begin()->second.getNThreads());
    } else {
      LOGP(info, "skipping grouping of IDCs for TF {} for CRUs {} to {}", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back());
    }

    // retrieving map containing the status flags for the static outlier
    if (mLoadPadMapCCDB) {
      updateTimeDependentParams(pc);
    }

    if (!mLoadFromFile) {
      for (int i = 0; i < mCRUs.size() + mLoadPadMapCCDB; ++i) {
        const DataRef ref = pc.inputs().getByPos(i);
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const auto descr = tpcCRUHeader->dataDescription;
        if (TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim) == descr) {
          const int cru = tpcCRUHeader->subSpecification >> 7;
          if constexpr (std::is_same_v<Type, TPCFLPIDCDeviceGroup>) {
            mIDCStruct.mIDCs[cru].setIDCs(pc.inputs().get<std::vector<float>>(ref));
            mIDCStruct.mIDCs[cru].processIDCs(mPadFlagsMap);
          } else {
            mIDCStruct.mIDCs[cru] = pc.inputs().get<std::vector<float>>(ref);
          }
          // send the output for one CRU for one TF
          sendOutput(pc.outputs(), cru);
        }
      }
    } else {
      for (int i = 0; i < mCRUs.size(); ++i) {
        sendOutput(pc.outputs(), mCRUs[i]);
      }
    }

    if (mDebug) {
      TFile fOut(fmt::format("IDCGroup_{}_tf_{}.root", mLane, processing_helpers::getCurrentTF(pc)).data(), "RECREATE");
      for (int i = 0; i < mCRUs.size(); ++i) {
        const DataRef ref = pc.inputs().getByPos(i);
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        fOut.WriteObject(&mIDCStruct.mIDCs[cru], fmt::format("CRU_{}", cru).data());
      }
    }
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADSTATUSMAP", 0)) {
      LOGP(info, "Updating pad status from CCDB");
      mPadFlagsMap = static_cast<o2::tpc::CalDet<PadFlags>*>(obj);
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

  /// return data description for buffered 1D IDCs for EPNs
  static constexpr header::DataDescription getDataDescription1DIDCEPN() { return header::DataDescription{"1DIDCEPN"}; }

  /// set minimum IDCs which will be received per TF
  /// \param nMinIDCsPerTF minimal number of IDCs per TF
  static void setMinIDCsPerTF(const unsigned int nMinIDCsPerTF) { mMinIDCsPerTF = nMinIDCsPerTF; }

  /// \return returns the minimum IDCs which will be received per TF
  static unsigned int getMinIDCsPerTF() { return mMinIDCsPerTF; }

 private:
  inline static unsigned int mMinIDCsPerTF{10};                                     ///< minimum number of IDCs per TF (depends only on number of orbits per TF and (fixed) number of orbits per integration interval). 11 for 128 orbits per TF and 21 for 256 orbits per TF
  const int mLane{};                                                                ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                                              ///< CRUs to process in this instance
  const unsigned int mRangeIDC{};                                                   ///< number of IDCs used for the calculation of fourier coefficients
  const bool mDebug{};                                                              ///< dump IDCs to tree for debugging
  const bool mLoadFromFile{};                                                       ///< load ungrouped IDCs from file
  const bool mLoadPadMapCCDB{};                                                     ///< load status map for pads from CCDB
  std::vector<float> mOneDIDCs{};                                                   ///< 1D-IDCs which will be send to the EPNs
  IDCFLPIDCStruct<Type> mIDCStruct{};                                               ///< object for averaging and grouping of the IDCs
  std::unordered_map<unsigned int, std::deque<std::vector<float>>> mBuffer1DIDCs{}; ///< buffer for 1D-IDCs. The buffered 1D-IDCs for n TFs will be send to the EPNs for synchronous reco. Zero initialized to avoid empty first TFs!
  CalDet<PadFlags>* mPadFlagsMap{nullptr};                                          ///< status flag for each pad (i.e. if the pad is dead)

  /// update the time dependent parameters if they have changed (i.e. update the pad status map)
  void updateTimeDependentParams(ProcessingContext& pc) { pc.inputs().get<o2::tpc::CalDet<PadFlags>*>("tpcpadmap").get(); }

  void sendOutput(DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};

    std::vector<float> vec1DIDCs;
    if constexpr (std::is_same_v<Type, TPCFLPIDCDeviceGroup>) {
      // calculate 1D-IDCs = sum over 2D-IDCs as a function of the integration interval
      vec1DIDCs = mIDCStruct.mIDCs[cru].getIDCGroup().get1DIDCs();
    } else {
      vec1DIDCs = IDCGroup::get1DIDCsUngrouped(mIDCStruct.mIDCs[cru], cru, mPadFlagsMap);
    }
    output.snapshot(Output{gDataOriginTPC, getDataDescription1DIDC(), subSpec, Lifetime::Timeframe}, vec1DIDCs);

    // TODO use this and fix #include <boost/container/pmr/polymorphic_allocator.hpp> in ROOT CINT
    // output.adoptContainer(Output{gDataOriginTPC, getDataDescriptionIDCGroup(), subSpec, Lifetime::Timeframe}, std::move(mIDCs[cru]).getIDCGroupData());
    output.snapshot(Output{gDataOriginTPC, getDataDescriptionIDCGroup(), subSpec, Lifetime::Timeframe}, mIDCStruct.getData(cru));

    mBuffer1DIDCs[cru].emplace_back(std::move(vec1DIDCs));
    mBuffer1DIDCs[cru].pop_front(); // removing oldest 1D-IDCs

    fill1DIDCs(cru);
    output.snapshot(Output{gDataOriginTPC, getDataDescription1DIDCEPN(), subSpec, Lifetime::Timeframe}, mOneDIDCs);
  }

  void fill1DIDCs(const uint32_t cru)
  {
    // fill 1D-IDC vector with mRangeIDC buffered values
    unsigned int i = mRangeIDC;
    for (auto it = mBuffer1DIDCs[cru].crbegin(); it != mBuffer1DIDCs[cru].crend(); ++it) {
      for (auto idc = it->crbegin(); idc != it->crend(); ++idc) {
        mOneDIDCs[--i] = *idc;
        if (i == 0) {
          return;
        }
      }
    }
  }
};

template <class Type>
DataProcessorSpec getTPCFLPIDCSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int rangeIDC, const bool debug, const bool loadFromFile, const bool loadStatusMap = false)
{
  std::vector<OutputSpec> outputSpecs;
  std::vector<InputSpec> inputSpecs;
  outputSpecs.reserve(crus.size());
  inputSpecs.reserve(crus.size());

  for (const auto& cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    if (!loadFromFile) {
      inputSpecs.emplace_back(InputSpec{"idcs", gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), subSpec, Lifetime::Timeframe});
    }

    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice<Type>::getDataDescriptionIDCGroup(), subSpec});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice<Type>::getDataDescription1DIDC(), subSpec});
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice<Type>::getDataDescription1DIDCEPN(), subSpec});
  }

  if (loadStatusMap) {
    LOGP(info, "Using pad status map from CCDB");
    inputSpecs.emplace_back("tpcpadmap", gDataOriginTPC, "PADSTATUSMAP", 0, Lifetime::Condition, ccdbParamSpec("TPC/Calib/IDC/PadStatusMap"));
  }

  const auto id = fmt::format("tpc-flp-idc-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFLPIDCDevice<Type>>(ilane, crus, rangeIDC, debug, loadFromFile, loadStatusMap)}}; // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
