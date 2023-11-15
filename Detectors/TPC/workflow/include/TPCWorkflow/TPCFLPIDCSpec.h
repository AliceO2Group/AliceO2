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
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/ConfigParamRegistry.h"
#include "Headers/DataHeader.h"
#include "TPCWorkflow/TPCIntegrateIDCSpec.h"
#include "TPCCalibration/IDCFactorization.h"
#include "Framework/CCDBParamSpec.h"
#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCBase/CDBInterface.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFLPIDCDevice : public o2::framework::Task
{
 public:
  TPCFLPIDCDevice(const int lane, const std::vector<uint32_t>& crus, const unsigned int rangeIDC, const bool loadStatusMap, const std::string idc0File, const bool loadIDC0CCDB, const bool enableSynchProc, const int nTFsBuffer)
    : mLane{lane}, mCRUs{crus}, mRangeIDC{rangeIDC}, mLoadPadMapCCDB{loadStatusMap}, mLoadIDC0CCDB{loadIDC0CCDB}, mEnableSynchProc{enableSynchProc}, mNTFsBuffer{nTFsBuffer}, mOneDIDCs{std::vector<float>(rangeIDC), std::vector<unsigned int>(rangeIDC)}
  {
    const auto nSizeDeque = std::ceil(static_cast<float>(mRangeIDC) / mMinIDCsPerTF);
    for (const auto& cru : mCRUs) {
      mBuffer1DIDCs.emplace(cru, std::deque<std::pair<std::vector<float>, std::vector<unsigned int>>>(nSizeDeque, {std::vector<float>(mMinIDCsPerTF, 0), std::vector<unsigned int>(mMinIDCsPerTF, 1)}));
    }

    // loading IDC0
    if (!idc0File.empty()) {
      TFile f(idc0File.data(), "READ");
      IDCZero* idcs = nullptr;
      f.GetObject("IDC0", idcs);
      if (idcs) {
        LOGP(info, "Setting IDC0 from file {}", idc0File);
        mIDCZero = *idcs;
        delete idcs;
      }
    } else if (!mLoadIDC0CCDB) {
      LOGP(info, "setting standard IDC0 values");
      mIDCZero.mIDCZero = std::vector<float>(Mapper::getNumberOfPadsPerSide(), 1);
    }
  }

  void init(o2::framework::InitContext& ic) final
  {
    mDumpIDCs = ic.options().get<bool>("dump-idcs-flp");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(debug, "Processing IDCs for TF {} for CRUs {} to {}", processing_helpers::getCurrentTF(pc), mCRUs.front(), mCRUs.back());

    // retrieving map containing the status flags for the static outlier
    if (mLoadPadMapCCDB) {
      updateTimeDependentParams(pc);
    }

    if (mLoadIDC0CCDB) {
      LOGP(debug, "Loading IDC0 from CCDB as reference for calculating IDC1");
      auto idc = pc.inputs().get<o2::tpc::IDCZero*>("idczero");
      mIDCZero = *idc;
      // load IDC0 only once TODO use correct time stamp for CCDB access?
      mLoadIDC0CCDB = false;
    }

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      ++mCountTFsForBuffer;
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification >> 7;
      auto vecIDCs = pc.inputs().get<o2::pmr::vector<float>>(ref);
      mIDCs[cru].insert(mIDCs[cru].end(), vecIDCs.begin(), vecIDCs.end());

      if (mEnableSynchProc) {
        sendOutputSync(pc.outputs(), vecIDCs, cru);
      }
    }

    // check if the condition for sending the data is met
    if (mCountTFsForBuffer >= mNTFsBuffer) {
      mCountTFsForBuffer = 0;
      for (const auto cru : mCRUs) {
        LOGP(debug, "Sending IDCs of size {} for TF {}", mIDCs[cru].size(), processing_helpers::getCurrentTF(pc));
        sendOutput(pc.outputs(), cru);
      }
    }

    if (mDumpIDCs) {
      TFile fOut(fmt::format("IDCs_{}_tf_{}.root", mLane, processing_helpers::getCurrentTF(pc)).data(), "RECREATE");
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
        auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int cru = tpcCRUHeader->subSpecification >> 7;
        auto vec = pc.inputs().get<std::vector<float>>(ref);
        fOut.WriteObject(&vec, fmt::format("CRU_{}", cru).data());
      }
    }
  }

  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final
  {
    if (matcher == ConcreteDataMatcher(gDataOriginTPC, "PADSTATUSMAP", 0)) {
      LOGP(debug, "Updating pad status from CCDB");
      mPadFlagsMap = static_cast<o2::tpc::CalDet<PadFlags>*>(obj);
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final { ec.services().get<ControlService>().readyToQuit(QuitRequest::Me); }

  static constexpr header::DataDescription getDataDescriptionIDCGroup(const Side side) { return (side == Side::A) ? getDataDescriptionIDCGroupA() : getDataDescriptionIDCGroupC(); }

  /// return data description for IDC Group on A Side
  static constexpr header::DataDescription getDataDescriptionIDCGroupA() { return header::DataDescription{"IDCGROUPA"}; }

  /// return data description for IDC Group on C Side
  static constexpr header::DataDescription getDataDescriptionIDCGroupC() { return header::DataDescription{"IDCGROUPC"}; }

  /// return data description for buffered 1D IDCs for EPNs
  static constexpr header::DataDescription getDataDescription1DIDCEPN() { return header::DataDescription{"1DIDCEPN"}; }

  /// return data description for buffered weights for 1D IDCs for EPNs
  static constexpr header::DataDescription getDataDescription1DIDCEPNWeights() { return header::DataDescription{"1DIDCEPNWEIGHTS"}; }

  /// set minimum IDCs which will be received per TF
  /// \param nMinIDCsPerTF minimal number of IDCs per TF
  static void setMinIDCsPerTF(const unsigned int nMinIDCsPerTF) { mMinIDCsPerTF = nMinIDCsPerTF; }

  /// \return returns the minimum IDCs which will be received per TF
  static unsigned int getMinIDCsPerTF() { return mMinIDCsPerTF; }

 private:
  inline static unsigned int mMinIDCsPerTF{10};                                                                                                                                                      ///< minimum number of IDCs per TF (depends only on number of orbits per TF and (fixed) number of orbits per integration interval). 11 for 128 orbits per TF and 21 for 256 orbits per TF
  const int mLane{};                                                                                                                                                                                 ///< lane number of processor
  const std::vector<uint32_t> mCRUs{};                                                                                                                                                               ///< CRUs to process in this instance
  const unsigned int mRangeIDC{};                                                                                                                                                                    ///< number of IDCs used for the calculation of fourier coefficients
  const bool mLoadPadMapCCDB{};                                                                                                                                                                      ///< load status map for pads from CCDB
  bool mLoadIDC0CCDB{};                                                                                                                                                                              ///< loading reference IDC0 from CCDB
  const bool mEnableSynchProc{};                                                                                                                                                                     ///< flag for enabling calculation of 1D-IDCs
  int mNTFsBuffer{1};                                                                                                                                                                                ///< number of TFs for which the IDCs will be buffered
  bool mDumpIDCs{};                                                                                                                                                                                  ///< dump IDCs to tree for debugging
  int mCountTFsForBuffer{0};                                                                                                                                                                         ///< count processed TFs to track when the output will be send
  std::pair<std::vector<float>, std::vector<unsigned int>> mOneDIDCs{};                                                                                                                              ///< 1D-IDCs which will be send to the EPNs
  std::unordered_map<unsigned int, o2::pmr::vector<float>> mIDCs{};                                                                                                                                  ///< object for averaging and grouping of the IDCs
  std::unordered_map<unsigned int, std::deque<std::pair<std::vector<float>, std::vector<unsigned int>>>> mBuffer1DIDCs{};                                                                            ///< buffer for 1D-IDCs. The buffered 1D-IDCs for n TFs will be send to the EPNs for synchronous reco. Zero initialized to avoid empty first TFs!
  CalDet<PadFlags>* mPadFlagsMap{nullptr};                                                                                                                                                           ///< status flag for each pad (i.e. if the pad is dead)
  IDCZero mIDCZero{};                                                                                                                                                                                ///< I_0(r,\phi) = <I(r,\phi,t)>_t: Used for calculating IDC1 (provided from input file or CCDB)
  const std::vector<InputSpec> mFilter = {{"idcs", ConcreteDataTypeMatcher{gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim)}, Lifetime::Timeframe}}; ///< filter for looping over input data

  /// update the time dependent parameters if they have changed (i.e. update the pad status map)
  void updateTimeDependentParams(ProcessingContext& pc) { pc.inputs().get<o2::tpc::CalDet<PadFlags>*>("tpcpadmap").get(); }

  void sendOutputSync(DataAllocator& output, const o2::pmr::vector<float>& idc, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    const CRU cruTmp(cru);
    std::pair<std::vector<float>, std::vector<unsigned int>> idcOne;
    const int integrationIntervalOffset = 0;
    const auto region = cruTmp.region();
    const unsigned int indexOffset = (cruTmp.sector() % SECTORSPERSIDE) * Mapper::getPadsInSector() + Mapper::GLOBALPADOFFSET[region]; // TODO get correct offset for TPCFLPIDCDeviceGroup case
    const auto nIDCsPerIntegrationInterval = Mapper::PADSPERREGION[region];
    const auto integrationIntervals = idc.size() / nIDCsPerIntegrationInterval;
    idcOne.first.resize(integrationIntervals);
    idcOne.second.resize(integrationIntervals);
    IDCFactorization::calcIDCOne(idc, nIDCsPerIntegrationInterval, integrationIntervalOffset, indexOffset, cruTmp, idcOne.first, idcOne.second, &mIDCZero, mPadFlagsMap);

    // normalize to pad size
    std::transform(idcOne.first.begin(), idcOne.first.end(), idcOne.first.begin(), [normVal = Mapper::INVPADAREA[region]](auto& val) { return val * normVal; });

    mBuffer1DIDCs[cru].emplace_back(std::move(idcOne));
    mBuffer1DIDCs[cru].pop_front(); // removing oldest 1D-IDCs

    fill1DIDCs(cru);
    LOGP(debug, "Sending 1D-IDCs to EPNs of size {} and weights of size {}", mOneDIDCs.first.size(), mOneDIDCs.second.size());
    output.snapshot(Output{gDataOriginTPC, getDataDescription1DIDCEPN(), subSpec, Lifetime::Timeframe}, mOneDIDCs.first);
    output.snapshot(Output{gDataOriginTPC, getDataDescription1DIDCEPNWeights(), subSpec, Lifetime::Timeframe}, mOneDIDCs.second);
  }

  void sendOutput(DataAllocator& output, const uint32_t cru)
  {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    output.adoptContainer(Output{gDataOriginTPC, getDataDescriptionIDCGroup(CRU(cru).side()), subSpec, Lifetime::Timeframe}, std::move(mIDCs[cru]));
  }

  void fill1DIDCs(const uint32_t cru)
  {
    // fill 1D-IDC vector with mRangeIDC buffered values
    unsigned int i = mRangeIDC;
    for (int indexDeque = mBuffer1DIDCs[cru].size() - 1; indexDeque >= 0; --indexDeque) {
      for (int indexIDCs = mBuffer1DIDCs[cru][indexDeque].first.size() - 1; indexIDCs >= 0; --indexIDCs) {
        mOneDIDCs.first[--i] = mBuffer1DIDCs[cru][indexDeque].first[indexIDCs];
        mOneDIDCs.second[i] = mBuffer1DIDCs[cru][indexDeque].second[indexIDCs];
        if (i == 0) {
          return;
        }
      }
    }
  }
};

DataProcessorSpec getTPCFLPIDCSpec(const int ilane, const std::vector<uint32_t>& crus, const unsigned int rangeIDC, const bool loadStatusMap, const std::string idc0File, const bool disableIDC0CCDB, const bool enableSynchProc, const int nTFsBuffer = 1)
{
  std::vector<OutputSpec> outputSpecs;
  std::vector<InputSpec> inputSpecs;
  outputSpecs.reserve(crus.size());
  inputSpecs.reserve(crus.size());

  for (const auto& cru : crus) {
    const header::DataHeader::SubSpecificationType subSpec{cru << 7};
    inputSpecs.emplace_back(InputSpec{"idcs", gDataOriginTPC, TPCIntegrateIDCDevice::getDataDescription(TPCIntegrateIDCDevice::IDCFormat::Sim), subSpec, Lifetime::Timeframe});

    const Side side = CRU(cru).side();
    outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice::getDataDescriptionIDCGroup(side), subSpec}, Lifetime::Sporadic);
    if (enableSynchProc) {
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice::getDataDescription1DIDCEPN(), subSpec});
      outputSpecs.emplace_back(ConcreteDataMatcher{gDataOriginTPC, TPCFLPIDCDevice::getDataDescription1DIDCEPNWeights(), subSpec});
    }
  }

  const Side side = CRU(crus.front()).side();
  if (loadStatusMap) {
    LOGP(info, "Using pad status map from CCDB");
    inputSpecs.emplace_back("tpcpadmap", gDataOriginTPC, "PADSTATUSMAP", 0, Lifetime::Condition, ccdbParamSpec((side == Side::A) ? CDBTypeMap.at(CDBType::CalIDCPadStatusMapA) : CDBTypeMap.at(CDBType::CalIDCPadStatusMapC)));
  }

  const bool loadIDC0CCDB = !disableIDC0CCDB && idc0File.empty();
  if (loadIDC0CCDB) {
    inputSpecs.emplace_back("idczero", gDataOriginTPC, "IDC0", 0, Lifetime::Condition, ccdbParamSpec((side == Side::A) ? CDBTypeMap.at(CDBType::CalIDC0A) : CDBTypeMap.at(CDBType::CalIDC0C)));
  }

  const auto id = fmt::format("tpc-flp-idc-{:02}", ilane);
  return DataProcessorSpec{
    id.data(),
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFLPIDCDevice>(ilane, crus, rangeIDC, loadStatusMap, idc0File, loadIDC0CCDB, enableSynchProc, nTFsBuffer)},
    Options{{"dump-idcs-flp", VariantType::Bool, false, {"Dump IDCs to file"}}}}; // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
