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

/// \file TPCFourierTransformAggregatorSpec.h
/// \brief TPC aggregation of 1D-IDCs and fourier transform
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jun 16, 2021

#ifndef O2_TPCFOURIERTRANSFORMAGGREGATORSPEC_H
#define O2_TPCFOURIERTRANSFORMAGGREGATORSPEC_H

#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include "CCDB/CcdbApi.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/IDCFourierTransform.h"
#include "TPCWorkflow/TPCDistributeIDCSpec.h"
#include "TPCBase/CRU.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFourierTransformAggregatorSpec : public o2::framework::Task
{
 public:
  // Fourier type
  using IDCFType = IDCFourierTransform<IDCFTType::IDCFourierTransformBaseAggregator>;

  TPCFourierTransformAggregatorSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int nFourierCoefficientsStore, const unsigned int rangeIDC, const bool debug = false, const bool senddebug = false)
    : mTimeFrames{timeframes}, mCRUs{crus}, mIDCFourierTransform{rangeIDC, timeframes, nFourierCoefficientsStore}, mOneDIDCAggregator{timeframes}, mDebug{debug}, mSendOutDebug{senddebug} {};

  void init(o2::framework::InitContext& ic) final
  {
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    // set the min range of TFs for first TF
    if (mProcessedTFs == 0) {
      mTFRange[0] = getCurrentTF(pc);
    }

    for (int i = 0; i < mCRUs.size(); ++i) {
      const DataRef ref = pc.inputs().getByPos(i);
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const o2::tpc::CRU cruTmp(tpcCRUHeader->subSpecification);
      mOneDIDCAggregator.aggregate1DIDCs(cruTmp.side(), pc.inputs().get<std::vector<float>>(ref), mProcessedTFs, cruTmp.region());
    }
    ++mProcessedTFs;

    if (!(mProcessedTFs % ((mTimeFrames + 5) / 5))) {
      LOGP(info, "aggregated TFs: {}", mProcessedTFs);
    }

    if (mProcessedTFs == mTimeFrames) {
      mTFRange[1] = getCurrentTF(pc); // set the TF for last aggregated TF
      mProcessedTFs = 0;              // reset processed TFs for next aggregation interval

      // perform fourier transform of 1D-IDCs
      auto intervals = mOneDIDCAggregator.getIntegrationIntervalsPerTF();
      mIDCFourierTransform.setIDCs(std::move(mOneDIDCAggregator).getAggregated1DIDCs(), std::move(intervals));
      mIDCFourierTransform.calcFourierCoefficients();

      if (mDebug) {
        LOGP(info, "dumping FT to file");
        mIDCFourierTransform.dumpToFile(fmt::format("FourierAGG_{:02}.root", getCurrentTF(pc)).data());
      }

      // storing to CCDB
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionFourier() { return header::DataDescription{"FOURIER"}; }

 private:
  const unsigned int mTimeFrames{};             ///< number of time frames which will be aggregated
  const std::vector<uint32_t> mCRUs{};          ///< CRUs to process in this instance
  IDCFType mIDCFourierTransform{};              ///< object for performing the fourier transform of 1D-IDCs
  OneDIDCAggregator mOneDIDCAggregator{};       ///< helper class for aggregation of 1D-IDCs
  const bool mDebug{false};                     ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};              ///< flag if the output will be send (for debugging)
  o2::ccdb::CcdbApi mDBapi;                     ///< API for storing the IDCs in the CCDB
  std::map<std::string, std::string> mMetadata; ///< meta data of the stored object in CCDB
  bool mWriteToDB{};                            ///< flag if writing to CCDB will be done
  std::array<uint32_t, 2> mTFRange{};           ///< storing of first and last TF used when setting the validity of the objects when writing to CCDB
  int mProcessedTFs{0};                         ///< number of processed time frames to keep track of when the writing to CCDB will be done

  /// \return returns TF of current processed data
  uint32_t getCurrentTF(o2::framework::ProcessingContext& pc) const { return o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getByPos(0))->tfCounter; }

  void sendOutput(DataAllocator& output)
  {
    if (mSendOutDebug) {
      output.snapshot(Output{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, mIDCFourierTransform.getFourierCoefficients());
    }

    if (mWriteToDB) {
      mDBapi.storeAsTFileAny<o2::tpc::FourierCoeff>(&mIDCFourierTransform.getFourierCoefficients(), "TPC/Calib/IDC/FOURIER", mMetadata, mTFRange[0], mTFRange[1]);
    }
  }
};

DataProcessorSpec getTPCFourierTransformAggregatorSpec(const std::vector<uint32_t>& crus, const unsigned int timeframes, const unsigned int rangeIDC, const unsigned int nFourierCoefficientsStore, const bool debug = false, const bool senddebug = false)
{
  std::vector<OutputSpec> outputSpecs;
  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()});
  }

  std::vector<InputSpec> inputSpecs;
  inputSpecs.reserve(crus.size());
  for (const auto cru : crus) {
    inputSpecs.emplace_back(InputSpec{"1didc", gDataOriginTPC, TPCDistributeIDCSpec::getDataDescription1DIDC(), header::DataHeader::SubSpecificationType{cru}, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "tpc-aggregator-ft",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformAggregatorSpec>(crus, timeframes, nFourierCoefficientsStore, rangeIDC, debug, senddebug)},
    Options{{"ccdb-uri", VariantType::String, "http://ccdb-test.cern.ch:8080", {"URI for the CCDB access."}}}}; // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
