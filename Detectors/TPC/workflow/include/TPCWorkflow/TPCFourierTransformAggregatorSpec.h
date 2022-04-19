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
#include "TPCCalibration/IDCFourierTransform.h"
#include "TPCWorkflow/TPCFactorizeIDCSpec.h"
#include "TPCBase/CRU.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFourierTransformAggregatorSpec : public o2::framework::Task
{
 public:
  // Fourier type
  using IDCFType = IDCFourierTransform<IDCFourierTransformBaseAggregator>;

  TPCFourierTransformAggregatorSpec(const unsigned int timeframes, const unsigned int nFourierCoefficientsStore, const unsigned int rangeIDC, const bool debug = false, const bool senddebug = false)
    : mIDCFourierTransform{rangeIDC, timeframes, nFourierCoefficientsStore}, mDebug{debug}, mSendOutDebug{senddebug} {};

  void init(o2::framework::InitContext& ic) final
  {
    mDBapi.init(ic.options().get<std::string>("ccdb-uri")); // or http://localhost:8080 for a local installation
    mWriteToDB = mDBapi.isHostReachable() ? true : false;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(info, "1D-IDCs received. Performing FFT");

    const auto& idcOneA = pc.inputs().get<std::vector<float>>("idconeA");
    const auto& idcOneC = pc.inputs().get<std::vector<float>>("idconeC");
    const auto& timeStampsCCDB = pc.inputs().get<std::vector<uint64_t>>("tsccdb");
    auto intervals = pc.inputs().get<std::vector<unsigned int>>("intervals");

    LOGP(info, "Received {} 1D-IDCs for A-Side and {} 1D-IDCs for C-Side", idcOneA.size(), idcOneC.size());
    LOGP(info, "Received data for timestamp {} to {}", timeStampsCCDB.front(), timeStampsCCDB.back());

    if (mProcessedTimeStamp > timeStampsCCDB.front()) {
      LOGP(error, "Already processed a later time stamp {} then the received time stamp {}!", mProcessedTimeStamp, timeStampsCCDB.front());
    } else {
      mProcessedTimeStamp = timeStampsCCDB.back();
    }

    // TODO avoid copy
    o2::tpc::IDCOne idcOne;
    idcOne.mIDCOne[Side::A] = idcOneA;
    idcOne.mIDCOne[Side::C] = idcOneC;

    // perform fourier transform of 1D-IDCs
    mIDCFourierTransform.setIDCs(std::move(idcOne), std::move(intervals));
    mIDCFourierTransform.calcFourierCoefficients();

    if (mWriteToDB) {
      mDBapi.storeAsTFileAny<o2::tpc::FourierCoeff>(&mIDCFourierTransform.getFourierCoefficients(), CDBTypeMap.at(CDBType::CalIDCFourier), mMetadata, timeStampsCCDB.front(), timeStampsCCDB.back());
    }

    if (mDebug) {
      LOGP(info, "dumping FT to file");
      mIDCFourierTransform.dumpToFile(fmt::format("FourierAGG_{:02}.root", processing_helpers::getCurrentTF(pc)).data());
    }

    if (mSendOutDebug) {
      sendOutput(pc.outputs());
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionFourier() { return header::DataDescription{"FOURIER"}; }

 private:
  IDCFType mIDCFourierTransform{};              ///< object for performing the fourier transform of 1D-IDCs
  const bool mDebug{false};                     ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};              ///< flag if the output will be send (for debugging)
  o2::ccdb::CcdbApi mDBapi;                     ///< API for storing the IDCs in the CCDB
  std::map<std::string, std::string> mMetadata; ///< meta data of the stored object in CCDB
  bool mWriteToDB{};                            ///< flag if writing to CCDB will be done
  uint64_t mProcessedTimeStamp{0};              ///< to keep track of the processed timestamps

  void sendOutput(DataAllocator& output)
  {
    output.snapshot(Output{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, mIDCFourierTransform.getFourierCoefficients());
  }
};

DataProcessorSpec getTPCFourierTransformAggregatorSpec(const unsigned int timeframes, const unsigned int rangeIDC, const unsigned int nFourierCoefficientsStore, const bool debug = false, const bool senddebug = false)
{
  std::vector<OutputSpec> outputSpecs;
  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()});
  }

  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"idconeA", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::A}, Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"idconeC", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIDC1(), header::DataHeader::SubSpecificationType{Side::C}, Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionTimeStamp(), Lifetime::Timeframe});
  inputSpecs.emplace_back(InputSpec{"intervals", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIntervals(), Lifetime::Timeframe});

  return DataProcessorSpec{
    "tpc-aggregator-ft",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformAggregatorSpec>(timeframes, nFourierCoefficientsStore, rangeIDC, debug, senddebug)},
    Options{{"ccdb-uri", VariantType::String, o2::base::NameConf::getCCDBServer(), {"URI for the CCDB access."}}}}; // end DataProcessorSpec
}

} // namespace o2::tpc

#endif
