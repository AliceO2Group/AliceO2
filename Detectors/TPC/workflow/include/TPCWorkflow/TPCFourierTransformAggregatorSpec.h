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
    : mIDCFourierTransform{IDCFType(rangeIDC, timeframes, nFourierCoefficientsStore), IDCFType(rangeIDC, timeframes, nFourierCoefficientsStore)}, mDebug{debug}, mSendOutDebug{senddebug} {};

  void run(o2::framework::ProcessingContext& pc) final
  {
    LOGP(info, "1D-IDCs received. Performing FFT");

    const auto& timeStampsCCDB = pc.inputs().get<std::vector<uint64_t>>("tsccdb");
    auto intervals = pc.inputs().get<std::vector<unsigned int>>("intervals");

    // LOGP(info, "Received {} 1D-IDCs for A-Side and {} 1D-IDCs for C-Side", idcOneA.size(), idcOneC.size());
    LOGP(info, "Received data for timestamp {} to {}", timeStampsCCDB.front(), timeStampsCCDB.back());

    if (mProcessedTimeStamp > timeStampsCCDB.front()) {
      LOGP(error, "Already processed a later time stamp {} then the received time stamp {}!", mProcessedTimeStamp, timeStampsCCDB.front());
    } else {
      mProcessedTimeStamp = timeStampsCCDB.front();
    }

    // TODO avoid copy
    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      auto const* dataHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int side = dataHeader->subSpecification;

      o2::tpc::IDCOne idcOne;
      idcOne.mIDCOne = pc.inputs().get<std::vector<float>>(ref);
      LOGP(info, "Received {} 1D-IDCs for side {}", idcOne.mIDCOne.size(), side);

      // perform fourier transform of 1D-IDCs
      mIDCFourierTransform[side].setIDCs(std::move(idcOne), intervals);
      mIDCFourierTransform[side].calcFourierCoefficients();

      o2::ccdb::CcdbObjectInfo ccdbInfo(CDBTypeMap.at((side == 0) ? CDBType::CalIDCFourierA : CDBType::CalIDCFourierC), std::string{}, std::string{}, std::map<std::string, std::string>{}, timeStampsCCDB.front(), timeStampsCCDB.back());
      auto imageFFT = o2::ccdb::CcdbApi::createObjectImage(&mIDCFourierTransform[side].getFourierCoefficients(), &ccdbInfo);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfo.getPath(), ccdbInfo.getFileName(), imageFFT->size(), ccdbInfo.getStartValidityTimestamp(), ccdbInfo.getEndValidityTimestamp());
      pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBFourier(), 0}, *imageFFT.get());
      pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBFourier(), 0}, ccdbInfo);

      if (mDebug) {
        LOGP(info, "dumping FT to file");
        mIDCFourierTransform[side].dumpToFile(fmt::format("FourierAGG_{:02}_side{}.root", processing_helpers::getCurrentTF(pc), side).data());
      }

      if (mSendOutDebug) {
        sendOutput(pc.outputs(), side);
      }
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionFourier() { return header::DataDescription{"FOURIER"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBFourier() { return header::DataDescription{"TPC_CalibFFT"}; }

 private:
  std::array<IDCFType, SIDES> mIDCFourierTransform{};                                                                                                                            ///< object for performing the fourier transform of 1D-IDCs
  const bool mDebug{false};                                                                                                                                                      ///< dump IDCs to tree for debugging
  const bool mSendOutDebug{false};                                                                                                                                               ///< flag if the output will be send (for debugging)
  uint64_t mProcessedTimeStamp{0};                                                                                                                                               ///< to keep track of the processed timestamps
  const std::vector<InputSpec> mFilter = {{"idcone", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIDC1()}, Lifetime::Sporadic}}; ///< filter for looping over input data

  void sendOutput(DataAllocator& output, const int side)
  {
    output.snapshot(Output{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, mIDCFourierTransform[side].getFourierCoefficients());
  }
};

DataProcessorSpec getTPCFourierTransformAggregatorSpec(const unsigned int timeframes, const unsigned int rangeIDC, const unsigned int nFourierCoefficientsStore, const bool debug = false, const bool senddebug = false)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);

  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputSpecs;
  inputSpecs.emplace_back(InputSpec{"idcone", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIDC1()}, Lifetime::Sporadic});
  inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionTimeStamp(), Lifetime::Sporadic});
  inputSpecs.emplace_back(InputSpec{"intervals", gDataOriginTPC, TPCFactorizeIDCSpec<>::getDataDescriptionIntervals(), Lifetime::Sporadic});

  return DataProcessorSpec{
    "tpc-aggregator-ft",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformAggregatorSpec>(timeframes, nFourierCoefficientsStore, rangeIDC, debug, senddebug)}};
}

} // namespace o2::tpc

#endif
