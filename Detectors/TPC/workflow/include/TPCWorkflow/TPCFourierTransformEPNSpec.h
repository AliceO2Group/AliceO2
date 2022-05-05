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

/// \file TPCFourierTransformEPNSpec.h
/// \brief fourier transform of 1D-IDCs used during synchronous reconstruction
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jun 10, 2021

#ifndef O2_TPCFOURIERTRANSFORMEPNSPEC_H
#define O2_TPCFOURIERTRANSFORMEPNSPEC_H

#include <vector>
#include <fmt/format.h>
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "TPCCalibration/IDCFourierTransform.h"
#include "TPCWorkflow/TPCFLPIDCSpec.h"
#include "TPCBase/CRU.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

class TPCFourierTransformEPNSpec : public o2::framework::Task
{
 public:
  using IDCFType = IDCFourierTransform<IDCFourierTransformBaseEPN>;

  TPCFourierTransformEPNSpec(const std::vector<uint32_t>& crus, const unsigned int nFourierCoefficientsSend, const unsigned int rangeIDC, const bool debug = false) : mCRUs{crus}, mIDCFourierTransform{rangeIDC, nFourierCoefficientsSend}, mDebug{debug} {};

  void run(o2::framework::ProcessingContext& pc) final
  {
    for (auto const& ref : InputRecordWalker(pc.inputs(), mFilter)) {
      const auto currTF = processing_helpers::getCurrentTF(pc);
      if (mReceivedCRUs == 0) {
        mCurrentTF = currTF;
      } else if (mCurrentTF != currTF) {
        LOGP(error, "Received TF {} expected TF {}", currTF, mCurrentTF);
        continue;
      }

      ++mReceivedCRUs;
      auto const* tpcCRUHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int cru = tpcCRUHeader->subSpecification >> 7;
      const o2::tpc::CRU cruTmp(cru);
      const auto descr = tpcCRUHeader->dataDescription;

      if (TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescription1DIDCEPN() == descr) {
        LOGP(debug, "Receiving IDC1 for TF {} for CRU {}", currTF, cru);
        mIDCOneAggregator.aggregate1DIDCs(cruTmp.side(), pc.inputs().get<std::vector<float>>(ref));
      } else {
        LOGP(debug, "Receiving IDC1 weights for TF {} for CRU {}", currTF, cru);
        mIDCOneAggregator.aggregate1DIDCsWeights(cruTmp.side(), pc.inputs().get<std::vector<unsigned int>>(ref));
      }
    }

    LOGP(debug, "Received data {} of a total of {}", mReceivedCRUs, 2 * mCRUs.size());
    if (mReceivedCRUs != 2 * mCRUs.size()) {
      return;
    } else {
      mReceivedCRUs = 0;
    }

    // perform fourier transform of 1D-IDCs
    LOGP(debug, "normalize IDCs");
    mIDCOneAggregator.normalizeIDCOne();
    mIDCFourierTransform.setIDCs(std::move(mIDCOneAggregator).get());
    LOGP(debug, "calculate fourier coefficients");
    mIDCFourierTransform.calcFourierCoefficients();

    if (mDebug) {
      LOGP(info, "dumping FT to file");
      mIDCFourierTransform.dumpToFile(fmt::format("FourierEPN_{:02}.root", o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true))->tfCounter).data());
    }

    sendOutput(pc.outputs());
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  /// \return returns data description for output fourier coefficients
  static constexpr header::DataDescription getDataDescription() { return header::DataDescription{"FOURIERCOEFF"}; }

 private:
  const std::vector<uint32_t> mCRUs{};                                    ///< CRUs to process in this instance
  IDCFourierTransform<IDCFourierTransformBaseEPN> mIDCFourierTransform{}; ///< object for performing the fourier transform of 1D-IDCs
  IDCOneAggregator mIDCOneAggregator{};                                   ///< helper class for aggregation of 1D-IDCs
  const bool mDebug{false};                                               ///< dump IDCs to tree for debugging
  int mReceivedCRUs = 0;                                                  ///< counter to keep track of the number of received data from CRUs
  uint32_t mCurrentTF{0};                                                 ///< currently processed TF
  const std::vector<InputSpec> mFilter = {{"1didcepn", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescription1DIDCEPN()}, Lifetime::Timeframe},
                                          {"1didcepnweights", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescription1DIDCEPNWeights()}, Lifetime::Timeframe}}; ///< filter for looping over input data

  void sendOutput(DataAllocator& output)
  {
    output.snapshot(Output{gDataOriginTPC, getDataDescription(), header::DataHeader::SubSpecificationType{Side::A}, Lifetime::Timeframe}, mIDCFourierTransform.getFourierCoefficients().getFourierCoefficients(Side::A));
    output.snapshot(Output{gDataOriginTPC, getDataDescription(), header::DataHeader::SubSpecificationType{Side::C}, Lifetime::Timeframe}, mIDCFourierTransform.getFourierCoefficients().getFourierCoefficients(Side::C));
  }
};

DataProcessorSpec getTPCFourierTransformEPNSpec(const std::vector<uint32_t>& crus, const unsigned int rangeIDC, const unsigned int nFourierCoefficientsSend, const bool debug = false)
{
  std::vector<InputSpec> inputSpecs{InputSpec{"1didcepn", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescription1DIDCEPN()}, Lifetime::Timeframe},
                                    InputSpec{"1didcepnweights", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFLPIDCDevice<TPCFLPIDCDeviceGroup>::getDataDescription1DIDCEPNWeights()}, Lifetime::Timeframe}};

  std::vector<OutputSpec> outputSpecs{ConcreteDataMatcher{gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription(), header::DataHeader::SubSpecificationType{o2::tpc::Side::A}},
                                      ConcreteDataMatcher{gDataOriginTPC, TPCFourierTransformEPNSpec::getDataDescription(), header::DataHeader::SubSpecificationType{o2::tpc::Side::C}}};

  return DataProcessorSpec{
    "tpc-epn-ft",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformEPNSpec>(crus, nFourierCoefficientsSend, rangeIDC, debug)}};
}

} // namespace o2::tpc

#endif
