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
#include "TPCWorkflow/TPCFactorizeSACSpec.h"
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

  TPCFourierTransformAggregatorSpec(const unsigned int nFourierCoefficientsStore, const unsigned int rangeIDC, const bool senddebug = false, const bool processSACs = false, const int inputLanes = 1)
    : mIDCFourierTransform{IDCFType(rangeIDC, nFourierCoefficientsStore), IDCFType(rangeIDC, nFourierCoefficientsStore)}, mSendOutDebug{senddebug}, mProcessSACs{processSACs}, mInputLanes{inputLanes} {};

  void init(o2::framework::InitContext& ic) final
  {
    mDumpFFT = ic.options().get<bool>("dump-coefficients-agg");
    mIntervalsSACs = ic.options().get<int>("intervalsSACs");
    resizeBuffer(mInputLanes);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const int lane = pc.inputs().get<int>("lane");
    if (lane >= mInputLanes) {
      LOGP(error, "Received data from lane {} which is >= than the specified number of expected lanes of {}!", lane, mInputLanes);
      return;
    }

    mCCDBBuffer[lane] = pc.inputs().get<std::vector<long>>("tsccdb");
    if (mProcessedTimeStamp > mCCDBBuffer[lane].front()) {
      LOGP(warning, "Already received data from a later time stamp {} then the currently received time stamp {}! (This might not be an issue)", mProcessedTimeStamp, mCCDBBuffer[lane].front());
    } else {
      mProcessedTimeStamp = mCCDBBuffer[lane].front();
    }

    if (!mProcessSACs) {
      mIntervalsBuffer[lane] = pc.inputs().get<std::vector<unsigned int>>("intervals");
    }

    for (auto& ref : InputRecordWalker(pc.inputs(), mFilter[mProcessSACs])) {
      auto const* dataHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      const int side = dataHeader->subSpecification;
      mIDCOneBuffer[lane][side].mIDCOne = pc.inputs().get<std::vector<float>>(ref);
      LOGP(info, "Received {} 1D-IDCs for side {}", mIDCOneBuffer[lane][side].mIDCOne.size(), side);

      if (mProcessSACs && mIntervalsBuffer[lane].empty()) {
        const auto nValues = mIDCOneBuffer[lane][side].mIDCOne.size();
        const int nIntervals = nValues / mIntervalsSACs;
        const int nFirstInterval = nValues % mIntervalsSACs;
        if (nFirstInterval == 0) {
          mIntervalsBuffer[lane] = std::vector<unsigned int>(nIntervals, mIntervalsSACs);
        } else {
          mIntervalsBuffer[lane] = std::vector<unsigned int>(nIntervals + 1, mIntervalsSACs);
          mIntervalsBuffer[lane].front() = nFirstInterval;
        }
      }
    }

    FourierCoeffSAC coeffSAC;
    if (lane == mExpectedInputLane) {
      const int nSides = mIDCOneBuffer[lane][Side::A].mIDCOne.empty() + mIDCOneBuffer[lane][Side::C].mIDCOne.empty();
      // int iProcessLane = lane;
      for (int iProcessLaneTmp = 0; iProcessLaneTmp < mInputLanes; ++iProcessLaneTmp) {
        const int nSidesCurrLane = mIDCOneBuffer[mExpectedInputLane][Side::A].mIDCOne.empty() + mIDCOneBuffer[mExpectedInputLane][Side::C].mIDCOne.empty();
        if (nSidesCurrLane != nSides) {
          break;
        }

        for (int iSide = 0; iSide < SIDES; ++iSide) {
          const Side side = (iSide == 0) ? A : C;
          if (mIDCOneBuffer[mExpectedInputLane][side].mIDCOne.empty()) {
            continue;
          }
          LOGP(info, "Processing input lane: {} for Side: {}", mExpectedInputLane, iSide);

          // perform fourier transform of 1D-IDCs
          mIDCFourierTransform[side].setIDCs(std::move(mIDCOneBuffer[mExpectedInputLane][side]), mIntervalsBuffer[mExpectedInputLane]);
          mIDCFourierTransform[side].calcFourierCoefficients(mIntervalsBuffer[mExpectedInputLane].size());

          if (!mProcessSACs) {
            o2::ccdb::CcdbObjectInfo ccdbInfo(CDBTypeMap.at(((side == 0) ? CDBType::CalIDCFourierA : CDBType::CalIDCFourierC)), std::string{}, std::string{}, std::map<std::string, std::string>{}, mCCDBBuffer[mExpectedInputLane].front(), mCCDBBuffer[mExpectedInputLane].back());
            auto imageFFT = o2::ccdb::CcdbApi::createObjectImage(&mIDCFourierTransform[side].getFourierCoefficients(), &ccdbInfo);
            LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfo.getPath(), ccdbInfo.getFileName(), imageFFT->size(), ccdbInfo.getStartValidityTimestamp(), ccdbInfo.getEndValidityTimestamp());
            pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBFourier(), 0}, *imageFFT.get());
            pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBFourier(), 0}, ccdbInfo);
          } else {
            coeffSAC.mCoeff[side] = mIDCFourierTransform[side].getFourierCoefficients();
          }

          if (mDumpFFT) {
            LOGP(info, "dumping FT to file");
            mIDCFourierTransform[side].dumpToFile(fmt::format("FourierAGG_{:02}_side{}.root", processing_helpers::getCurrentTF(pc), side).data());
          }

          if (mSendOutDebug) {
            sendOutput(pc.outputs(), side);
          }
        }

        if (mProcessSACs) {
          o2::ccdb::CcdbObjectInfo ccdbInfo(CDBTypeMap.at(CDBType::CalSACFourier), std::string{}, std::string{}, std::map<std::string, std::string>{}, mCCDBBuffer[mExpectedInputLane].front(), mCCDBBuffer[mExpectedInputLane].back());
          auto imageFFT = o2::ccdb::CcdbApi::createObjectImage(&coeffSAC, &ccdbInfo);
          LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfo.getPath(), ccdbInfo.getFileName(), imageFFT->size(), ccdbInfo.getStartValidityTimestamp(), ccdbInfo.getEndValidityTimestamp());
          pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBFourier(), 0}, *imageFFT.get());
          pc.outputs().snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBFourier(), 0}, ccdbInfo);
        }
        mExpectedInputLane = ++mExpectedInputLane % mInputLanes;
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
  std::array<IDCFType, SIDES> mIDCFourierTransform{};              ///< object for performing the fourier transform of 1D-IDCs
  const bool mSendOutDebug{false};                                 ///< flag if the output will be send (for debugging)
  const bool mProcessSACs{false};                                  ///< flag for processing SACs instead of IDCs
  const int mInputLanes{1};                                        ///< number of lanes from which input is expected
  bool mDumpFFT{false};                                            ///< dump fourier coefficients to file
  uint64_t mProcessedTimeStamp{0};                                 ///< to keep track of the processed timestamps
  std::vector<std::vector<long>> mCCDBBuffer{};                    ///< buffer for CCDB time stamp in case one facotorize lane is earlier sending data the n the other lane
  std::vector<std::vector<unsigned int>> mIntervalsBuffer{};       ///< buffer for the intervals in case one facotorize lane is earlier sending data the n the other lane
  std::vector<std::array<o2::tpc::IDCOne, SIDES>> mIDCOneBuffer{}; ///< buffer for the received IDCOne in case one facotorize lane is earlier sending data the n the other lane
  unsigned int mIntervalsSACs{12};                                 ///< number of intervals which are skipped for calculationg the fourier coefficients
  int mExpectedInputLane{0};                                       ///< expeceted data from this input lane
  const std::array<std::vector<InputSpec>, 2> mFilter = {std::vector<InputSpec>{{"idcone", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC1()}, Lifetime::Sporadic}},
                                                         std::vector<InputSpec>{{"sacone", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1()}, Lifetime::Sporadic}}}; ///< filter for looping over input data

  void sendOutput(DataAllocator& output, const int side)
  {
    output.snapshot(Output{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, mIDCFourierTransform[side].getFourierCoefficients());
  }

  void resizeBuffer(const int expectedLanes)
  {
    mCCDBBuffer.resize(expectedLanes);
    mIntervalsBuffer.resize(expectedLanes);
    mIDCOneBuffer.resize(expectedLanes);
  }
};
DataProcessorSpec getTPCFourierTransformAggregatorSpec(const unsigned int rangeIDC, const unsigned int nFourierCoefficientsStore, const bool senddebug, const bool processSACs, const int inputLanes)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);

  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputSpecs;
  if (!processSACs) {
    inputSpecs.emplace_back(InputSpec{"idcone", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC1()}, Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionTimeStamp(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"intervals", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIntervals(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"lane", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionLane(), Lifetime::Sporadic});
  } else {
    inputSpecs.emplace_back(InputSpec{"sacone", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1()}, Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionTimeStamp(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"lane", gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionLane(), Lifetime::Sporadic});
  }

  return DataProcessorSpec{
    "tpc-aggregator-ft",
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformAggregatorSpec>(nFourierCoefficientsStore, rangeIDC, senddebug, processSACs, inputLanes)},
    Options{{"intervalsSACs", VariantType::Int, 11, {"Number of integration intervals which will be sampled for the fourier coefficients"}},
            {"dump-coefficients-agg", VariantType::Bool, false, {"Dump fourier coefficients to file"}}}};
}

} // namespace o2::tpc

#endif
