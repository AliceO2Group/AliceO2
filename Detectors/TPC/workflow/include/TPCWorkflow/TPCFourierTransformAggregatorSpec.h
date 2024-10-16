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
#include "TPCCalibration/TPCScaler.h"

using namespace o2::framework;
using o2::header::gDataOriginTPC;
using namespace o2::tpc;

namespace o2::tpc
{

struct TPCScalerProc {
  std::array<long, 2> timestamp;          ///< start -> end timestamp
  std::array<std::vector<float>, 2> idc1; ///< IDC1
};

struct TPCScalerProcContainer {
  std::unordered_map<long, TPCScalerProc> idcs;
};

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
    mLengthIDCScalerSeconds = ic.options().get<float>("tpcScalerLengthS");
    mDisableScaler = ic.options().get<bool>("disable-scaler");
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

    // buffer IDCs for TPC scaler
    if (!mProcessSACs && !mDisableScaler) {
      const long startTS = mCCDBBuffer[lane].front();
      TPCScalerProc& scaler = mTPCScalerCont.idcs[startTS];
      scaler.timestamp[0] = startTS;
      scaler.timestamp[1] = mCCDBBuffer[lane].back();
      for (auto& ref : InputRecordWalker(pc.inputs(), mFilterI0)) {
        auto const* dataHeader = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        const int side = dataHeader->subSpecification;
        const float idc0mean = pc.inputs().get<float>(ref);
        LOGP(info, "Received {} IDC0 mean for side {}", idc0mean, side);
        scaler.idc1[side] = mIDCOneBuffer[lane][side].mIDCOne;
        auto& vecIDC = scaler.idc1[side];

        // normalize 1D-IDCs to unity as it should be
        if (vecIDC.size() > 0) {
          const float mean = std::reduce(vecIDC.begin(), vecIDC.end()) / static_cast<float>(vecIDC.size());
          LOGP(info, "normalizing by {}", mean);
          if (std::abs(mean) > 0.001) {
            std::transform(vecIDC.begin(), vecIDC.end(), vecIDC.begin(), [&mean](auto val) { return val / mean; });
          }
        }

        // scale IDC1 with IDC0Mean
        std::transform(scaler.idc1[side].begin(), scaler.idc1[side].end(), scaler.idc1[side].begin(), [&idc0mean](auto idc) { return idc * idc0mean; });
      }
      // check if A- and C-side has the same length!
      const int lenA = scaler.idc1[0].size();
      const int lenC = scaler.idc1[1].size();
      if (lenA != lenC) {
        // This should never happen
        LOGP(warning, "Received IDCs have different length! A-side length: {} and C-side length: {}", lenA, lenC);
        // add dummy to shorter vector
        const int maxLen = std::max(lenA, lenC);
        scaler.idc1[0].resize(maxLen);
        scaler.idc1[1].resize(maxLen);
      }

      mRun = processing_helpers::getRunNumber(pc);
      makeTPCScaler(pc.outputs(), false);
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
            mIDCFourierTransform[side].dumpToFile(fmt::format("FourierAGG_{:02}_side{}.root", processing_helpers::getCurrentTF(pc), (int)side).data());
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
    if (!mDisableScaler) {
      makeTPCScaler(ec.outputs(), true);
    }
    ec.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }

  static constexpr header::DataDescription getDataDescriptionFourier() { return header::DataDescription{"FOURIER"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBFourier() { return header::DataDescription{"TPC_CalibFFT"}; }
  static constexpr header::DataDescription getDataDescriptionCCDBTPCScaler() { return header::DataDescription{"TPC_IDCScaler"}; }

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
  TPCScalerProcContainer mTPCScalerCont;                           ///< container for buffering the IDCs for the creation of the TPC scalers
  float mLengthIDCScalerSeconds = 300;                             ///< length of the IDC scaler in seconds
  long mIDCSCalerEndTSLast = 0;                                    ///< end time stamp of last TPC IDC scaler object to ensure no gapps
  o2::tpc::TPCScaler mScalerLast;                                  ///< buffer last scaler to easily add internal overlap for the beginning
  bool mDisableScaler{false};                                      ///< disable the creation of TPC IDC scalers
  int mRun{};
  const std::array<std::vector<InputSpec>, 2> mFilter = {std::vector<InputSpec>{{"idcone", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC1()}, Lifetime::Sporadic}},
                                                         std::vector<InputSpec>{{"sacone", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1()}, Lifetime::Sporadic}}}; ///< filter for looping over input data
  const std::vector<InputSpec> mFilterI0 = std::vector<InputSpec>{{"idczeromean", ConcreteDataTypeMatcher{o2::header::gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC0Mean()}, Lifetime::Sporadic}};       ///< filter for looping over input data from IDC0 mean

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

  void makeTPCScaler(DataAllocator& output, const bool eos)
  {
    LOGP(info, "Making TPC scalers");
    // check if IDC scalers can be created - check length of continous received IDCs
    std::vector<std::pair<long, long>> times;
    times.reserve(mTPCScalerCont.idcs.size());
    for (const auto& idc : mTPCScalerCont.idcs) {
      times.emplace_back(idc.second.timestamp[0], idc.second.timestamp[1]);
    }

    // sort received times of the IDCs
    std::sort(times.begin(), times.end());

    // loop over times and make checks
    const int checkGapp = 10;
    // if the diff between end of data[i] and start of data[i+1] is smaller than this, the data is contigous
    long timesDuration = (times.front().second - times.front().first);

    // make check and store lastValid index in case IDC scalers can be created
    int lastValidIdx = -1;
    for (int i = 1; i < times.size(); ++i) {
      // check time diff between start of current IDCs and end time of last IDCs
      const auto deltaTime = times[i].first - times[i - 1].second;
      // check if IDCs are contigous
      if (deltaTime > (timesDuration / checkGapp)) {
        // check if the gap is very large - in this case the gapp might be lost, so just write out the TPC scaler until the gap
        if (deltaTime > (checkGapp * timesDuration)) {
          lastValidIdx = i - 1;
        }
        LOGP(info, "Breaking as big gap between IDCs of {} detected", deltaTime);
        break;
      }

      // check if time length is >= than mLengthIDCScalerSeconds
      if ((times[i].first - times.front().first) / 1000 >= mLengthIDCScalerSeconds) {
        lastValidIdx = i;
      }
    }

    LOGP(info, "Creating IDC scalers with {} IDC objects", lastValidIdx);

    if (eos) {
      // in case of eos write out everything
      lastValidIdx = times.empty() ? -1 : times.size() - 1;
    }

    // create IDC scaler in case index is valid
    if (lastValidIdx >= 0) {
      o2::tpc::TPCScaler scaler;
      scaler.setIonDriftTimeMS(170);
      scaler.setRun(mRun);
      scaler.setStartTimeStampMS(times.front().first);
      const auto idcIntegrationTime = 12 /*12 orbits integration interval per IDC*/ * o2::constants::lhc::LHCOrbitMUS / 1000;
      scaler.setIntegrationTimeMS(idcIntegrationTime);

      std::vector<float> idc1A;
      std::vector<float> idc1C;
      long idc1ASize = 0;
      long idc1CSize = 0;

      // in case already one object is stored add internal overlap
      if (mIDCSCalerEndTSLast != 0) {
        const int nOverlap = 500; /// ~500ms overlap
        idc1ASize += nOverlap;
        idc1CSize += nOverlap;
        const auto& scalerALast = mScalerLast.getScalers(o2::tpc::Side::A);
        const auto& scalerCLast = mScalerLast.getScalers(o2::tpc::Side::C);
        if (scalerALast.size() > nOverlap) {
          idc1A.insert(idc1A.end(), scalerALast.end() - nOverlap, scalerALast.end());
          idc1C.insert(idc1C.end(), scalerCLast.end() - nOverlap, scalerCLast.end());
          // adjust start time
          scaler.setStartTimeStampMS(scaler.getStartTimeStampMS() - nOverlap * idcIntegrationTime);
        }
      } else {
        // store end timestamp as start time stamp for first object for correct time stamp in CCDB
        mIDCSCalerEndTSLast = scaler.getStartTimeStampMS();
      }

      for (int iter = 0; iter < 2; ++iter) {
        if (iter == 1) {
          idc1A.reserve(idc1ASize);
          idc1C.reserve(idc1CSize);
        }
        for (int i = 0; i <= lastValidIdx; ++i) {
          const auto& time = times[i];
          const auto& idc = mTPCScalerCont.idcs[time.first];
          if (iter == 0) {
            idc1ASize += idc.idc1[0].size();
            idc1CSize += idc.idc1[1].size();
          } else {
            idc1A.insert(idc1A.end(), idc.idc1[0].begin(), idc.idc1[0].end());
            idc1C.insert(idc1C.end(), idc.idc1[1].begin(), idc.idc1[1].end());
            // in case of eos check if the IDCs are contigous and add dummy values!
            if (eos && (i < lastValidIdx)) {
              const float deltaTime = times[i + 1].first - time.second;
              // if delta time is too large add dummy values
              if (deltaTime > (timesDuration / checkGapp)) {
                const int nDummyValues = deltaTime / idcIntegrationTime + 0.5;
                // add dummy to A
                if (idc.idc1[0].size() > 0) {
                  float meanA = std::reduce(idc.idc1[0].begin(), idc.idc1[0].end()) / static_cast<float>(idc.idc1[0].size());
                  idc1A.insert(idc1A.end(), nDummyValues, meanA);
                }

                if (idc.idc1[1].size() > 0) {
                  // add dummy to C
                  float meanC = std::reduce(idc.idc1[1].begin(), idc.idc1[1].end()) / static_cast<float>(idc.idc1[1].size());
                  idc1C.insert(idc1C.end(), nDummyValues, meanC);
                }
              }
            }
            mTPCScalerCont.idcs.erase(time.first);
          }
        }
      }
      scaler.setScaler(idc1A, o2::tpc::Side::A);
      scaler.setScaler(idc1C, o2::tpc::Side::C);

      // store in CCDB
      TTree tree("ccdb_object", "ccdb_object");
      tree.Branch("TPCScaler", &scaler);
      tree.Fill();

      o2::ccdb::CcdbObjectInfo ccdbInfoIDC(CDBTypeMap.at(CDBType::CalScaler), std::string{}, std::string{}, std::map<std::string, std::string>{}, mIDCSCalerEndTSLast, scaler.getEndTimeStampMS(o2::tpc::Side::A));
      auto imageIDC = o2::ccdb::CcdbApi::createObjectImage(&tree, &ccdbInfoIDC);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", ccdbInfoIDC.getPath(), ccdbInfoIDC.getFileName(), imageIDC->size(), ccdbInfoIDC.getStartValidityTimestamp(), ccdbInfoIDC.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, getDataDescriptionCCDBTPCScaler(), 0}, *imageIDC.get());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, getDataDescriptionCCDBTPCScaler(), 0}, ccdbInfoIDC);

      // store end timestamp
      mIDCSCalerEndTSLast = scaler.getEndTimeStampMS(o2::tpc::Side::A);

      // for debugging
      if (mDumpFFT) {
        static int countwrite = 0;
        scaler.dumpToFile(fmt::format("TPCScaler_snapshot_{}.root", countwrite++).data(), "ccdb_object");
      }

      // buffer current scaler object
      mScalerLast = std::move(scaler);
    }
  }
};
DataProcessorSpec getTPCFourierTransformAggregatorSpec(const unsigned int rangeIDC, const unsigned int nFourierCoefficientsStore, const bool senddebug, const bool processSACs, const int inputLanes)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBFourier()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBTPCScaler()}, Lifetime::Sporadic);
  outputSpecs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, TPCFourierTransformAggregatorSpec::getDataDescriptionCCDBTPCScaler()}, Lifetime::Sporadic);

  if (senddebug) {
    outputSpecs.emplace_back(ConcreteDataTypeMatcher{gDataOriginTPC, TPCFourierTransformAggregatorSpec::getDataDescriptionFourier()}, Lifetime::Sporadic);
  }

  std::vector<InputSpec> inputSpecs;
  if (!processSACs) {
    inputSpecs.emplace_back(InputSpec{"idczeromean", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC0Mean()}, Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"idcone", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIDC1()}, Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionTimeStamp(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"intervals", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionIntervals(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"lane", gDataOriginTPC, TPCFactorizeIDCSpec::getDataDescriptionLane(), Lifetime::Sporadic});
  } else {
    inputSpecs.emplace_back(InputSpec{"sacone", ConcreteDataTypeMatcher{gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionSAC1()}, Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"tsccdb", gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionTimeStamp(), Lifetime::Sporadic});
    inputSpecs.emplace_back(InputSpec{"lane", gDataOriginTPC, TPCFactorizeSACSpec::getDataDescriptionLane(), Lifetime::Sporadic});
  }

  std::string processorName = "tpc-aggregator-ft";
  if (processSACs) {
    processorName = "tpc-aggregator-ft-sac";
  }

  return DataProcessorSpec{
    processorName,
    inputSpecs,
    outputSpecs,
    AlgorithmSpec{adaptFromTask<TPCFourierTransformAggregatorSpec>(nFourierCoefficientsStore, rangeIDC, senddebug, processSACs, inputLanes)},
    Options{{"intervalsSACs", VariantType::Int, 11, {"Number of integration intervals which will be sampled for the fourier coefficients"}},
            {"dump-coefficients-agg", VariantType::Bool, false, {"Dump fourier coefficients to file"}},
            {"tpcScalerLengthS", VariantType::Float, 300.f, {"Length of the TPC scalers in seconds"}},
            {"disable-scaler", VariantType::Bool, false, {"Disable creation of IDC scaler"}}}};
}

} // namespace o2::tpc

#endif
