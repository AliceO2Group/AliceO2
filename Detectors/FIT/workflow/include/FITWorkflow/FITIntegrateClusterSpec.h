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

#ifndef O2_FITINTEGRATECLUSTERSPEC_SPEC
#define O2_FITINTEGRATECLUSTERSPEC_SPEC

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "DetectorsCalibration/IntegratedClusterCalibrator.h"

using namespace o2::framework;

namespace o2
{

namespace fv0
{
class RecPoints;
}

namespace ft0
{
class RecPoints;
}

namespace fit
{

template <typename DataT>
struct DataDescriptionFITCurrents;

template <>
struct DataDescriptionFITCurrents<o2::fv0::RecPoints> {
  using DataTStruct = IFV0C;
  std::string static getName() { return "fv0"; }
  std::string static getCCDBPath() { return "FT0/Calib/IFV0C"; }
  static constexpr header::DataDescription getDataDescriptionFITC() { return header::DataDescription{"IFV0C"}; }
  static constexpr header::DataDescription getDataDescriptionFITTFId() { return header::DataDescription{"IFV0TFID"}; }
  static constexpr header::DataDescription getDataDescriptionCCDB() { return header::DataDescription{"IFV0CCCDB"}; }
  static constexpr header::DataOrigin getDataOrigin() { return header::gDataOriginFV0; }
};

template <>
struct DataDescriptionFITCurrents<o2::ft0::RecPoints> {
  using DataTStruct = IFT0C;
  std::string static getName() { return "ft0"; }
  std::string static getCCDBPath() { return "FV0/Calib/IFT0C"; }
  static constexpr header::DataDescription getDataDescriptionFITC() { return header::DataDescription{"IFT0C"}; }
  static constexpr header::DataDescription getDataDescriptionFITTFId() { return header::DataDescription{"IFT0TFID"}; }
  static constexpr header::DataDescription getDataDescriptionCCDB() { return header::DataDescription{"IFT0CCCDB"}; }
  static constexpr header::DataOrigin getDataOrigin() { return header::gDataOriginFT0; }
};

template <typename DataT>
class FITIntegrateClusters : public Task
{
 public:
  /// \constructor
  FITIntegrateClusters(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool disableWriter, const int minNChan, const int minAmpl) : mCCDBRequest(req), mDisableWriter(disableWriter), mMinNChan(minNChan), mMinAmpl(minAmpl){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNSlicesTF = ic.options().get<int>("nSlicesTF");
    mBufferCurrents.resize(mNSlicesTF);
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const int nHBFPerTF = o2::base::GRPGeomHelper::instance().getNHBFPerTF();
    const int nBunchesPerTF = nHBFPerTF * o2::constants::lhc::LHCMaxBunches;
    const uint32_t firstTFOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const float bunchesPerSlice = nBunchesPerTF / float(mNSlicesTF);
    const float bunchesPerSliceInv = 1.f / bunchesPerSlice;

    // reset buffered currents
    mBufferCurrents.reset();

    // looping over cluster and integrating the currents
    const auto clusters = pc.inputs().get<gsl::span<DataT>>("recpoints");
    for (const auto& cluster : clusters) {
      const unsigned int orbit = cluster.getInteractionRecord().orbit;
      const uint32_t relOrbit = orbit - firstTFOrbit; // from 0->128
      const unsigned int bunchInSlice = relOrbit * o2::constants::lhc::LHCMaxBunches + cluster.getInteractionRecord().bc;
      const unsigned int sliceInTF = bunchInSlice / bunchesPerSlice;
      if (sliceInTF < mNSlicesTF) {
        const float nChanA = static_cast<float>(cluster.getTrigger().getNChanA());
        const float amplA = static_cast<float>(cluster.getTrigger().getAmplA());

        if ((nChanA > mMinNChan) && (amplA > mMinAmpl)) {
          mBufferCurrents.mINChanA[sliceInTF] += nChanA;
          mBufferCurrents.mIAmplA[sliceInTF] += amplA;
        }

        if constexpr (std::is_same_v<DataT, o2::ft0::RecPoints>) {
          const float nChanC = static_cast<float>(cluster.getTrigger().getNChanC());
          const float amplC = static_cast<float>(cluster.getTrigger().getAmplC());
          if ((nChanC > mMinNChan) && (amplC > mMinAmpl)) {
            mBufferCurrents.mINChanC[sliceInTF] += nChanC;
            mBufferCurrents.mIAmplC[sliceInTF] += amplC;
          }
        }
      } else {
        LOGP(info, "slice in TF {} is larger than max expected slice {} with relOrbit {} and {} orbits per slice", sliceInTF, mNSlicesTF, relOrbit, bunchesPerSlice);
      }
    }

    // normalize currents to integration time
    mBufferCurrents.normalize(bunchesPerSliceInv);
    sendOutput(pc);
  }

  void endOfStream(EndOfStreamContext& eos) final { eos.services().get<ControlService>().readyToQuit(QuitRequest::Me); }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

 private:
  int mNSlicesTF = 11;                                                     ///< number of slices a TF is divided into
  const int mMinNChan = 2;                                                 ///< Minimum NChan signal required to avoid noise
  const int mMinAmpl = 2;                                                  ///< Minimum Ampl signal required to avoid noise
  const bool mDisableWriter{false};                                        ///< flag if no ROOT output will be written
  typename DataDescriptionFITCurrents<DataT>::DataTStruct mBufferCurrents; ///< buffer for integrate currents
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;                  ///< info for CCDB request

  void sendOutput(ProcessingContext& pc)
  {
    using FitType = DataDescriptionFITCurrents<DataT>;
    pc.outputs().snapshot(Output{FitType::getDataOrigin(), FitType::getDataDescriptionFITC()}, mBufferCurrents);
    // in case of ROOT output also store the TFinfo in the TTree
    if (!mDisableWriter) {
      o2::dataformats::TFIDInfo tfinfo;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfinfo);
      pc.outputs().snapshot(Output{FitType::getDataOrigin(), FitType::getDataDescriptionFITTFId()}, tfinfo);
    }
  }
};

template <typename DataT>
o2::framework::DataProcessorSpec getFITIntegrateClusterSpec(const bool disableWriter, const int minNChan, const int minAmpl)
{
  using FitType = DataDescriptionFITCurrents<DataT>;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("recpoints", FitType::getDataOrigin(), "RECPOINTS", 0, Lifetime::Timeframe);
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(FitType::getDataOrigin(), FitType::getDataDescriptionFITC(), 0, Lifetime::Sporadic);
  if (!disableWriter) {
    outputs.emplace_back(FitType::getDataOrigin(), FitType::getDataDescriptionFITTFId(), 0, Lifetime::Sporadic);
  }

  return DataProcessorSpec{
    fmt::format("{}-integrate-clusters", FitType::getName()),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<FITIntegrateClusters<DataT>>(ccdbRequest, disableWriter, minNChan, minAmpl)},
    Options{
      {"nSlicesTF", VariantType::Int, 11, {"number of slices into which a TF is divided"}}}};
}

} // namespace fit
} // end namespace o2

#endif
