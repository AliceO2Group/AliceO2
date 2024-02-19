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

/// \file TPCScalerSpec.cxx
/// \brief device for providing tpc scaler per TF
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Nov 4, 2023

#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCBase/CDBInterface.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCCalibration/TPCScaler.h"
#include "TPCCalibration/TPCMShapeCorrection.h"
#include "TTree.h"
#include "TPCCalibration/TPCFastSpaceChargeCorrectionHelper.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "CommonUtils/TreeStreamRedirector.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

class TPCScalerSpec : public Task
{
 public:
  TPCScalerSpec(std::shared_ptr<o2::base::GRPGeomRequest> req, bool enableIDCs, bool enableMShape) : mCCDBRequest(req), mEnableIDCs(enableIDCs), mEnableMShape(enableMShape){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mIonDriftTimeMS = ic.options().get<float>("ion-drift-time");
    mMaxTimeWeightsMS = ic.options().get<float>("max-time-for-weights");
    mMShapeScalingFac = ic.options().get<float>("m-shape-scaling-factor");
    mEnableWeights = !(ic.options().get<bool>("disableWeights"));
    const bool enableStreamer = ic.options().get<bool>("enableStreamer");
    const int mshapeThreads = ic.options().get<int>("n-threads");
    mKnotsYMshape = ic.options().get<int>("n-knots-y");
    mKnotsZMshape = ic.options().get<int>("n-knots-z");
    TPCFastSpaceChargeCorrectionHelper::instance()->setNthreads(mshapeThreads);
    o2::tpc::SpaceCharge<double>::setNThreads(mshapeThreads);

    if (enableStreamer) {
      mStreamer = std::make_unique<o2::utils::TreeStreamRedirector>("M_Shape.root", "recreate");
    }
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    if (mStreamer) {
      mStreamer->Close();
    }
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    if (mEnableIDCs && pc.inputs().isValid("tpcscaler")) {
      pc.inputs().get<TTree*>("tpcscaler");
    }

    if (mEnableWeights && mEnableIDCs) {
      if (pc.inputs().isValid("tpcscalerw")) {
        pc.inputs().get<TPCScalerWeights*>("tpcscalerw");
      }
    }
    if (mEnableMShape && pc.inputs().isValid("mshape")) {
      pc.inputs().get<TTree*>("mshape");
    }

    const auto orbitResetTimeMS = o2::base::GRPGeomHelper::instance().getOrbitResetTimeMS();
    const auto firstTFOrbit = pc.services().get<o2::framework::TimingInfo>().firstTForbit;
    const double timestamp = orbitResetTimeMS + firstTFOrbit * o2::constants::lhc::LHCOrbitMUS * 0.001;

    if (mEnableMShape) {
      if ((mMShapeTPCScaler.getRun() != -1) && pc.services().get<o2::framework::TimingInfo>().runNumber != mMShapeTPCScaler.getRun()) {
        LOGP(error, "Run number {} of processed data and run number {} of loaded TPC M-shape scaler doesnt match!", pc.services().get<o2::framework::TimingInfo>().runNumber, mMShapeTPCScaler.getRun());
      }

      const auto& boundaryPotential = mMShapeTPCScaler.getBoundaryPotential(timestamp);
      if (!boundaryPotential.mPotential.empty()) {
        LOGP(info, "Calculating M-shape correction from input boundary potential");
        const Side side = Side::A;
        o2::tpc::SpaceCharge<double> sc = o2::tpc::SpaceCharge<double>(mMShapeTPCScaler.getMShapes().mBField, mMShapeTPCScaler.getMShapes().mZ, mMShapeTPCScaler.getMShapes().mR, mMShapeTPCScaler.getMShapes().mPhi);
        for (int iz = 0; iz < mMShapeTPCScaler.getMShapes().mZ; ++iz) {
          for (int iphi = 0; iphi < mMShapeTPCScaler.getMShapes().mPhi; ++iphi) {
            const float pot = mMShapeScalingFac * boundaryPotential.mPotential[iz];
            sc.setPotential(iz, 0, iphi, side, pot);
          }
        }

        sc.poissonSolver(side);
        sc.calcEField(side);
        sc.calcGlobalCorrections(sc.getElectricFieldsInterpolator(side));

        std::function<void(int, double, double, double, double&, double&, double&)> getCorrections = [&sc = sc](const int roc, double x, double y, double z, double& dx, double& dy, double& dz) {
          Side side = roc < 18 ? Side::A : Side::C;
          if (side == Side::C) {
            dx = 0;
            dy = 0;
            dz = 0;
          } else {
            sc.getCorrections(x, y, z, side, dx, dy, dz);
          }
        };

        std::unique_ptr<TPCFastSpaceChargeCorrection> spCorrection = TPCFastSpaceChargeCorrectionHelper::instance()->createFromGlobalCorrection(getCorrections, mKnotsYMshape, mKnotsZMshape);
        std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0, *spCorrection));
        pc.outputs().snapshot(Output{header::gDataOriginTPC, "TPCMSHAPE"}, *fastTransform);
      } else {
        // send empty dummy object
        LOGP(info, "Sending default (no) M-shape correction");
        auto fastTransform = o2::tpc::TPCFastTransformHelperO2::instance()->create(0);
        pc.outputs().snapshot(Output{header::gDataOriginTPC, "TPCMSHAPE"}, *fastTransform);
      }

      if (mStreamer) {
        (*mStreamer) << "treeMShape"
                     << "firstTFOrbit=" << firstTFOrbit
                     << "timestamp=" << timestamp
                     << "boundaryPotential=" << boundaryPotential
                     << "mMShapeScalingFac=" << mMShapeScalingFac
                     << "\n";
      }
    }

    if (mEnableIDCs) {
      if (pc.services().get<o2::framework::TimingInfo>().runNumber != mTPCScaler.getRun()) {
        LOGP(error, "Run number {} of processed data and run number {} of loaded TPC scaler doesnt match!", pc.services().get<o2::framework::TimingInfo>().runNumber, mTPCScaler.getRun());
      }
      float scalerA = mTPCScaler.getMeanScaler(timestamp, o2::tpc::Side::A);
      float scalerC = mTPCScaler.getMeanScaler(timestamp, o2::tpc::Side::C);
      float meanScaler = (scalerA + scalerC) / 2;
      LOGP(info, "Publishing TPC scaler: {} for timestamp: {}, firstTFOrbit: {}", meanScaler, timestamp, firstTFOrbit);
      pc.outputs().snapshot(Output{header::gDataOriginTPC, "TPCSCALER"}, meanScaler);
      if (mStreamer) {
        (*mStreamer) << "treeIDC"
                     << "scalerA=" << scalerA
                     << "scalerC=" << scalerC
                     << "firstTFOrbit=" << firstTFOrbit
                     << "timestamp=" << timestamp
                     << "\n";
      }
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCSCALERCCDB", 0)) {
      LOGP(info, "Updating TPC scaler");
      mTPCScaler.setFromTree(*((TTree*)obj));
      if (mIonDriftTimeMS > 0) {
        LOGP(info, "Setting ion drift time to: {}", mIonDriftTimeMS);
        mTPCScaler.setIonDriftTimeMS(mIonDriftTimeMS);
      }
      if (mScalerWeights.isValid()) {
        LOGP(info, "Setting TPC scaler weights");
        mTPCScaler.setScalerWeights(mScalerWeights);
        mTPCScaler.useWeights(true);
        if (mIonDriftTimeMS == -1) {
          overWriteIntegrationTime();
        }
      }
    }
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "TPCSCALERWCCDB", 0)) {
      LOGP(info, "Updating TPC scaler weights");
      mScalerWeights = *(TPCScalerWeights*)obj;
      mTPCScaler.setScalerWeights(mScalerWeights);
      mTPCScaler.useWeights(true);
      // in case ion drift time is not specified it is overwritten by the value in the weight object
      if (mIonDriftTimeMS == -1) {
        overWriteIntegrationTime();
      }
    }
    if (matcher == ConcreteDataMatcher(o2::header::gDataOriginTPC, "MSHAPEPOTCCDB", 0)) {
      LOGP(info, "Updating M-shape TPC scaler");
      mMShapeTPCScaler.setFromTree(*((TTree*)obj));
      if (mMShapeTPCScaler.getRun() == -1) {
        LOGP(info, "Loaded default M-Shape correction object from CCDB");
      }
    }
  }

 private:
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;     ///< info for CCDB request
  const bool mEnableIDCs{true};                               ///< enable IDCs
  const bool mEnableMShape{false};                            ///< enable v shape scalers
  bool mEnableWeights{false};                                 ///< use weights for TPC scalers
  TPCScalerWeights mScalerWeights{};                          ///< scaler weights
  float mIonDriftTimeMS{-1};                                  ///< ion drift time
  float mMaxTimeWeightsMS{500};                               ///< maximum integration time when weights are used
  TPCScaler mTPCScaler;                                       ///< tpc scaler
  float mMShapeScalingFac{0};                                 ///< scale m-shape scalers with this value
  TPCMShapeCorrection mMShapeTPCScaler;                       ///< TPC M-shape scalers
  int mKnotsYMshape{4};                                       ///< number of knots used for the spline object for M-Shape distortions
  int mKnotsZMshape{4};                                       ///< number of knots used for the spline object for M-Shape distortions
  std::unique_ptr<o2::utils::TreeStreamRedirector> mStreamer; ///< streamer

  void overWriteIntegrationTime()
  {
    float integrationTime = std::abs(mScalerWeights.mFirstTimeStampMS);
    if (integrationTime <= 0) {
      return;
    }
    if (integrationTime > mMaxTimeWeightsMS) {
      integrationTime = mMaxTimeWeightsMS;
    }
    LOGP(info, "Setting maximum integration time for weights to: {}", integrationTime);
    mTPCScaler.setIonDriftTimeMS(integrationTime);
  }
};

o2::framework::DataProcessorSpec getTPCScalerSpec(bool enableIDCs, bool enableMShape)
{
  std::vector<InputSpec> inputs;
  if (enableIDCs) {
    LOGP(info, "Publishing IDC scalers for space-charge distortion fluctuation correction");
    inputs.emplace_back("tpcscaler", o2::header::gDataOriginTPC, "TPCSCALERCCDB", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalScaler), {}, 1));          // time-dependent
    inputs.emplace_back("tpcscalerw", o2::header::gDataOriginTPC, "TPCSCALERWCCDB", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalScalerWeights), {}, 0)); // non time-dependent
  }
  if (enableMShape) {
    LOGP(info, "Publishing M-shape correction map");
    inputs.emplace_back("mshape", o2::header::gDataOriginTPC, "MSHAPEPOTCCDB", 0, Lifetime::Condition, ccdbParamSpec(o2::tpc::CDBTypeMap.at(o2::tpc::CDBType::CalMShape), {}, 1)); // time-dependent
  }

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                false,                          // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  if (enableIDCs) {
    outputs.emplace_back(o2::header::gDataOriginTPC, "TPCSCALER", 0, Lifetime::Timeframe);
  }
  if (enableMShape) {
    outputs.emplace_back(o2::header::gDataOriginTPC, "TPCMSHAPE", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-scaler",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCScalerSpec>(ccdbRequest, enableIDCs, enableMShape)},
    Options{
      {"ion-drift-time", VariantType::Float, -1.f, {"Overwrite ion drift time if a value >0 is provided"}},
      {"max-time-for-weights", VariantType::Float, 500.f, {"Maximum possible integration time in ms when weights are used"}},
      {"m-shape-scaling-factor", VariantType::Float, 1.f, {"Scale M-shape scaler with this value"}},
      {"disableWeights", VariantType::Bool, false, {"Disable weights for TPC scalers"}},
      {"enableStreamer", VariantType::Bool, false, {"Enable streaming of M-shape scalers"}},
      {"n-threads", VariantType::Int, 4, {"Number of threads used for the M-shape correction"}},
      {"n-knots-y", VariantType::Int, 4, {"Number of knots in y-direction used for the M-shape correction"}},
      {"n-knots-z", VariantType::Int, 4, {"Number of knots in z-direction used for the M-shape correction"}}}};
}

} // namespace tpc
} // namespace o2
