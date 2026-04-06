// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cmath>
#include <memory>
#include <chrono>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include <Eigen/Dense>
#include <GblTrajectory.h>
#include <GblData.h>
#include <GblPoint.h>
#include <GblMeasurement.h>
#include <MilleBinary.h>
#include <nlohmann/json.hpp>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "Steer/MCKinematicsReader.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/IOUtils.h"
#include "ITS3Reconstruction/IOUtils.h"
#include "ITS3Align/TrackFit.h"
#include "ITS3Align/AlignmentSpec.h"
#include "ITS3Align/AlignmentParams.h"
#include "ITS3Align/AlignmentTypes.h"
#include "ITS3Align/AlignmentHierarchy.h"
#include "ITS3Align/AlignmentSensors.h"
#include "MathUtils/LegendrePols.h"

namespace o2::its3::align
{
using namespace o2::framework;
using DetID = o2::detectors::DetID;
using DataRequest = o2::globaltracking::DataRequest;
using PVertex = o2::dataformats::PrimaryVertex;
using V2TRef = o2::dataformats::VtxTrackRef;
using VTIndex = o2::dataformats::VtxTrackIndex;
using GTrackID = o2::dataformats::GlobalTrackID;
using TrackD = o2::track::TrackParCovD;

namespace
{
// compute normalized (u,v) in [-1,1] from global position on a sensor
std::pair<double, double> computeUV(double gloX, double gloY, double gloZ, int sensorID, double radius)
{
  const bool isTop = sensorID % 2 == 0;
  const double phi = o2::math_utils::to02Pid(std::atan2(gloY, gloX));
  const double phiBorder1 = o2::math_utils::to02Pid(((isTop ? 0. : 1.) * TMath::Pi()) + std::asin(constants::equatorialGap / 2. / radius));
  const double phiBorder2 = o2::math_utils::to02Pid(((isTop ? 1. : 2.) * TMath::Pi()) - std::asin(constants::equatorialGap / 2. / radius));
  const double u = (((phi - phiBorder1) * 2.) / (phiBorder2 - phiBorder1)) - 1.;
  const double v = ((2. * gloZ + constants::segment::lengthSensitive) / constants::segment::lengthSensitive) - 1.;
  return {u, v};
}

// evaluate Legendre polynomials P_0(x) through P_order(x) via recurrence
std::vector<double> legendrePols(int order, double x)
{
  std::vector<double> p(order + 1);
  p[0] = 1.;
  if (order > 0) {
    p[1] = x;
  }
  for (int n = 1; n < order; ++n) {
    p[n + 1] = ((2 * n + 1) * x * p[n] - n * p[n - 1]) / (n + 1);
  }
  return p;
}
} // namespace

class AlignmentSpec final : public Task
{
 public:
  ~AlignmentSpec() final = default;
  AlignmentSpec(const AlignmentSpec&) = delete;
  AlignmentSpec(AlignmentSpec&&) = delete;
  AlignmentSpec& operator=(const AlignmentSpec&) = delete;
  AlignmentSpec& operator=(AlignmentSpec&&) = delete;
  AlignmentSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, GTrackID::mask_t src, bool useMC, bool withPV, bool withITS, OutputEnum out)
    : mDataRequest(dr), mGGCCDBRequest(gr), mTracksSrc(src), mUseMC(useMC), mWithPV(withPV), mIsITS3(!withITS), mOutOpt(out)
  {
  }

  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;
  void finaliseCCDB(ConcreteDataMatcher& matcher, void* obj) final;
  void process();

 private:
  void updateTimeDependentParams(ProcessingContext& pc);
  void buildHierarchy();

  // calculate the transport jacobian for points FROM and TO numerically via ridder's method
  // this assumes the track is already at point FROM and will be extrapolated to TO's x (xTo)
  // method does not modify the original track
  bool getTransportJacobian(const TrackD& track, double xTo, double alphaTo, gbl::Matrix5d& jac, gbl::Matrix5d& err);

  // refit ITS track with inward/outward fit (opt. impose pv as additional constraint)
  // after this we have the refitted track at the innermost update point
  bool prepareITSTrack(int iTrk, const o2::its::TrackITS& itsTrack, Track& resTrack);

  // prepare ITS measuremnt points
  void prepareMeasurments(std::span<const itsmft::CompClusterExt> clusters, std::span<const unsigned char> pattIt);

  // build track to vertex association
  void buildT2V();

  // apply some misalignment on inner ITS3 layers
  // it can happen that a measurement is pushed outside of
  // ITS3 acceptance so false is to discard track
  bool applyMisalignment(Eigen::Vector2d& res, const FrameInfoExt& frame, const TrackD& wTrk, size_t iTrk);

  OutputEnum mOutOpt;
  std::unique_ptr<o2::utils::TreeStreamRedirector> mDBGOut;
  std::vector<dataformats::VertexBase> mPVs;
  std::vector<int> mT2PV;
  bool mIsITS3{true};
  const o2::itsmft::TopologyDictionary* mITSDict{nullptr};
  const o2::its3::TopologyDictionary* mIT3Dict{nullptr};
  o2::globaltracking::RecoContainer* mRecoData = nullptr;
  std::unique_ptr<steer::MCKinematicsReader> mcReader;
  std::vector<FrameInfoExt> mITSTrackingInfo;
  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  std::unique_ptr<AlignableVolume> mHierarchy;   // tree-hiearchy
  AlignableVolume::SensorMapping mChip2Hiearchy; // global label mapping to leaves in the tree
  bool mUseMC{false};
  bool mWithPV{false};
  GTrackID::mask_t mTracksSrc;
  int mNThreads{1};
  const AlignmentParams* mParams{nullptr};
  std::array<o2::math_utils::Legendre2DPolynominal, 6> mDeformations; // one per sensorID (0-5)
  std::array<Eigen::Matrix<double, 6, 1>, 6> mRigidBodyParams;        // (dx,dy,dz,rx,ry,rz) in LOC per sensorID
};

void AlignmentSpec::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mNThreads = ic.options().get<int>("nthreads");
  if (mOutOpt) {
    LOG(info) << mOutOpt.pstring();
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>("its3_debug_alg.root", "recreate");
  }
  if (mUseMC) {
    mcReader = std::make_unique<steer::MCKinematicsReader>("collisioncontext.root");
  }
}

void AlignmentSpec::run(ProcessingContext& pc)
{
  if (mOutOpt[OutputOpt::MilleRes]) {
    updateTimeDependentParams(pc);
    writeMillepedeResults(mHierarchy.get(), mParams->milleResFile, mParams->milleResOutJson, mParams->misAlgJson);
  } else {
    o2::globaltracking::RecoContainer recoData;
    mRecoData = &recoData;
    mRecoData->collectData(pc, *mDataRequest);
    updateTimeDependentParams(pc);
    process();
  }
  mRecoData = nullptr;
}

void AlignmentSpec::process()
{
  if (!mITSDict && !mIT3Dict) {
    LOGP(fatal, "ITS data is not loaded");
  }
  auto prop = o2::base::PropagatorD::Instance();
  const auto bz = prop->getNominalBz();
  const auto itsTracks = mRecoData->getITSTracks();
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  const auto clusITS = mRecoData->getITSClusters();
  const auto patterns = mRecoData->getITSClustersPatterns();
  std::span<const o2::MCCompLabel> mcLbls;
  if (mUseMC) {
    mcLbls = mRecoData->getITSTracksMCLabels();
  }
  prepareMeasurments(clusITS, patterns);

  if (mWithPV) {
    buildT2V();
  }

  LOGP(info, "Starting fits with {} threads", mNThreads);

  // Data
  std::vector<std::vector<gbl::GblTrajectory>> gblTrajSlots(mNThreads);
  std::vector<std::vector<Track>> resTrackSlots(mNThreads);

  auto timeStart = std::chrono::high_resolution_clock::now();
  int cFailedRefit{0}, cFailedProp{0}, cSelected{0}, cGBLFit{0}, cGBLFitFail{0}, cGBLChi2Rej{0}, cGBLConstruct{0};
  double chi2Sum{0}, lostWeightSum{0};
  int ndfSum{0};
#ifdef WITH_OPENMP
#pragma omp parallel num_threads(mNThreads) \
  reduction(+ : cFailedRefit)               \
  reduction(+ : cFailedProp)                \
  reduction(+ : cSelected)                  \
  reduction(+ : cGBLFit)                    \
  reduction(+ : cGBLFitFail)                \
  reduction(+ : cGBLChi2Rej)                \
  reduction(+ : cGBLConstruct)              \
  reduction(+ : chi2Sum)                    \
  reduction(+ : lostWeightSum)              \
  reduction(+ : ndfSum)
#endif
  {
#ifdef WITH_OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    auto& gblTrajSlot = gblTrajSlots[tid];
    auto& resTrackSlot = resTrackSlots[tid];

#ifdef WITH_OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t iTrk = 0; iTrk < (int)itsTracks.size(); ++iTrk) {
      const auto& trk = itsTracks[iTrk];
      if (trk.getNClusters() < mParams->minITSCls ||
          (trk.getChi2() / ((float)trk.getNClusters() * 2 - 5)) >= mParams->maxITSChi2Ndf ||
          trk.getPt() < mParams->minPt ||
          (mUseMC && (!mcLbls[iTrk].isValid() || !mcLbls[iTrk].isCorrect()))) {
        continue;
      }
      ++cSelected;
      Track& resTrack = resTrackSlot.emplace_back();
      if (!prepareITSTrack((int)iTrk, trk, resTrack)) {
        ++cFailedRefit;
        resTrackSlot.pop_back();
        continue;
      }

      o2::track::TrackParD* refLin = nullptr;
      if (mParams->useStableRef) {
        refLin = &resTrack.track;
      }

      // outward stepping from track IU
      auto wTrk = resTrack.track;
      const bool hasPV = resTrack.info[0].lr == -1;
      std::vector<gbl::GblPoint> points;
      bool failed = false;
      const int np = (int)resTrack.points.size();
      track::TrackLTIntegral lt;
      lt.setTimeNotNeeded();
      constexpr int perm[5] = {4, 2, 3, 0, 1}; // ALICE->GBL: Q/Pt,Snp,Tgl,Y,Z
      for (int ip{0}; ip < np; ++ip) {
        const auto& frame = resTrack.info[ip];
        gbl::Matrix5d err = gbl::Matrix5d::Identity(), jacALICE = gbl::Matrix5d::Identity(), jacGBL;
        float msErr = 0.f;
        if (ip) {
          // numerically calculates the transport jacobian from prev. point to this point
          // then we actually do the step to the point and accumulate the material
          if (!getTransportJacobian(wTrk, frame.x, frame.alpha, jacALICE, err) ||
              !prop->propagateToAlphaX(wTrk, refLin, frame.alpha, frame.x, false, mParams->maxSnp, mParams->maxStep, 1, mParams->corrType, &lt)) {
            ++cFailedProp;
            failed = true;
            break;
          }
          msErr = its::math_utils::MSangle(trk.getPID().getMass(), trk.getP(), lt.getX2X0());
          // after computing jac, reorder to GBL convention
          for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
              jacGBL(i, j) = jacALICE(perm[i], perm[j]);
            }
          }
        }

        // wTrk is now in the measurment frame
        gbl::GblPoint point(jacGBL);
        // measurement
        Eigen::Vector2d res, prec;
        res << frame.positionTrackingFrame[0] - wTrk.getY(), frame.positionTrackingFrame[1] - wTrk.getZ();

        // here we can apply some misalignment on the measurment
        if (!applyMisalignment(res, frame, wTrk, iTrk)) {
          failed = true;
          break;
        }

        prec << 1. / resTrack.points[ip].sig2y, 1. / resTrack.points[ip].sig2z;
        // the projection matrix is in the tracking frame the idendity so no need to diagonalize it
        point.addMeasurement(res, prec);
        if (msErr > mParams->minMS && ip < np - 1) {
          Eigen::Vector2d scat(0., 0.), scatPrec = Eigen::Vector2d::Constant(1. / (msErr * msErr));
          point.addScatterer(scat, scatPrec);
          lt.clearFast(); // clear if accounted
        }

        if (frame.lr >= 0) {
          GlobalLabel lbl(0, frame.sens, true);
          if (mChip2Hiearchy.find(lbl) == mChip2Hiearchy.end()) {
            LOGP(fatal, "Cannot find global label: {}", lbl.asString());
          }

          // derivatives for all sensitive volumes and their parents
          // this is the derivative in TRK but we want to align in LOC
          // so dr/da_(LOC) = dr/da_(TRK) * da_(TRK)/da_(LOC)
          const auto* tileVol = mChip2Hiearchy.at(lbl);
          Matrix36 der = getRigidBodyDerivatives(wTrk);

          // count rigid body columns: only volumes with real DOFs (not DOFPseudo)
          int nColRB{0};
          for (const auto* v = tileVol; v && !v->isRoot(); v = v->getParent()) {
            if (v->getRigidBody()) {
              nColRB += v->getRigidBody()->nDOFs();
            }
          }

          // count calibration columns
          const auto* sensorVol = tileVol->getParent();
          const auto* calibSet = sensorVol ? sensorVol->getCalib() : nullptr;
          const int nCalib = calibSet ? calibSet->nDOFs() : 0;
          const int nCol = nColRB + nCalib;

          std::vector<int> gLabels;
          gLabels.reserve(nCol);
          Eigen::MatrixXd gDer(3, nCol);
          gDer.setZero();
          Eigen::Index curCol{0};

          // 1) tile: TRK -> LOC via precomputed T2L and J_L2T
          const double posTrk[3] = {frame.x, 0., 0.};
          double posLoc[3];
          tileVol->getT2L().LocalToMaster(posTrk, posLoc);
          Matrix66 jacL2T;
          tileVol->computeJacobianL2T(posLoc, jacL2T);
          der *= jacL2T;
          if (tileVol->getRigidBody()) {
            const int nd = tileVol->getRigidBody()->nDOFs();
            for (int iDOF = 0; iDOF < nd; ++iDOF) {
              gLabels.push_back(tileVol->getLabel().rawGBL(iDOF));
            }
            gDer.middleCols(curCol, nd) = der;
            curCol += nd;
          }

          // 2) chain through parents: child's J_L2P
          for (const auto* child = tileVol; child->getParent() && !child->getParent()->isRoot(); child = child->getParent()) {
            der *= child->getJL2P();
            const auto* parent = child->getParent();
            if (parent->getRigidBody()) {
              const int nd = parent->getRigidBody()->nDOFs();
              for (int iDOF = 0; iDOF < nd; ++iDOF) {
                gLabels.push_back(parent->getLabel().rawGBL(iDOF));
              }
              gDer.middleCols(curCol, nd) = der;
              curCol += nd;
            }
          }

          // 3) calibration derivatives (e.g. Legendre for ITS3 sensors, apply directly on the whole sensor, not on inidividual tiles)
          if (calibSet && calibSet->type() == DOFSet::Type::Legendre) {
            const auto* legSet = static_cast<const LegendreDOFSet*>(calibSet);
            const int N = legSet->order();
            const int sensorID = constants::detID::getSensorID(frame.sens);
            const int layerID = constants::detID::getDetID2Layer(frame.sens);

            const double r = frame.x;
            const double gX = r * std::cos(frame.alpha);
            const double gY = r * std::sin(frame.alpha);
            const double gZ = frame.positionTrackingFrame[1];
            auto [u, v] = computeUV(gX, gY, gZ, sensorID, constants::radii[layerID]);

            const double snp = wTrk.getSnp();
            const double tgl = wTrk.getTgl();
            const double csci = 1. / std::sqrt(1. - (snp * snp));
            const double dydx = snp * csci;
            const double dzdx = tgl * csci;

            auto pu = legendrePols(N, u);
            auto pv = legendrePols(N, v);

            int legIdx = 0;
            const int legColStart = nColRB;
            for (int i = 0; i <= N; ++i) {
              for (int j = 0; j <= i; ++j) {
                const double basis = pu[j] * pv[i - j];
                gLabels.push_back(sensorVol->getLabel().asCalib().rawGBL(legIdx));
                gDer(0, legColStart + legIdx) = dydx * basis;
                gDer(1, legColStart + legIdx) = dzdx * basis;
                ++legIdx;
              }
            }
          }
          point.addGlobals(gLabels, gDer);
        }

        if (mOutOpt[OutputOpt::VerboseGBL]) {
          static Eigen::IOFormat fmt(4, 0, ", ", "\n", "[", "]");
          LOGP(info, "WORKING-POINT {}", ip);
          LOGP(info, "Track: {}", wTrk.asString());
          LOGP(info, "FrameInfo: {}", frame.asString());
          std::cout << "jacALICE:\n"
                    << jacALICE.format(fmt) << '\n';
          std::cout << "jacGBL:\n"
                    << jacGBL.format(fmt) << '\n';
          LOGP(info, "Point {}: GBL res=({}, {}), KF stored res=({}, {})",
               ip, res[0], res[1], resTrack.points[ip].dy, resTrack.points[ip].dz);
          LOGP(info, "residual: dy={} dz={}", res[0], res[1]);
          LOGP(info, "precision: precY={} precZ={}", prec[0], prec[1]);
          point.printPoint(5);
        }
        points.push_back(point);
      }
      if (!failed) {
        gbl::GblTrajectory traj(points, std::abs(bz) > 0.01);
        if (traj.isValid()) {
          double chi2 = NAN, lostWeight = NAN;
          int ndf = 0;
          if (auto ierr = traj.fit(chi2, ndf, lostWeight); !ierr) {
            if (mOutOpt[OutputOpt::VerboseGBL]) {
              LOGP(info, "GBL FIT chi2 {} ndf {}", chi2, ndf);
              traj.printTrajectory(5);
            }
            if (chi2 / ndf > mParams->maxChi2Ndf && cGBLChi2Rej++ < 10) {
              LOGP(error, "GBL fit exceeded red chi2 {}", chi2 / ndf);
              if (std::abs(resTrack.kfFit.chi2Ndf - 1) < 0.02) {
                LOGP(error, "\tGBL is far away from good KF fit!!!!");
                continue;
              }
            } else {
              ++cGBLFit;
              chi2Sum += chi2;
              lostWeightSum += lostWeight;
              ndfSum += ndf;
              if (mOutOpt[OutputOpt::MilleData]) {
                gblTrajSlot.push_back(traj);
              }
              FitInfo fit;
              fit.ndf = ndf;
              fit.chi2 = (float)chi2;
              fit.chi2Ndf = (float)chi2 / (float)ndf;
              resTrack.gblFit = fit;
            }
          } else {
            ++cGBLFitFail;
          }
        } else {
          ++cGBLConstruct;
        }
      }
    }
  }
  auto timeEnd = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart);
  LOGP(info, "Fitted {} tracks out of {} (selected {}) in {} sec", cGBLFit, itsTracks.size(), cSelected, duration.count() / 1e3);
  LOGP(info, "\tRefit failed for {} tracks; Failed prop for {} tracks", cFailedRefit, cFailedProp);
  LOGP(info, "\tGBL SUMMARY:");
  LOGP(info, "\t\tGBL construction failed {}", cGBLConstruct);
  LOGP(info, "\t\tGBL fit failed {}", cGBLFitFail);
  LOGP(info, "\t\tGBL chi2Ndf rejected {}", cGBLChi2Rej);
  if (!ndfSum) {
    LOGP(info, "\t\tGBL Chi2/Ndf = NDF IS 0");
  } else {
    LOGP(info, "\t\tGBL Chi2/Ndf = {}", chi2Sum / ndfSum);
  }
  LOGP(info, "\t\tGBL LostWeight = {}", lostWeightSum);
  LOGP(info, "Streaming results to output");
  if (mOutOpt[OutputOpt::MilleData]) {
    gbl::MilleBinary mille(mParams->milleBinFile, true);
    for (auto& slot : gblTrajSlots) {
      for (auto& traj : slot) {
        traj.milleOut(mille);
      }
    }
  }
  if (mOutOpt[OutputOpt::Debug]) {
    for (auto& slot : resTrackSlots) {
      for (auto& res : slot) {
        (*mDBGOut) << "res"
                   << "trk=" << res
                   << "\n";
      }
    }
  }
}

void AlignmentSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  if (static bool initOnce{false}; !initOnce) {
    initOnce = true;
    auto geom = o2::its::GeometryTGeo::Instance();
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G, o2::math_utils::TransformType::T2G));
    mParams = &AlignmentParams::Instance();
    mParams->printKeyValues(true, true);

    buildHierarchy();

    if (mParams->doMisalignmentLeg || mParams->doMisalignmentRB) {
      TMatrixD null(1, 1);
      null(0, 0) = 0;
      for (int i = 0; i < 6; ++i) {
        mDeformations[i] = o2::math_utils::Legendre2DPolynominal(null);
        mRigidBodyParams[i].setZero();
      }
      if (!mParams->misAlgJson.empty()) {
        using json = nlohmann::json;
        std::ifstream f(mParams->misAlgJson);
        auto data = json::parse(f);
        for (const auto& item : data) {
          int id = item["id"].get<int>();
          if (mParams->doMisalignmentLeg && item.contains("matrix")) {
            auto v = item["matrix"].get<std::vector<std::vector<double>>>();
            TMatrixD m(v.size(), v[v.size() - 1].size());
            for (size_t r{0}; r < v.size(); ++r) {
              for (size_t c{0}; c < v[r].size(); ++c) {
                m(r, c) = v[r][c];
              }
            }
            mDeformations[id] = o2::math_utils::Legendre2DPolynominal(m);
          }
          if (mParams->doMisalignmentRB && item.contains("rigidBody")) {
            auto rb = item["rigidBody"].get<std::vector<double>>();
            for (int k = 0; k < 6 && k < (int)rb.size(); ++k) {
              mRigidBodyParams[id](k) = rb[k];
            }
          }
        }
      }
    }
  }
}

void AlignmentSpec::buildHierarchy()
{
  if (mIsITS3) {
    mHierarchy = buildHierarchyIT3(mChip2Hiearchy);
  } else {
    mHierarchy = buildHierarchyITS(mChip2Hiearchy);
  }

  if (!mParams->dofConfigJson.empty()) {
    applyDOFConfig(mHierarchy.get(), mParams->dofConfigJson);
  }

  mHierarchy->finalise();
  if (mOutOpt[OutputOpt::MilleSteer]) {
    std::ofstream tree(mParams->milleTreeFile);
    mHierarchy->writeTree(tree);
    std::ofstream cons(mParams->milleConFile);
    mHierarchy->writeRigidBodyConstraints(cons);
    std::ofstream par(mParams->milleParamFile);
    mHierarchy->writeParameters(par);
  }
}

bool AlignmentSpec::getTransportJacobian(const TrackD& track, double xTo, double alphaTo, gbl::Matrix5d& jac, gbl::Matrix5d& err)
{
  auto prop = o2::base::PropagatorD::Instance();
  const auto bz = prop->getNominalBz();
  const auto minStep = std::sqrt(std::numeric_limits<double>::epsilon());
  const gbl::Vector5d x0(track.getParams());
  auto trackC = track;
  o2::track::TrackParD* refLin{nullptr};
  if (mParams->useStableRef) {
    refLin = &trackC;
  }

  auto propagate = [&](gbl::Vector5d& p) -> bool {
    TrackD tmp(track);
    for (int i{0}; i < track::kNParams; ++i) {
      tmp.setParam(p[i], i);
    }
    if (!prop->propagateToAlphaX(tmp, refLin, alphaTo, xTo, false, mParams->maxSnp, mParams->maxStep, 1, mParams->corrType)) {
      return false;
    }
    p = gbl::Vector5d(tmp.getParams());
    return true;
  };

  for (int iPar{0}; iPar < track::kNParams; ++iPar) {
    // step size
    double h = std::min(mParams->ridderMaxIniStep[iPar], std::max(minStep, std::abs(track.getParam(iPar)) * mParams->ridderRelIniStep[iPar]) * std::pow(mParams->ridderShrinkFac, mParams->ridderMaxExtrap / 2));
    ;
    // romberg tableu
    Eigen::MatrixXd cur(track::kNParams, mParams->ridderMaxExtrap);
    Eigen::MatrixXd pre(track::kNParams, mParams->ridderMaxExtrap);
    double normErr = std::numeric_limits<double>::max();
    gbl::Vector5d bestDeriv = gbl::Vector5d::Constant(std::numeric_limits<double>::max());
    for (int iExt{0}; iExt < mParams->ridderMaxExtrap; ++iExt) {
      gbl::Vector5d xPlus = x0, xMinus = x0;
      xPlus(iPar) += h;
      xMinus(iPar) -= h;
      if (!propagate(xPlus) || !propagate(xMinus)) {
        return false;
      }
      cur.col(0) = (xPlus - xMinus) / (2.0 * h);
      if (!iExt) {
        bestDeriv = cur.col(0);
      }
      // shrink step in next iteration
      h /= mParams->ridderShrinkFac;
      // richardson extrapolation
      double fac = mParams->ridderShrinkFac * mParams->ridderShrinkFac;
      for (int k{1}; k <= iExt; ++k) {
        cur.col(k) = (fac * cur.col(k - 1) - pre.col(k - 1)) / (fac - 1.0);
        fac *= mParams->ridderShrinkFac * mParams->ridderShrinkFac;
        double e = std::max((cur.col(k) - cur.col(k - 1)).norm(), (cur.col(k) - pre.col(k - 1)).norm());
        if (e <= normErr) {
          normErr = e;
          bestDeriv = cur.col(k);
          if (normErr < mParams->ridderEps) {
            break;
          }
        }
      }
      if (normErr < mParams->ridderEps) {
        break;
      }
      // check stability
      if (iExt > 0) {
        double tableauErr = (cur.col(iExt) - pre.col(iExt - 1)).norm();
        if (tableauErr >= 2.0 * normErr) {
          break;
        }
      }
      std::swap(cur, pre);
    }
    if (bestDeriv.isApproxToConstant(std::numeric_limits<double>::max())) {
      return false;
    }
    jac.col(iPar) = bestDeriv;
    err.col(iPar) = gbl::Vector5d::Constant(normErr);
  }

  if (jac.isIdentity(1e-8)) {
    LOGP(error, "Near jacobian idendity for taking track from {} to {}", track.getX(), xTo);
    return false;
  }

  return true;
}

bool AlignmentSpec::prepareITSTrack(int iTrk, const o2::its::TrackITS& itsTrack, align::Track& resTrack)
{
  const auto itsClRefs = mRecoData->getITSTracksClusterRefs();
  auto trFit = convertTrack<double>(itsTrack.getParamOut()); // take outer track fit as start of refit
  auto prop = o2::base::PropagatorD::Instance();
  auto geom = o2::its::GeometryTGeo::Instance();
  const auto bz = prop->getNominalBz();
  std::array<const FrameInfoExt*, 8> frameArr{};
  o2::track::TrackParD trkOut, *refLin = nullptr;
  if (mParams->useStableRef) {
    refLin = &(trkOut = trFit);
  }

  auto accountCluster = [&](int i, TrackD& tr, float& chi2, Measurement& meas, o2::track::TrackParD* refLin) {
    if (frameArr[i]) { // update with cluster
      if (!prop->propagateToAlphaX(tr, refLin, frameArr[i]->alpha, frameArr[i]->x, false, mParams->maxSnp, mParams->maxStep, 1, mParams->corrType)) {
        return 2;
      }
      meas.dy = frameArr[i]->positionTrackingFrame[0] - tr.getY();
      meas.dz = frameArr[i]->positionTrackingFrame[1] - tr.getZ();
      meas.sig2y = frameArr[i]->covarianceTrackingFrame[0];
      meas.sig2z = frameArr[i]->covarianceTrackingFrame[2];
      meas.z = tr.getZ();
      meas.phi = tr.getPhi();
      o2::math_utils::bringTo02Pid(meas.phi);
      chi2 += (float)tr.getPredictedChi2Quiet(frameArr[i]->positionTrackingFrame, frameArr[i]->covarianceTrackingFrame);
      if (!tr.update(frameArr[i]->positionTrackingFrame, frameArr[i]->covarianceTrackingFrame)) {
        return 2;
      }
      if (refLin) { // displace the reference to the last updated cluster
        refLin->setY(frameArr[i]->positionTrackingFrame[0]);
        refLin->setZ(frameArr[i]->positionTrackingFrame[1]);
      }
      return 0;
    }
    return 1;
  };

  FrameInfoExt pvInfo;
  if (mWithPV) { // add PV as constraint
    const int iPV = mT2PV[iTrk];
    if (iPV < 0) {
      return false;
    }
    const auto& pv = mPVs[iPV];
    auto tmp = convertTrack<double>(itsTrack.getParamIn());
    if (!prop->propagateToDCA(pv, tmp, bz)) {
      return false;
    }
    pvInfo.alpha = (float)tmp.getAlpha();
    double ca{0}, sa{0};
    o2::math_utils::bringToPMPid(pvInfo.alpha);
    o2::math_utils::sincosd(pvInfo.alpha, sa, ca);
    pvInfo.x = tmp.getX();
    pvInfo.positionTrackingFrame[0] = -pv.getX() * sa + pv.getY() * ca;
    pvInfo.positionTrackingFrame[1] = pv.getZ();
    pvInfo.covarianceTrackingFrame[0] = 0.5 * (pv.getSigmaX2() + pv.getSigmaY2());
    pvInfo.covarianceTrackingFrame[2] = pv.getSigmaY2();
    pvInfo.sens = -1;
    pvInfo.lr = -1;
    frameArr[0] = &pvInfo;
  }

  // collect all track clusters to array, placing them to layer+1 slot
  int nCl = itsTrack.getNClusters();
  for (int i = 0; i < nCl; i++) { // clusters are ordered from the outermost to the innermost
    const auto& curInfo = mITSTrackingInfo[itsClRefs[itsTrack.getClusterEntry(i)]];
    frameArr[1 + curInfo.lr] = &curInfo;
  }

  // start refit
  resTrack.points.clear();
  resTrack.info.clear();
  trFit.resetCovariance();
  trFit.setCov(trFit.getQ2Pt() * trFit.getQ2Pt() * trFit.getCov()[14], 14);
  float chi2{0};
  for (int i{7}; i >= 0; --i) {
    Measurement point;
    int res = accountCluster(i, trFit, chi2, point, refLin);
    if (res == 2) {
      return false;
    } else if (res == 0) {
      resTrack.points.push_back(point);
      resTrack.info.push_back(*frameArr[i]);
      resTrack.track = trFit; // put track to whatever the IU is
    }
  }
  // reverse inserted points so they are in the same order as the track
  std::reverse(resTrack.info.begin(), resTrack.info.end());
  std::reverse(resTrack.points.begin(), resTrack.points.end());
  resTrack.kfFit.chi2 = chi2;
  resTrack.kfFit.ndf = (int)resTrack.info.size() * 2 - 5;
  resTrack.kfFit.chi2Ndf = chi2 / (float)resTrack.kfFit.ndf;

  return true;
}

void AlignmentSpec::prepareMeasurments(std::span<const itsmft::CompClusterExt> clusters, std::span<const unsigned char> patterns)
{
  LOGP(info, "Preparing {} measurments", clusters.size());
  auto geom = its::GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  mITSTrackingInfo.clear();
  mITSTrackingInfo.reserve(clusters.size());
  auto pattIt = patterns.begin();
  for (const auto& cls : clusters) {
    const auto sens = cls.getSensorID();
    const auto lay = geom->getLayer(sens);
    double sigmaY2{0}, sigmaZ2{0};
    math_utils::Point3D<float> locXYZ;
    if (mIsITS3) {
      locXYZ = o2::its3::ioutils::extractClusterData(cls, pattIt, mIT3Dict, sigmaY2, sigmaZ2);
    } else {
      locXYZ = o2::its::ioutils::extractClusterData(cls, pattIt, mITSDict, sigmaY2, sigmaZ2);
    }
    sigmaY2 += mParams->extraClsErrY[lay] * mParams->extraClsErrY[lay];
    sigmaZ2 += mParams->extraClsErrZ[lay] * mParams->extraClsErrZ[lay];
    // Transformation to the local --> global
    const auto gloXYZ = geom->getMatrixL2G(sens) * locXYZ;
    // Inverse transformation to the local --> tracking
    auto trkXYZf = geom->getMatrixT2L(sens) ^ locXYZ;
    o2::math_utils::Point3D<double> trkXYZ;
    trkXYZ.SetCoordinates(trkXYZf.X(), trkXYZf.Y(), trkXYZf.Z());
    // Tracking alpha angle
    // We want that each cluster rotates its tracking frame to the clusters phi
    // that way the track linearization around the measurement is less biases to the arc
    // this means automatically that the measurement on the arc is at 0 for the curved layers
    double alpha = geom->getSensorRefAlpha(sens);
    double x = trkXYZ.x();
    if (mIsITS3 && constants::detID::isDetITS3(sens)) {
      trkXYZ.SetY(0.f);
      // alpha&x always have to be defined wrt to the global Z axis!
      x = std::hypot(gloXYZ.x(), gloXYZ.y());
      trkXYZ.SetX(x);
      alpha = std::atan2(gloXYZ.y(), gloXYZ.x());
      auto chip = constants::detID::getSensorID(sens);
      sigmaY2 += mParams->extraClsErrY[chip] * mParams->extraClsErrY[chip];
      sigmaZ2 += mParams->extraClsErrZ[chip] * mParams->extraClsErrZ[chip];
    }
    math_utils::bringToPMPid(alpha);
    mITSTrackingInfo.emplace_back(sens, lay, x, alpha,
                                  std::array<double, 2>{trkXYZ.y(), trkXYZ.z()},
                                  std::array<double, 3>{sigmaY2, 0., sigmaZ2});
  }
}

void AlignmentSpec::buildT2V()
{
  const auto& itsTracks = mRecoData->getITSTracks();
  mT2PV.clear();
  mT2PV.resize(itsTracks.size(), -1);
  if (mUseMC) {
    mPVs.reserve(mcReader->getNEvents(0));
    for (int iEve{0}; iEve < mcReader->getNEvents(0); ++iEve) {
      const auto& eve = mcReader->getMCEventHeader(0, iEve);
      dataformats::VertexBase vtx;
      constexpr float err{22e-4f};
      vtx.setX((float)eve.GetX());
      vtx.setY((float)eve.GetY());
      vtx.setZ((float)eve.GetZ());
      vtx.setSigmaX(err);
      vtx.setSigmaY(err);
      vtx.setSigmaZ(err);
      mPVs.push_back(vtx);
    }
    const auto& mcLbls = mRecoData->getITSTracksMCLabels();
    for (size_t iTrk{0}; iTrk < mcLbls.size(); ++iTrk) {
      const auto& lbl = mcLbls[iTrk];
      if (!lbl.isValid() || !lbl.isCorrect()) {
        continue;
      }
      const auto& mcTrk = mcReader->getTrack(lbl);
      if (mcTrk->isPrimary()) {
        mT2PV[iTrk] = lbl.getEventID();
      }
    }
  } else {
    LOGP(fatal, "Data PV to track TODO");
  }
}

bool AlignmentSpec::applyMisalignment(Eigen::Vector2d& res, const FrameInfoExt& frame, const TrackD& wTrk, size_t iTrk)
{
  if (!constants::detID::isDetITS3(frame.sens)) {
    return true;
  }

  const int sensorID = constants::detID::getSensorID(frame.sens);
  const int layerID = constants::detID::getDetID2Layer(frame.sens);

  // --- Legendre deformation (non-rigid-body) ---
  if (mParams->doMisalignmentLeg && mIsITS3 && mUseMC) {
    const auto prop = o2::base::PropagatorD::Instance();

    const auto lbl = mRecoData->getITSTracksMCLabels()[iTrk];
    const auto mcTrk = mcReader->getTrack(lbl);
    if (!mcTrk) {
      return false;
    }
    std::array<double, 3> xyz{mcTrk->GetStartVertexCoordinatesX(), mcTrk->GetStartVertexCoordinatesY(), mcTrk->GetStartVertexCoordinatesZ()};
    std::array<double, 3> pxyz{mcTrk->GetStartVertexMomentumX(), mcTrk->GetStartVertexMomentumY(), mcTrk->GetStartVertexMomentumZ()};
    TParticlePDG* pPDG = TDatabasePDG::Instance()->GetParticle(mcTrk->GetPdgCode());
    if (!pPDG) {
      return false;
    }
    o2::track::TrackParD mcPar(xyz, pxyz, TMath::Nint(pPDG->Charge() / 3), false);

    const double r = frame.x;
    const double gloX = r * std::cos(frame.alpha);
    const double gloY = r * std::sin(frame.alpha);
    const double gloZ = frame.positionTrackingFrame[1];
    auto [u, v] = computeUV(gloX, gloY, gloZ, sensorID, constants::radii[layerID]);
    const double h = mDeformations[sensorID](u, v);

    auto mcAtCl = mcPar;
    if (!mcAtCl.rotate(frame.alpha) || !prop->PropagateToXBxByBz(mcAtCl, frame.x)) {
      return false;
    }

    const double snp = mcAtCl.getSnp();
    const double tgl = mcAtCl.getTgl();
    const double csci = 1. / std::sqrt(1. - (snp * snp));
    const double dydx = snp * csci;
    const double dzdx = tgl * csci;
    const double dy = dydx * h;
    const double dz = dzdx * h;

    const double newGloY = (r * std::sin(frame.alpha)) + (dy * std::cos(frame.alpha));
    const double newGloX = (r * std::cos(frame.alpha)) - (dy * std::sin(frame.alpha));
    const double newGloZ = gloZ + dz;
    auto [uNew, vNew] = computeUV(newGloX, newGloY, newGloZ, sensorID, constants::radii[layerID]);
    if (std::abs(uNew) > 1. || std::abs(vNew) > 1.) {
      return false;
    }

    res[0] += dy;
    res[1] += dz;
  }

  // --- Rigid body misalignment ---
  // Must use the same derivative chain as GBL:
  //   dres/da_parent = dres/da_TRK * J_L2T_tile * J_L2P_tile
  // The tile is a pseudo-volume; Millepede fits at the halfBarrel (parent) level.
  if (mParams->doMisalignmentRB) {
    GlobalLabel lbl(0, frame.sens, true);
    if (mChip2Hiearchy.find(lbl) == mChip2Hiearchy.end()) {
      return true; // sensor not in hierarchy, skip
    }
    const auto* tileVol = mChip2Hiearchy.at(lbl);

    // derivative in TRK frame (3x6: rows = dy, dz, dsnp)
    Matrix36 der = getRigidBodyDerivatives(wTrk);

    // TRK -> tile LOC
    const double posTrk[3] = {frame.x, 0., 0.};
    double posLoc[3];
    tileVol->getT2L().LocalToMaster(posTrk, posLoc);
    Matrix66 jacL2T;
    tileVol->computeJacobianL2T(posLoc, jacL2T);
    der *= jacL2T;

    // tile LOC -> halfBarrel LOC (same chain as GBL hierarchy walk)
    der *= tileVol->getJL2P();

    // apply: delta_res = der * delta_a_halfBarrel
    Eigen::Vector3d shift = der * mRigidBodyParams[sensorID];
    res[0] += shift[0]; // dy
    res[1] += shift[1]; // dz
  }

  return true;
}

void AlignmentSpec::endOfStream(EndOfStreamContext& /*ec*/)
{
  mDBGOut->Close();
  mDBGOut.reset();
}

void AlignmentSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "its cluster dictionary updated";
    mITSDict = (const o2::itsmft::TopologyDictionary*)obj;
    return;
  }
  if (matcher == ConcreteDataMatcher("IT3", "CLUSDICT", 0)) {
    LOG(info) << "it3 cluster dictionary updated";
    mIT3Dict = (const o2::its3::TopologyDictionary*)obj;
    return;
  }
}

DataProcessorSpec getAlignmentSpec(GTrackID::mask_t srcTracks, GTrackID::mask_t srcClusters, bool useMC, bool withPV, bool withITS, OutputEnum out)
{
  auto dataRequest = std::make_shared<DataRequest>();
  std::shared_ptr<o2::base::GRPGeomRequest> ggRequest{nullptr};
  if (!out[OutputOpt::MilleRes]) {
    dataRequest->requestTracks(srcTracks, useMC);
    if (!withITS) {
      dataRequest->requestIT3Clusters(useMC);
    } else {
      dataRequest->requestClusters(srcClusters, useMC);
    }
    if (withPV && !useMC) {
      dataRequest->requestPrimaryVertices(useMC);
    }
    ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                           false,                             // GRPECS=true
                                                           true,                              // GRPLHCIF
                                                           true,                              // GRPMagField
                                                           true,                              // askMatLUT
                                                           o2::base::GRPGeomRequest::Aligned, // geometry
                                                           dataRequest->inputs,               // inputs
                                                           true,                              // askOnce
                                                           true);                             // propagatorD
  } else {
    dataRequest->inputs.emplace_back("dummy", "GLO", "DUMMY_OUT", 0);
    ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                           false,                             // GRPECS=true
                                                           false,                             // GRPLHCIF
                                                           false,                             // GRPMagField
                                                           false,                             // askMatLUT
                                                           o2::base::GRPGeomRequest::Aligned, // geometry
                                                           dataRequest->inputs);
  }

  Options opts{
    {"nthreads", VariantType::Int, 1, {"number of threads"}},
  };

  return DataProcessorSpec{
    .name = "its3-alignment",
    .inputs = dataRequest->inputs,
    .outputs = {},
    .algorithm = AlgorithmSpec{adaptFromTask<AlignmentSpec>(dataRequest, ggRequest, srcTracks, useMC, withPV, withITS, out)},
    .options = opts};
}
} // namespace o2::its3::align
