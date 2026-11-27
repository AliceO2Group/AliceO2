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

#include "StrangenessTracking/StrangenessTracker.h"
#include "StrangenessTracking/ITSClusterBinRange.h"
#include "Framework/Logger.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/TrackHelpers.h"
#include "ReconstructionDataFormats/DCA.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "DetectorsBase/GlobalParams.h"
#include "DataFormatsParameters/GRPECSObject.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "ITSMFTTracking/ITSTrackingConfigParam.h"
#include "DataFormatsITSMFT/DPLAlpideParam.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <tuple>

namespace o2::strangeness_tracking
{
namespace
{
using Track = o2::track::TrackParCovF;
// Fit/search bookkeeping is transient. Public device outputs reuse StrangeTrack
// and TrackITS rather than persisting this implementation-specific state.
struct ClusterCandidate {
  Track mTrack, mOuter;
  o2::dataformats::StrangeTrack source{};
  int mPVRef = -1;
  std::array<int, 7> mClusterRefs{{-1, -1, -1, -1, -1, -1, -1}};
  double mTimeMinBC = 0., mTimeMaxBC = 0.;
  float mHitChi2 = 0.f;
  uint8_t mMissingLayerMask = 0;
  int getNClusters() const
  {
    return std::count_if(mClusterRefs.begin(), mClusterRefs.end(), [](int id) { return id >= 0; });
  }
};
struct Hit {
  int ref = -1, layer = -1;
  double lower = 0, upper = 0;
};
// Reject invalid states before they can participate in the global ordering.
bool valid(const Track& t)
{
  if (std::abs(t.getSnp()) >= 1.f) {
    return false;
  }
  // A Cholesky check also catches finite but non-positive covariance matrices.
  double l[5][5]{};
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j <= i; ++j) {
      double v = t.getCov()[i * (i + 1) / 2 + j];
      for (int k = 0; k < j; ++k) {
        v -= l[i][k] * l[j][k];
      }
      if (i == j) {
        if (v <= 0.) {
          return false;
        }
        l[i][j] = std::sqrt(v);
      } else {
        l[i][j] = v / l[j][j];
      }
    }
  }
  return true;
}
int holes(const ClusterCandidate& t) { return std::popcount(unsigned(t.mMissingLayerMask)); }
} // namespace

void StrangenessTracker::remapExtensionSeeds(int thread, const std::vector<int>& v0, const std::vector<int>& casc, const std::vector<int>& three)
{
  mExtensionSeedsRemapped = true;
  for (auto& seed : mExtensionSeeds[thread]) {
    for (const auto& accepted : mStrangeTrackVec[thread]) {
      seed.matched |= accepted.mPartType == seed.source.mPartType && accepted.mDecayRef == seed.source.mDecayRef;
    }
    const auto& map = seed.source.mPartType == o2::dataformats::kStrkV0 ? v0 : seed.source.mPartType == o2::dataformats::kStrkCascade ? casc
                                                                                                                                      : three;
    seed.source.mDecayRef = map.at(seed.source.mDecayRef);
  }
}

void StrangenessTracker::finishClusterTracking()
{
  if (!clusterTrackingEnabled() || mClusterTrackingFinished) {
    return;
  }
  mClusterTrackingFinished = true;
  if (!mClusterTrackingData) {
    throw std::runtime_error("ITS extension/short tracking has no current reconstruction inputs");
  }
  const auto* grp = o2::base::GRPGeomHelper::instance().getGRPECS();
  if (!grp || !grp->isDetContinuousReadOut(o2::detectors::DetID::ITS)) {
    throw std::runtime_error("ITS extension/short tracking currently requires continuous ITS readout and GRPECS");
  }
#ifdef ENABLE_UPGRADES
  if (o2::GlobalParams::Instance().withITS3) {
    throw std::runtime_error("ITS extension/short tracking currently supports ITS2 geometry only");
  }
#endif
  const auto& data = *mClusterTrackingData;
  const auto& cfg = *mStrParams;
  if (cfg.extensionMinHits < 2 || cfg.extensionMinHits > 7 || cfg.extensionMaxMissing < 0 || cfg.extensionMaxMissing > 1 ||
      mPVTimeMargin < 0 || !(cfg.clusterSearchNSigma > 0) || !(cfg.extensionMaxHitChi2 > 0) || cfg.extensionMaxBranches < 1) {
    throw std::runtime_error("Invalid ITS extension/short tracking configuration");
  }
  if (!cfg.itsOnly && !data.hasITSABClusterUsage()) {
    throw std::runtime_error("ITS extension/short tracking requires afterburner references (or explicit itsOnly profile)");
  }
  mStrangenessTrackingStats.bz = mBz;
  mStrangenessTrackingStats.itsOnly = cfg.itsOnly;
  mStrangenessTrackingStats.clustersPerLayer = data.getITSPerLayer();
  mStrangenessTrackingStats.decayFollowingEnabled = cfg.enableDecayFollowing;
  mStrangenessTrackingStats.tripletsEnabled = cfg.enableShortIB;
  mStrangenessTrackingStats.tripletsDeferredZeroField = cfg.enableShortIB && std::abs(mBz) < 0.01f;
  auto* geom = o2::its::GeometryTGeo::Instance();
  const o2::its::TrackingParameters itsParams;
  o2::its::IndexTableUtils<7> utils;
  utils.setTrackingParameters(itsParams);
  const auto& alpide = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
  const auto& itsCfg = o2::its::TrackerParamConfig::Instance();
  std::array<o2::itsmft::tracking::LayerTiming, 7> timing;
  std::vector<Hit> hits(mInputITSclusters.size());
  // Index (timing ROF, phi bin) directly within each layer and keep only
  // occupied hits in the Z-ordered columns.
  struct BinHit {
    int z;
    int flat;
  };
  const int nPhiBins = utils.getNphiBins();
  std::array<std::vector<std::vector<BinHit>>, 7> grid;
  std::array<float, 7> radiusMin, radiusMax, maxErrY{}, maxErrZ{};
  for (int layer = 0; layer < 7; ++layer) {
    radiusMin[layer] = radiusMax[layer] = itsParams.LayerRadii[layer];
    if (alpide.getROFLengthInBC(layer) <= 0 || alpide.getROFBiasInBC(layer) < 0 || alpide.getROFDelayInBC(layer) < 0 || itsCfg.addTimeError[layer] < 0) {
      throw std::runtime_error("Invalid configured ITS timing for cluster tracking");
    }
    const auto length = alpide.getROFLengthInBC(layer);
    if (o2::constants::lhc::LHCMaxBunches % length != 0) {
      throw std::runtime_error("ITS extension/short tracking ROF length must divide an orbit");
    }
    const auto nROFs = (o2::constants::lhc::LHCMaxBunches / length) * o2::base::GRPGeomHelper::getNHBFPerTF();
    if (nROFs <= 0) {
      throw std::runtime_error("ITS extension/short tracking requires the configured TF orbit count");
    }
    timing[layer] = {uint32_t(nROFs), uint32_t(alpide.getROFLengthInBC(layer)), uint32_t(alpide.getROFDelayInBC(layer)),
                     uint32_t(alpide.getROFBiasInBC(layer)), uint32_t(itsCfg.addTimeError[layer])};
    if (cfg.enableDecayFollowing) {
      grid[layer].resize(size_t(nROFs) * nPhiBins);
    }
  }
  const int nSlots = mInputITSclusters.getNLayers();
  if (nSlots == 1) {
    for (int layer = 1; layer < 7; ++layer) {
      if (timing[layer].mROFLength != timing[0].mROFLength || timing[layer].mROFBias != timing[0].mROFBias || timing[layer].mROFDelay != timing[0].mROFDelay) {
        throw std::runtime_error("Staggered ITS extension/short tracking timing requires layer-separated cluster inputs");
      }
    }
  }
  // The ROF ordinal is the timing index, so the per-TF ROF set must be complete.
  // The ITS decoder/clusterer normally guarantees it (ensureContinuousROF); if
  // an input violates it, skip extension/short tracking for this TF only and
  // keep the legacy output rather than aborting the device.
  for (int slot = 0; slot < nSlots; ++slot) {
    const auto nROFs = data.getITSClustersROFRecords(slot).size();
    if (nROFs != timing[slot].mNROFsTF) {
      LOGP(warning, "ITS extension/short tracking skipped for this TF: {} ROFs in cluster stream {}, {} expected from TF timing",
           nROFs, slot, timing[slot].mNROFsTF);
      return;
    }
  }
  for (int slot = 0; slot < nSlots; ++slot) {
    const auto rofs = data.getITSClustersROFRecords(slot);
    for (int ir = 0; ir < int(rofs.size()); ++ir) {
      const auto& rof = rofs[ir];
      for (int local = rof.getFirstEntry(); local < rof.getFirstEntry() + rof.getNEntries(); ++local) {
        if (local < 0 || local >= mInputITSclusters.getNClusters(slot)) {
          throw std::runtime_error("Invalid extension/short tracking ROF cluster range");
        }
        const int ref = o2::itsmft::composeClusID(slot, local);
        const int flat = mInputITSclusters.flatIndex(ref);
        const auto& cl = mInputITSclusters[ref];
        const int layer = geom->getLayer(cl.getSensorID());
        if (layer < 0 || layer >= 7 || (nSlots > 1 && layer != slot)) {
          throw std::runtime_error("Invalid ITS extension/short tracking layer mapping");
        }
        // ITS tracking uses the supplied ROF ordinal. Effective bias/delay is
        // applied by LayerTiming, not subtracted from ROFRecord acquisition IR.
        auto& h = hits[flat];
        if (h.ref >= 0) {
          throw std::runtime_error("Overlapping ITS extension/short tracking ROF ranges");
        }
        const auto bounds = timing[layer].getROFTimeBounds(ir, true);
        h = {ref, layer, double(bounds.lower()), double(bounds.upper())};
        if (!cfg.enableDecayFollowing) {
          continue;
        }
        const auto p = geom->getMatrixT2G(cl.getSensorID())(o2::math_utils::Point3D<float>{cl.getX(), cl.getY(), cl.getZ()});
        const float r = std::hypot(p.X(), p.Y());
        radiusMin[layer] = std::min(radiusMin[layer], r);
        radiusMax[layer] = std::max(radiusMax[layer], r);
        maxErrY[layer] = std::max(maxErrY[layer], cl.getSigmaY2());
        maxErrZ[layer] = std::max(maxErrZ[layer], cl.getSigmaZ2());
        const float phi = o2::its::math_utils::getNormalizedPhi(std::atan2(p.Y(), p.X()));
        const int pb = std::clamp(utils.getPhiBinIndex(phi), 0, nPhiBins - 1);
        const int zb = std::clamp(utils.getZBinIndex(layer, p.Z()), 0, utils.getNzBins() - 1);
        grid[layer][size_t(ir) * nPhiBins + pb].push_back({zb, flat});
      }
    }
  }
  for (auto& layer : grid) {
    for (auto& column : layer) {
      std::sort(column.begin(), column.end(), [](const BinHit& a, const BinHit& b) { return a.z < b.z; });
    }
  }
  for (const auto& h : hits) {
    if (h.ref < 0) {
      throw std::runtime_error("ITS extension/short tracking input cluster has no ROF");
    }
  }
  std::vector<uint8_t> used(hits.size(), 0);
  auto mark = [&](int ref) {
    if (ref < 0) {
      throw std::runtime_error("Invalid upstream ITS extension/short tracking cluster reference");
    }
    const int slot = o2::itsmft::clusID2Layer(ref), index = o2::itsmft::clusID2Index(ref);
    if (slot >= nSlots || index >= mInputITSclusters.getNClusters(slot)) {
      throw std::runtime_error("Invalid upstream ITS extension/short tracking cluster reference");
    }
    used[mInputITSclusters.flatIndex(ref)] = 1;
  };
  auto markUpstreamHits = [&] {
    for (const auto& tr : mInputITStracks) {
      if (tr.getFirstClusterEntry() < 0 || tr.getFirstClusterEntry() + tr.getNumberOfClusters() > mInputITSidxs.size()) {
        throw std::runtime_error("Invalid ITS track cluster span");
      }
      for (int i = 0; i < tr.getNumberOfClusters(); ++i) {
        mark(mInputITSidxs[tr.getFirstClusterEntry() + i]);
      }
    }
    if (data.hasITSABClusterUsage()) {
      const auto refs = data.getITSABClusterRefs();
      for (const auto& ab : data.getITSABRefs()) {
        if (ab.getFirstEntry() + ab.getNClusters() > refs.size()) {
          throw std::runtime_error("Invalid afterburner cluster span");
        }
        for (int i = 0; i < ab.getNClusters(); ++i) {
          mark(refs[ab.getFirstEntry() + i]);
        }
      }
    }
  };
  if (!cfg.decayFollowingOnly) {
    markUpstreamHits();
  }
  const auto vertices = data.getPrimaryVertices();
  std::vector<std::pair<double, double>> pvWindows(vertices.size(), {0., 0.});
  for (int pv = 0; pv < vertices.size(); ++pv) {
    const auto& v = vertices[pv];
    if (v.getIRMin().isDummy() || v.getIRMax().isDummy() || v.getIRMax() < v.getIRMin()) {
      ++mStrangenessTrackingStats.nInvalidPVTime;
      continue;
    }
    const double margin = mPVTimeMargin / o2::constants::lhc::LHCBunchSpacingMUS;
    pvWindows[pv] = {v.getIRMin().differenceInBC(data.startIR) - 0.5 - margin,
                     v.getIRMax().differenceInBC(data.startIR) + 0.5 + margin};
  }
  auto pvTime = [&](int pv) { return pvWindows[pv]; };
  auto query = [&](int layer, float phi, float z, float dphi, float dz, double lo, double hi) {
    std::vector<int> result;
    if (!(hi > lo) || dphi < 0 || dz < 0) {
      return result;
    }
    const auto rr = timing[layer].getROFRange(float(lo), float(hi));
    const auto rect = o2::its::getBinsRect(layer, phi, z, dz, dphi, utils);
    const auto bins = clustertracking::makeBinRange(rect.x, rect.y, rect.z, rect.w,
                                                    utils.getNzBins(), utils.getNphiBins(), dphi);
    if (bins.empty()) {
      return result;
    }
    for (int offset = 0; offset < bins.phiCount(); ++offset) {
      const int p = bins.phiBin(offset);
      for (int clock = rr.getFirstEntry(); clock < rr.getFirstEntry() + rr.getEntries(); ++clock) {
        const auto& column = grid[layer][size_t(clock) * nPhiBins + p];
        auto it = std::lower_bound(column.begin(), column.end(), bins.firstZ,
                                   [](const BinHit& hit, int zBin) { return hit.z < zBin; });
        for (; it != column.end() && it->z <= bins.lastZ; ++it) {
          const int flat = it->flat;
          if (!used[flat] && hits[flat].lower < hi && hits[flat].upper > lo) {
            result.push_back(flat);
          }
        }
      }
    }
    std::sort(result.begin(), result.end(), [&](int a, int b) { return hits[a].ref < hits[b].ref; });
    return result;
  };
  auto attach = [&](ClusterCandidate& out, int flat) {
    const auto& h = hits[flat];
    out.mClusterRefs[h.layer] = h.ref;
    out.mTimeMinBC = std::max(out.mTimeMinBC, h.lower);
    out.mTimeMaxBC = std::min(out.mTimeMaxBC, h.upper);
  };
  auto advance = [&](Track& tr, float x) {
    return o2::base::Propagator::Instance()->propagateToX(tr, x, mBz, o2::base::PropagatorF::MAX_SIN_PHI, o2::base::PropagatorF::MAX_STEP, mCorrType);
  };
  struct Proposal {
    ClusterCandidate track;
    std::array<uint32_t, 4> key;
  };
  std::vector<Proposal> proposals;
  std::set<std::array<uint32_t, 4>> matched;
  std::vector<ExtensionSeed> seeds;
  for (int thread = 0; thread < mNThreads; ++thread) {
    for (auto seed : mExtensionSeeds[thread]) {
      // Standalone IDs are already final. Embedded IDs were remapped and the
      // accepted-association result captured before the legacy output merge.
      if (!seed.matched && !mExtensionSeedsRemapped) {
        for (const auto& accepted : mStrangeTrackVec[thread]) {
          seed.matched |= accepted.mPartType == seed.source.mPartType && accepted.mDecayRef == seed.source.mDecayRef;
        }
      }
      if (seed.matched) {
        matched.insert(seed.key);
      }
      seeds.push_back(seed);
    }
  }
  mStrangenessTrackingStats.nSeeds = seeds.size();
  std::sort(seeds.begin(), seeds.end(), [](const auto& a, const auto& b) { return std::tie(a.key, a.pv, a.source.mDecayRef) < std::tie(b.key, b.pv, b.source.mDecayRef); });
  struct FollowBranch : ClusterCandidate {
    Track lastMeasured;
  };
  for (const auto& seed : seeds) {
    if (matched.count(seed.key) || seed.pv < 0 || seed.pv >= vertices.size() || !valid(seed.source.mMother)) {
      continue;
    }
    ++mStrangenessTrackingStats.nUnmatched;
    const auto [lo, hi] = pvTime(seed.pv);
    if (!(hi > lo)) {
      continue;
    }
    FollowBranch initial;
    initial.source = seed.source;
    initial.mTrack = seed.source.mMother;
    initial.lastMeasured = seed.source.mMother;
    initial.mPVRef = seed.pv;
    initial.mTimeMinBC = lo;
    initial.mTimeMaxBC = hi;
    std::vector<FollowBranch> branches{initial};
    const auto xyz = seed.source.mMother.getXYZGlo();
    const float decayR = std::hypot(xyz.X(), xyz.Y());
    for (int layer = 6; layer >= 0 && !branches.empty(); --layer) {
      if (radiusMin[layer] >= decayR) {
        continue;
      }
      std::vector<FollowBranch> next;
      for (const auto& branch : branches) {
        auto projected = branch.mTrack;
        float x;
        const auto here = projected.getXYZGlo();
        const float targetR = std::min(itsParams.LayerRadii[layer], std::hypot(here.X(), here.Y()) - 1.e-4f);
        if (!projected.getXatLabR(targetR, x, mBz, o2::track::DirInward) || !advance(projected, x)) {
          ++mStrangenessTrackingStats.nPropagationFailures;
          continue; // propagation failure is not a missing hit
        }
        if (!valid(projected)) {
          ++mStrangenessTrackingStats.nPropagationFailures;
          continue;
        }
        const auto p = projected.getXYZGlo();
        if (std::abs(p.Z()) > itsParams.LayerZ[layer]) {
          next.push_back(branch); // no active layer crossing
          continue;
        }
        const float r = std::hypot(p.X(), p.Y());
        if (!(r > 0)) {
          ++mStrangenessTrackingStats.nPropagationFailures;
          continue;
        }
        const float dr = radiusMax[layer] - radiusMin[layer];
        const float csp = std::sqrt(std::max(1.e-6f, 1.f - projected.getSnp() * projected.getSnp()));
        const float dy = cfg.clusterSearchNSigma * std::sqrt(projected.getSigmaY2() + maxErrY[layer]) + dr * (1.f + std::abs(projected.getSnp()) / csp);
        float dz = cfg.clusterSearchNSigma * std::sqrt(projected.getSigmaZ2() + maxErrZ[layer]) + dr * std::abs(projected.getTgl()) / csp;
        float dphi = std::min(float(M_PI), dy / r);
        const float ca = std::cos(projected.getAlpha()), sa = std::sin(projected.getAlpha());
        const float radialCos = std::abs((p.X() * (ca * csp - sa * projected.getSnp()) +
                                          p.Y() * (sa * csp + ca * projected.getSnp())) /
                                         r);
        // Local-frame csp is not the incidence cosine at a cylinder. Bound the
        // radius-to-sensor displacement and covariance projection with the
        // radial incidence instead. Sums of standard deviations also cover
        // unknown YZ correlations. Near a turning/grazing crossing, query the
        // full layer rather than let a singular linearization exclude hits.
        const float radialVariation = dr * (std::abs(projected.getCurvature(mBz)) + 1.f / radiusMin[layer]);
        if (radialCos < 1.e-3f || radialCos * radialCos <= 4.f * radialVariation) {
          dphi = float(M_PI);
          dz = std::max(dz, itsParams.LayerZ[layer] + std::abs(p.Z()));
        } else {
          const float sigmaY = std::sqrt(projected.getSigmaY2()) + std::sqrt(maxErrY[layer]);
          const float sigmaZ = std::sqrt(projected.getSigmaZ2()) + std::sqrt(maxErrZ[layer]);
          const float transverseSpan = (cfg.clusterSearchNSigma * sigmaY + dr) / radialCos;
          dphi = std::min(float(M_PI), std::max(dphi, transverseSpan / r));
          dz = std::max(dz, cfg.clusterSearchNSigma * sigmaZ + std::abs(projected.getTgl()) * transverseSpan);
        }
        const auto candidates = query(layer, std::atan2(p.Y(), p.X()), p.Z(), dphi, dz, branch.mTimeMinBC, branch.mTimeMaxBC);
        for (int flat : candidates) {
          auto candidate = branch;
          auto& tr = candidate.mTrack;
          const auto& cl = mInputITSclusters[hits[flat].ref];
          if (!tr.rotate(geom->getSensorRefAlpha(cl.getSensorID()))) {
            ++mStrangenessTrackingStats.nPropagationFailures;
            continue;
          }
          if (cl.getX() > tr.getX()) {
            continue;
          } // only measurements crossed inward of the branch
          if (!advance(tr, cl.getX())) {
            ++mStrangenessTrackingStats.nPropagationFailures;
            continue;
          }
          if (mCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE &&
              !tr.correctForMaterial(itsParams.LayerxX0[layer], itsParams.LayerxX0[layer] * o2::its::constants::Radl * o2::its::constants::Rho, true)) {
            ++mStrangenessTrackingStats.nPropagationFailures;
            continue;
          }
          const std::array<float, 2> pos{cl.getY(), cl.getZ()};
          const std::array<float, 3> cov{cl.getSigmaY2(), cl.getSigmaYZ(), cl.getSigmaZ2()};
          const float chi2 = tr.getPredictedChi2(pos, cov);
          if (chi2 < 0 || chi2 >= cfg.extensionMaxHitChi2 || !tr.update(pos, cov) || !valid(tr)) {
            continue;
          }
          attach(candidate, flat);
          candidate.mHitChi2 += chi2;
          candidate.lastMeasured = tr;
          next.push_back(candidate);
        }
        // Retain a missing-hit alternative even when a hit alternative exists.
        if (holes(branch) < cfg.extensionMaxMissing) {
          auto missing = branch;
          if (mCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE &&
              !projected.correctForMaterial(itsParams.LayerxX0[layer], itsParams.LayerxX0[layer] * o2::its::constants::Radl * o2::its::constants::Rho, true)) {
            ++mStrangenessTrackingStats.nPropagationFailures;
            continue;
          }
          if (!valid(projected)) {
            ++mStrangenessTrackingStats.nPropagationFailures;
            continue;
          }
          missing.mTrack = projected;
          missing.mMissingLayerMask |= 1 << layer;
          next.push_back(missing);
          ++mStrangenessTrackingStats.nMissingHits;
        }
      }
      // Bound the combinatorics: every layer can multiply the hypotheses (and a
      // grazing crossing queries the full layer). Keep the best ones with the
      // same quality ordering as the final selection, deterministically.
      if (next.size() > size_t(cfg.extensionMaxBranches)) {
        std::stable_sort(next.begin(), next.end(), [](const FollowBranch& a, const FollowBranch& b) {
          if (a.getNClusters() != b.getNClusters()) {
            return a.getNClusters() > b.getNClusters();
          }
          if (holes(a) != holes(b)) {
            return holes(a) < holes(b);
          }
          return std::tie(a.mHitChi2, a.mClusterRefs) < std::tie(b.mHitChi2, b.mClusterRefs);
        });
        mStrangenessTrackingStats.nMotherTruncated += next.size() - cfg.extensionMaxBranches;
        next.resize(cfg.extensionMaxBranches);
      }
      branches.swap(next);
    }
    for (auto& tr : branches) {
      if (tr.getNClusters() < cfg.extensionMinHits || !(tr.mTimeMaxBC > tr.mTimeMinBC)) {
        continue;
      }
      tr.mTrack = tr.lastMeasured; // persist the innermost successful measurement surface
      proposals.push_back({tr, seed.key});
    }
  }
  auto commit = [&](const ClusterCandidate& tr, bool shortTrack) {
    std::map<o2::MCCompLabel, unsigned> labelLayers;
    for (int layer = 0; layer < tr.mClusterRefs.size(); ++layer) {
      const int ref = tr.mClusterRefs[layer];
      if (ref < 0) {
        continue;
      }
      mark(ref);
      if (mMCTruthON) {
        const auto* labels = data.getITSClustersMCLabels(o2::itsmft::clusID2Layer(ref));
        if (!labels) {
          throw std::runtime_error("Missing per-stream extension/short tracking cluster MC labels");
        }
        for (auto label : labels->getLabels(o2::itsmft::clusID2Index(ref))) {
          if (label.isValid()) {
            label.setFakeFlag(false);
            labelLayers[label] |= 1u << layer;
          }
        }
      }
    }
    o2::MCCompLabel best;
    unsigned bestLayers = 0;
    for (const auto& [label, layers] : labelLayers) {
      if (std::popcount(layers) > std::popcount(bestLayers)) {
        best = label;
        bestLayers = layers;
      }
    }
    if (best.isValid()) {
      best.setFakeFlag(std::popcount(bestLayers) != tr.getNClusters());
    }
    const o2::its::TimeStamp time{float(0.5 * (tr.mTimeMinBC + tr.mTimeMaxBC)), float(0.5 * (tr.mTimeMaxBC - tr.mTimeMinBC))};
    auto clusterSize = [&](int layer) {
      const int ref = tr.mClusterRefs[layer];
      return ref < 0 ? 0 : std::clamp(mInputClusterSizes[mInputITSclusters.flatIndex(ref)], 0, 255);
    };
    if (!shortTrack) {
      auto output = tr.source;
      output.mMother = tr.mTrack;
      output.mITSRef = -1u;
      output.mMatchChi2 = -1.f;
      output.mTopoChi2 = tr.mHitChi2;
      output.mClusterSizes = 0;
      for (int layer = 0; layer < 7; ++layer) {
        output.setClusterSize(layer, clusterSize(layer));
      }
      mExtendedStrangeTracks.push_back(output);
      mExtendedStrangeTimes.push_back(time);
      if (mMCTruthON) {
        mExtendedStrangeLabels.push_back(best);
      }
    } else {
      TrackITS output(tr.mTrack, tr.mHitChi2, tr.mOuter);
      output.setClusterRefs(mITSShortClusterRefs.size(), 3);
      output.setPattern(0x7u | (bestLayers << 16));
      output.getTimeStamp() = time;
      for (int layer = 0; layer < 3; ++layer) {
        output.setClusterSize(layer, clusterSize(layer));
        mITSShortClusterRefs.push_back(tr.mClusterRefs[layer]);
      }
      mITSShortTracks.push_back(output);
      mITSShortPVRefs.push_back(tr.mPVRef);
      if (mMCTruthON) {
        mITSShortLabels.push_back(best);
      }
    }
  };
  auto free = [&](const ClusterCandidate& tr) {
    for (int ref : tr.mClusterRefs) {
      if (ref >= 0 && used[mInputITSclusters.flatIndex(ref)]) {
        return false;
      }
    }
    return true;
  };
  std::sort(proposals.begin(), proposals.end(), [](const auto& a, const auto& b) {
    if (a.track.getNClusters() != b.track.getNClusters()) {
      return a.track.getNClusters() > b.track.getNClusters();
    }
    if (holes(a.track) != holes(b.track)) {
      return holes(a.track) < holes(b.track);
    }
    const float qa = a.track.mHitChi2 / a.track.getNClusters(), qb = b.track.mHitChi2 / b.track.getNClusters();
    if (qa != qb) {
      return qa < qb;
    }
    return std::tie(a.key, a.track.mPVRef, a.track.mClusterRefs, a.track.source.mPartType, a.track.source.mDecayRef) < std::tie(b.key, b.track.mPVRef, b.track.mClusterRefs, b.track.source.mPartType, b.track.source.mDecayRef);
  });
  std::set<std::array<uint32_t, 4>> selected;
  for (const auto& proposal : proposals) {
    if (!selected.count(proposal.key) && free(proposal.track)) {
      commit(proposal.track, false);
      selected.insert(proposal.key);
    }
  }
  if (!cfg.enableShortIB || mStrangenessTrackingStats.tripletsDeferredZeroField || vertices.empty()) {
    return;
  }

  if (cfg.decayFollowingOnly) {
    // The decay pass starts with every cluster available. Short tracking
    // still excludes all upstream ITS and afterburner hits.
    markUpstreamHits();
  }

  // Run the actual ITS CPU tracklet and cell finders on the inner barrel.
  // Retain the async tracking passes; the vertex-seeding passes build PVs.
  using namespace o2::its;
  auto parameters = TrackingMode::getTrackingParameters(TrackingMode::Async);
  std::erase_if(parameters, [](const TrackingParameters& p) { return p.PassFlags[IterationStep::SeedingVertexPass]; });
  if (parameters.empty()) {
    return;
  }
  for (auto& p : parameters) {
    p.NLayers = 3;
    p.MinTrackLength = 3;
    p.MaxHoles = 0;
    p.SeedingLayers = 0;
    p.InactiveLayerMask = 0;
    p.UseDiamond = false; // output requires an association to an input PV
  }

  const auto setupStarted = std::chrono::steady_clock::now();
  auto pool = std::make_shared<BoundedMemoryResource>();
  TimeFrame<7> frame;
  frame.setMemoryPool(pool);
  frame.setIsStaggered(nSlots > 1);
  TimeFrame<7>::ROFOverlapTableN rofTable;
  TimeFrame<7>::ROFVertexLookupTableN vertexTable;
  for (int layer = 0; layer < 7; ++layer) {
    rofTable.defineLayer(layer, timing[layer]);
    vertexTable.defineLayer(layer, timing[layer]);
  }
  rofTable.init();
  frame.setROFOverlapTable(std::move(rofTable));
  TimeFrame<7>::ROFMaskTableN mask{frame.getROFOverlapTable()};
  mask.resetMask(1);
  frame.setMultiplicityCutMask(std::move(mask));
  for (int slot = 0; slot < std::min(nSlots, 3); ++slot) {
    auto pattern = data.getITSClustersPatterns(slot).begin();
    frame.loadROFrameData(data.getITSClustersROFRecords(slot), data.getITSClusters(slot), pattern, mITSDict, nSlots == 1 ? -1 : slot);
  }
  if (nSlots > 1) {
    for (int layer = 3; layer < 7; ++layer) {
      frame.resetROFrameData(layer);
    }
  }

  // ITS vertex lookup requires time ordering. Keep a reverse mapping to the
  // global PV IDs, since the sidecar refers to the original input collection.
  std::vector<int> vertexIds;
  vertexIds.reserve(vertices.size());
  const double endBC = double(o2::base::GRPGeomHelper::getNHBFPerTF()) * o2::constants::lhc::LHCMaxBunches;
  auto vertexStamp = [endBC](const auto& window) -> TimeEstBC {
    const auto start = TimeStampType(std::clamp(std::floor(window.first), 0., endBC));
    const auto end = TimeStampType(std::clamp(std::ceil(window.second), 0., endBC));
    return {start, end - start};
  };
  for (int id = 0; id < int(vertices.size()); ++id) {
    if (vertices[id].getNContributors() > 0 && vertexStamp(pvWindows[id]).getTimeStampError() > 0) {
      vertexIds.push_back(id);
    }
  }
  if (vertexIds.empty()) {
    return;
  }
  std::sort(vertexIds.begin(), vertexIds.end(), [&](int a, int b) {
    const auto left = vertexStamp(pvWindows[a]), right = vertexStamp(pvWindows[b]);
    return std::tuple{left.lower(), left.upper(), a} < std::tuple{right.lower(), right.upper(), b};
  });
  for (int id : vertexIds) {
    const auto& pv = vertices[id];
    Vertex vertex{{pv.getX(), pv.getY(), pv.getZ()}, pv.getCov(), pv.getNContributors(), pv.getChi2()};
    vertex.setTimeStamp(vertexStamp(pvWindows[id]));
    vertex.setFlags(pv.getFlags());
    frame.addPrimaryVertex(vertex);
  }
  vertexTable.init(frame.getPrimaryVertices().data(), frame.getPrimaryVertices().size());
  frame.setROFVertexLookupTable(std::move(vertexTable));

  TrackerTraits<7> tracker;
  std::shared_ptr<tbb::task_arena> arena;
  tracker.setNThreads(mNThreads, arena);
  tracker.adoptTimeFrame(&frame);
  tracker.setMemoryPool(pool);
  tracker.setBz(mBz);
  tracker.updateTrackingParameters(parameters);
  frame.initTrackerTopologies(parameters, 3);

  std::array<const TrackingFrameInfo*, 7> measurements{};
  std::array<const Cluster*, 7> clusters{};
  auto externalRef = [&](int layer, int id) {
    return o2::itsmft::composeClusID(nSlots == 1 ? 0 : layer, frame.getClusterExternalIndex(layer, id));
  };
  const auto* propagator = o2::base::Propagator::Instance();
  const int nVertices = parameters.front().PerPrimaryVertexProcessing ? frame.getROFVertexLookupTableView().getMaxVerticesPerROF() : -1;
  LOGP(info, "ITS short native cell setup: {} vertices, {} passes in {} s", vertexIds.size(), parameters.size(),
       std::chrono::duration<double>(std::chrono::steady_clock::now() - setupStarted).count());

  for (int iteration = 0; iteration < int(parameters.size()); ++iteration) {
    const auto& par = parameters[iteration];
    pool->setMaxMemory(par.MaxMemory);
    const auto started = std::chrono::steady_clock::now();
    tracker.initialiseTimeFrame(iteration);
    if (iteration == 0) {
      for (int layer = 0; layer < 3; ++layer) {
        measurements[layer] = frame.getTrackingFrameInfoOnLayer(layer).data();
        clusters[layer] = frame.getUnsortedClusters()[layer].data();
        for (int id = 0; id < int(frame.getUnsortedClusters()[layer].size()); ++id) {
          if (used[mInputITSclusters.flatIndex(externalRef(layer, id))]) {
            frame.markUsedCluster(layer, id);
          }
        }
      }
    }

    const o2::its::track::TrackFitContext<7> fitContext{
      measurements.data(), par.LayerxX0.data(), par.NLayers, mBz, par.MaxChi2ClusterAttachment, par.MaxChi2NDF,
      propagator, par.CorrType, par.ShiftRefToCluster, par.RepeatRefitOut};
    const o2::its::track::TrackSeedSelector<7> selectSeed{o2::its::constants::MaxTrackSeedQ2Pt, par.MaxChi2NDF, 1, 0, 3, {}, {}};
    // The standard refit has no length-specific setting for three hits.
    const std::array<float, 7> minPt{};
    const auto outputsBefore = mITSShortTracks.size();
    const auto fitsBefore = mStrangenessTrackingStats.nTripletProposals;
    size_t nTracklets = 0, nCells = 0;
    int vertexIndex = std::min(nVertices, 0);
    do {
      tracker.computeLayerTracklets(iteration, vertexIndex);
      nTracklets += frame.getNumberOfTracklets();
      tracker.computeLayerCells(iteration);
      nCells += frame.getNumberOfCells();

      struct Candidate {
        ClusterCandidate output;
        std::array<int, 3> ids{};
      };
      std::vector<Candidate> candidates;
      for (const auto& layerCells : frame.getCells()) {
        for (const auto& cell : layerCells) {
          const TrackSeed<7> seed{cell};
          if (!selectSeed(seed)) {
            continue;
          }
          TrackITSInternal<7> fitted;
          ++mStrangenessTrackingStats.nTripletProposals;
          if (!o2::its::track::refitTrackSeed(seed, fitted, fitContext, clusters.data(), par.LayerRadii.data(), minPt.data(), par.ReseedIfShorter)) {
            continue;
          }

          Candidate candidate;
          auto& out = candidate.output;
          out.mTrack = fitted.paramIn;
          out.mOuter = fitted.paramOut;
          out.mHitChi2 = fitted.getChi2();
          for (int layer = 0; layer < 3; ++layer) {
            candidate.ids[layer] = cell.getCluster(layer);
          }

          // The cell stores its tracklet time overlap, but no PV identity.
          // Associate the fitted track to a timed input PV using standard
          // DCA propagation on a copy for the existing PV sidecar.
          const auto timedVertices = frame.getPrimaryVertices(0, frame.getClusterROF(0, candidate.ids[0]));
          float bestScore = std::numeric_limits<float>::max();
          const auto ts = cell.getTimeStamp();
          for (const auto& vertex : timedVertices) {
            if (vertex.isFlagSet(Vertex::Flags::UPCMode) != par.PassFlags[IterationStep::SelectUPCVertices] ||
                !ts.isCompatible(vertex.getTimeStamp())) {
              continue;
            }
            const int originalId = vertexIds[&vertex - frame.getPrimaryVertices().data()];
            const auto [pvLo, pvHi] = pvTime(originalId);
            const double lo = std::max(double(ts.lower()), pvLo), hi = std::min(double(ts.upper()), pvHi);
            if (!(hi > lo)) {
              continue;
            }
            auto pointing = fitted.paramIn;
            o2::dataformats::DCA dca;
            if (!propagator->propagateToDCA(vertex, pointing, mBz, o2::base::Propagator::MAX_STEP, par.CorrType, &dca)) {
              continue;
            }
            const float yy = dca.getSigmaY2(), yz = dca.getSigmaYZ(), zz = dca.getSigmaZ2();
            const float determinant = yy * zz - yz * yz;
            if (!(yy > 0.f && determinant > 0.f)) {
              continue;
            }
            const float dy = dca.getY(), dz = dca.getZ();
            const float score = (zz * dy * dy - 2.f * yz * dy * dz + yy * dz * dz) / determinant;
            if (score < bestScore || (score == bestScore && originalId < out.mPVRef)) {
              bestScore = score;
              out.mPVRef = originalId;
              out.mTimeMinBC = lo;
              out.mTimeMaxBC = hi;
            }
          }
          if (out.mPVRef < 0) {
            continue;
          }
          for (int layer = 0; layer < 3; ++layer) {
            attach(out, mInputITSclusters.flatIndex(externalRef(layer, candidate.ids[layer])));
          }
          candidates.push_back(candidate);
        }
      }
      std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return std::tie(a.output.mHitChi2, a.output.mClusterRefs, a.output.mPVRef) <
               std::tie(b.output.mHitChi2, b.output.mClusterRefs, b.output.mPVRef);
      });
      for (const auto& candidate : candidates) {
        if (free(candidate.output)) {
          commit(candidate.output, true);
          for (int layer = 0; layer < 3; ++layer) {
            frame.markUsedCluster(layer, candidate.ids[layer]);
          }
        }
      }
    } while (++vertexIndex < nVertices);
    LOGP(info, "ITS short native cells: pass {}, tracklet min pT {} GeV/c, {} tracklets, {} cells, {} refits, accepted {} tracks, output range [{},{}) in {} s",
         iteration, par.TrackletMinPt, nTracklets, nCells, mStrangenessTrackingStats.nTripletProposals - fitsBefore,
         mITSShortTracks.size() - outputsBefore, outputsBefore, mITSShortTracks.size(),
         std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  }
}
} // namespace o2::strangeness_tracking
