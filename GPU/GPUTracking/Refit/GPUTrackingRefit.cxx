// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTrackingRefit.cxx
/// \author David Rohr

#define GPUCA_CADEBUG 0

#include "GPUTrackingRefit.h"
#include "GPUO2DataTypes.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPropagator.h"
#include "GPUConstantMem.h"
#include "TPCFastTransform.h"
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GPUParam.inc"
#include "GPUCommonArray.h"
#include "GPUParam.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::track;
using namespace o2::base;
using namespace o2::tpc;

static constexpr int kIGNORE_ENDS = 3;

#define IgnoreErrors(SNP)                                                                                            \
  if (mIgnoreErrorsOnTrackEnds) {                                                                                    \
    if (abs(i - stop) <= kIGNORE_ENDS && (CAMath::Abs(SNP) >= Propagator::MAX_SIN_PHI || abs(start - stop) < 30)) {  \
      break;                                                                                                         \
    }                                                                                                                \
    if (abs(i - start) <= kIGNORE_ENDS && (CAMath::Abs(SNP) >= Propagator::MAX_SIN_PHI || abs(start - stop) < 30)) { \
      continue;                                                                                                      \
    }                                                                                                                \
  }
// End IgnoreErrors

#ifndef GPUCA_GPUCODE
void GPUTrackingRefitProcessor::InitializeProcessor() {}

void GPUTrackingRefitProcessor::RegisterMemoryAllocation()
{
  AllocateAndInitializeLate();
}

void GPUTrackingRefitProcessor::SetMaxData(const GPUTrackingInOutPointers& io)
{
}
#endif

namespace
{
template <class T>
struct refitTrackTypes;
template <>
struct refitTrackTypes<GPUTPCGMTrackParam> {
  using propagator = GPUTPCGMPropagator;
};
template <>
struct refitTrackTypes<TrackParCov> {
  using propagator = const Propagator*;
};
} // anonymous namespace

template <>
GPUd() void GPUTrackingRefit::initProp<GPUTPCGMPropagator>(GPUTPCGMPropagator& prop)
{
  prop.SetMaterialTPC();
  prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  prop.SetToyMCEventsFlag(false);
  prop.SetSeedingErrors(false);
  prop.SetFitInProjections(mPparam->rec.fitInProjections != 0);
  prop.SetPropagateBzOnly(false);
  prop.SetPolynomialField(&mPparam->polynomialField);
  prop.SetMatLUT(mPmatLUT);
}

template <>
GPUd() void GPUTrackingRefit::initProp<const Propagator*>(const Propagator*& prop)
{
  prop = mPpropagator;
}

template <class T, class S, class U>
GPUd() void GPUTrackingRefit::convertTrack(T& trk, const S& trkX, U& prop, float* chi2)
{
  trk = trkX;
}
GPUd() static void convertTrackParam(GPUTPCGMTrackParam& trk, const TrackParCov& trkX)
{
  for (int i = 0; i < 5; i++) {
    trk.Par()[i] = trkX.getParams()[i];
  }
  for (int i = 0; i < 15; i++) {
    trk.Cov()[i] = trkX.getCov()[i];
  }
  trk.X() = trkX.getX();
}
GPUd() static void convertTrackParam(TrackParCov& trk, const GPUTPCGMTrackParam& trkX)
{
  for (int i = 0; i < 5; i++) {
    trk.setParam(trkX.GetPar()[i], i);
  }
  for (int i = 0; i < 15; i++) {
    trk.setCov(trkX.GetCov()[i], i);
  }
  trk.setX(trkX.GetX());
}
// Generic
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMTrackParam, TrackParCov, GPUTPCGMPropagator>(GPUTPCGMTrackParam& trk, const TrackParCov& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  convertTrackParam(trk, trkX);
  prop.SetTrack(&trk, trkX.getAlpha());
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackParCov, GPUTPCGMTrackParam, GPUTPCGMPropagator>(TrackParCov& trk, const GPUTPCGMTrackParam& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  convertTrackParam(trk, trkX);
  trk.setAlpha(prop.GetAlpha());
}
// GPUTPCGMMergedTrack input
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackParCov, GPUTPCGMMergedTrack, const Propagator*>(TrackParCov& trk, const GPUTPCGMMergedTrack& trkX, const Propagator*& prop, float* chi2)
{
  initProp(prop);
  convertTrackParam(trk, trkX.GetParam());
  trk.setAlpha(trkX.GetAlpha());
  *chi2 = trkX.GetParam().GetChi2();
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMMergedTrack, TrackParCov, const Propagator*>(GPUTPCGMMergedTrack& trk, const TrackParCov& trkX, const Propagator*& prop, float* chi2)
{
  convertTrackParam(trk.Param(), trkX);
  trk.SetAlpha(trkX.getAlpha());
  trk.Param().SetChi2(*chi2);
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMTrackParam, GPUTPCGMMergedTrack, GPUTPCGMPropagator>(GPUTPCGMTrackParam& trk, const GPUTPCGMMergedTrack& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  initProp(prop);
  trk = trkX.GetParam();
  prop.SetTrack(&trk, trkX.GetAlpha());
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMMergedTrack, GPUTPCGMTrackParam, GPUTPCGMPropagator>(GPUTPCGMMergedTrack& trk, const GPUTPCGMTrackParam& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  trk.SetParam(trkX);
  trk.SetAlpha(prop.GetAlpha());
}
// TrackTPC input
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackParCov, TrackTPC, const Propagator*>(TrackParCov& trk, const TrackTPC& trkX, const Propagator*& prop, float* chi2)
{
  initProp(prop);
  convertTrack<TrackParCov, TrackParCov, const Propagator*>(trk, trkX, prop, nullptr);
  *chi2 = trkX.getChi2();
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackTPC, TrackParCov, const Propagator*>(TrackTPC& trk, const TrackParCov& trkX, const Propagator*& prop, float* chi2)
{
  convertTrack<TrackParCov, TrackParCov, const Propagator*>(trk, trkX, prop, nullptr);
  trk.setChi2(*chi2);
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMTrackParam, TrackTPC, GPUTPCGMPropagator>(GPUTPCGMTrackParam& trk, const TrackTPC& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  initProp(prop);
  convertTrack<GPUTPCGMTrackParam, TrackParCov, GPUTPCGMPropagator>(trk, trkX, prop, nullptr);
  trk.SetChi2(trkX.getChi2());
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackTPC, GPUTPCGMTrackParam, GPUTPCGMPropagator>(TrackTPC& trk, const GPUTPCGMTrackParam& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  convertTrack<TrackParCov, GPUTPCGMTrackParam, GPUTPCGMPropagator>(trk, trkX, prop, nullptr);
  trk.setChi2(trkX.GetChi2());
}
// TrackParCovWithArgs input
template <>
GPUd() void GPUTrackingRefit::convertTrack<TrackParCov, GPUTrackingRefit::TrackParCovWithArgs, const Propagator*>(TrackParCov& trk, const GPUTrackingRefit::TrackParCovWithArgs& trkX, const Propagator*& prop, float* chi2)
{
  initProp(prop);
  convertTrack<TrackParCov, TrackParCov, const Propagator*>(trk, trkX.trk, prop, nullptr);
  *chi2 = trkX.chi2 ? *trkX.chi2 : 0.f;
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTrackingRefit::TrackParCovWithArgs, TrackParCov, const Propagator*>(GPUTrackingRefit::TrackParCovWithArgs& trk, const TrackParCov& trkX, const Propagator*& prop, float* chi2)
{
  convertTrack<TrackParCov, TrackParCov, const Propagator*>(trk.trk, trkX, prop, nullptr);
  if (trk.chi2) {
    *trk.chi2 = *chi2;
  }
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTPCGMTrackParam, GPUTrackingRefit::TrackParCovWithArgs, GPUTPCGMPropagator>(GPUTPCGMTrackParam& trk, const GPUTrackingRefit::TrackParCovWithArgs& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  initProp(prop);
  convertTrack<GPUTPCGMTrackParam, TrackParCov, GPUTPCGMPropagator>(trk, trkX.trk, prop, nullptr);
  trk.SetChi2(trkX.chi2 ? *trkX.chi2 : 0.f);
}
template <>
GPUd() void GPUTrackingRefit::convertTrack<GPUTrackingRefit::TrackParCovWithArgs, GPUTPCGMTrackParam, GPUTPCGMPropagator>(GPUTrackingRefit::TrackParCovWithArgs& trk, const GPUTPCGMTrackParam& trkX, GPUTPCGMPropagator& prop, float* chi2)
{
  convertTrack<TrackParCov, GPUTPCGMTrackParam, GPUTPCGMPropagator>(trk.trk, trkX, prop, chi2);
  if (trk.chi2) {
    *trk.chi2 = trkX.GetChi2();
  }
}

GPUd() static const float* getPar(const GPUTPCGMTrackParam& trk) { return trk.GetPar(); }
GPUd() static const float* getPar(const TrackParCov& trk) { return trk.getParams(); }

template <class T, class S>
GPUd() int GPUTrackingRefit::RefitTrack(T& trkX, bool outward, bool resetCov)
{
#ifndef __OPENCL__
  CADEBUG(int ii; printf("\nRefitting track\n"));
  typename refitTrackTypes<S>::propagator prop;
  S trk;
  float TrackParCovChi2 = 0.f;
  convertTrack(trk, trkX, prop, &TrackParCovChi2);
  int begin = 0, count;
  float tOffset;
  if constexpr (std::is_same<T, GPUTPCGMMergedTrack>::value) {
    count = trkX.NClusters();
    if (trkX.Looper()) {
      int leg = mPtrackHits[trkX.FirstClusterRef() + trkX.NClusters() - 1].leg;
      for (int i = trkX.NClusters() - 2; i > 0; i--) {
        if (mPtrackHits[trkX.FirstClusterRef() + i].leg != leg) {
          begin = i + 1;
          break;
        }
      }
    }
    tOffset = trkX.GetParam().GetTZOffset();
  } else if constexpr (std::is_same<T, TrackTPC>::value) {
    count = trkX.getNClusters();
    tOffset = trkX.getTime0();
  } else if constexpr (std::is_same<T, TrackParCovWithArgs>::value) {
    count = trkX.clusRef.getEntries();
    tOffset = trkX.time0;
  } else {
    static_assert("Invalid template");
  }
  int direction = outward ? -1 : 1;
  int start = outward ? count - 1 : begin;
  int stop = outward ? begin - 1 : count;
  const ClusterNative* cl = nullptr;
  uint8_t sector, row = 0, currentSector = 0, currentRow = 0;
  short clusterState = 0, nextState;
  int nFitted = 0;
  for (int i = start; i != stop; i += cl ? 0 : direction) {
    float x, y, z, charge = 0.f;
    int clusters = 0;
    while (true) {
      if (!cl) {
        CADEBUG(ii = i);
        if constexpr (std::is_same<T, GPUTPCGMMergedTrack>::value) {
          const auto& hit = mPtrackHits[trkX.FirstClusterRef() + i];
          cl = &mPclusterNative->clustersLinear[hit.num];
          if (hit.state & (GPUTPCGMMergedTrackHit::flagReject | GPUTPCGMMergedTrackHit::flagNotFit)) {
            cl = nullptr;
            if (i + direction != stop) {
              i += direction;
              continue;
            }
            break;
          }
          row = hit.row;
          sector = hit.slice;
          nextState = mPclusterState[hit.num];
        } else if constexpr (std::is_same<T, TrackTPC>::value) {
          cl = &trkX.getCluster(mPtrackHitReferences, i, *mPclusterNative, sector, row);
          nextState = mPclusterState[cl - mPclusterNative->clustersLinear];
        } else if constexpr (std::is_same<T, TrackParCovWithArgs>::value) {
          cl = &TrackTPC::getCluster(mPtrackHitReferences, i, *mPclusterNative, sector, row, trkX.clusRef);
          nextState = mPclusterState[cl - mPclusterNative->clustersLinear];
        } else {
          static_assert("Invalid template");
        }
      }
      if (clusters == 0 || (row == currentRow && sector == currentSector)) {
        if (clusters == 1) {
          x *= charge;
          y *= charge;
          z *= charge;
        }
        if (clusters == 0) {
          mPfastTransform->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, tOffset);
          CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f\n", ii, count, row, mPparam->Alpha(sector), (int)sector, x, y, z));
          currentRow = row;
          currentSector = sector;
          charge = cl->qTot;
          clusterState = nextState;
        } else {
          float xx, yy, zz;
          mPfastTransform->Transform(sector, row, cl->getPad(), cl->getTime(), xx, yy, zz, tOffset);
          CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f\n", ii, count, row, mPparam->Alpha(sector), (int)sector, xx, yy, zz));
          x += xx * cl->qTot;
          y += yy * cl->qTot;
          z += zz * cl->qTot;
          charge += cl->qTot;
          clusterState |= nextState;
        }
        cl = nullptr;
        clusters++;
        if (i + direction != stop) {
          i += direction;
          continue;
        }
      }
      break;
    }
    if (clusters == 0) {
      continue;
    } else if (clusters > 1) {
      x /= charge;
      y /= charge;
      z /= charge;
      CADEBUG(printf("\tMerged Hit  Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f\n", row, mPparam->Alpha(sector), (int)sector, x, y, z));
    }

    if constexpr (std::is_same<S, GPUTPCGMTrackParam>::value) {
      if (prop.PropagateToXAlpha(x, mPparam->Alpha(currentSector), !outward)) {
        IgnoreErrors(trk.GetSinPhi());
        return -2;
      }
      if (resetCov) {
        trk.ResetCovariance();
      }
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), x, trk.Par()[0], trk.Par()[1], trk.Par()[4], prop.GetQPt0(), trk.Par()[2], prop.GetSinPhi0(), trk.Par()[0] - y, trk.Par()[1] - z, sqrtf(trk.Cov()[0]), sqrtf(trk.Cov()[2]), sqrtf(trk.Cov()[5]), sqrtf(trk.Cov()[14]), trk.Cov()[10]));
      if (prop.Update(y, z, row, *mPparam, clusterState, 0, nullptr, true)) {
        IgnoreErrors(trk.GetSinPhi());
        return -3;
      }
      CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), x, trk.Par()[0], trk.Par()[1], trk.Par()[4], prop.GetQPt0(), trk.Par()[2], prop.GetSinPhi0(), trk.Par()[3], "", sqrtf(trk.Cov()[0]), sqrtf(trk.Cov()[2]), sqrtf(trk.Cov()[5]), sqrtf(trk.Cov()[14]), trk.Cov()[10]));
    } else if constexpr (std::is_same<S, TrackParCov>::value) {
      if (!trk.rotate(mPparam->Alpha(currentSector))) {
        IgnoreErrors(trk.getSnp());
        return -1;
      }
      if (!prop->PropagateToXBxByBz(trk, x, GPUCA_MAX_SIN_PHI_LOW)) {
        IgnoreErrors(trk.getSnp());
        return -2;
      }
      if (resetCov) {
        trk.resetCovariance();
        TrackParCovChi2 = 0.f;
      }
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", trk.getAlpha(), x, trk.getParams()[0], trk.getParams()[1], trk.getParams()[4], trk.getParams()[4], trk.getParams()[2], trk.getParams()[2], trk.getParams()[0] - y, trk.getParams()[1] - z, sqrtf(trk.getCov()[0]), sqrtf(trk.getCov()[2]), sqrtf(trk.getCov()[5]), sqrtf(trk.getCov()[14]), trk.getCov()[10]));
      gpu::gpustd::array<float, 2> p = {y, z};
      gpu::gpustd::array<float, 3> c = {0, 0, 0};
      mPparam->GetClusterErrors2(currentRow, z, getPar(trk)[2], getPar(trk)[3], c[0], c[2]);
      mPparam->UpdateClusterError2ByState(clusterState, c[0], c[2]);
      TrackParCovChi2 += trk.getPredictedChi2(p, c);
      if (!trk.update(p, c)) {
        IgnoreErrors(trk.getSnp());
        return -3;
      }
      CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", trk.getAlpha(), x, trk.getParams()[0], trk.getParams()[1], trk.getParams()[4], trk.getParams()[4], trk.getParams()[2], trk.getParams()[2], trk.getParams()[3], "", sqrtf(trk.getCov()[0]), sqrtf(trk.getCov()[2]), sqrtf(trk.getCov()[5]), sqrtf(trk.getCov()[14]), trk.getCov()[10]));
    } else {
      static_assert("Invalid template");
    }
    resetCov = false;
    nFitted++;
  }
  if constexpr (std::is_same<S, GPUTPCGMTrackParam>::value) {
    float alpha = prop.GetAlpha();
    trk.MoveToReference(prop, *mPparam, alpha);
    trk.NormalizeAlpha(alpha);
    prop.SetAlpha(alpha);
  } else if constexpr (std::is_same<S, TrackParCov>::value) {
    static constexpr float kDeg2Rad = M_PI / 180.f;
    static constexpr float kSectAngle = 2 * M_PI / 18.f;
    if (mPparam->rec.TrackReferenceX <= 500) {
      if (prop->PropagateToXBxByBz(trk, mPparam->rec.TrackReferenceX)) {
        if (CAMath::Abs(trk.getY()) > trk.getX() * CAMath::Tan(kSectAngle / 2.f)) {
          float newAlpha = trk.getAlpha() + floor(CAMath::ATan2(trk.getY(), trk.getX()) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
          GPUTPCGMTrackParam::NormalizeAlpha(newAlpha);
          trk.rotate(newAlpha) && prop->PropagateToXBxByBz(trk, mPparam->rec.TrackReferenceX);
        }
      }
    }
  } else {
    static_assert("Invalid template");
  }

  convertTrack(trkX, trk, prop, &TrackParCovChi2);
  return nFitted;
#else
  return 0; // TODO: Fixme, implement std::isSame for opencl
#endif
}

template GPUd() int GPUTrackingRefit::RefitTrack<GPUTPCGMMergedTrack, TrackParCov>(GPUTPCGMMergedTrack& trk, bool outward, bool resetCov);
template GPUd() int GPUTrackingRefit::RefitTrack<GPUTPCGMMergedTrack, GPUTPCGMTrackParam>(GPUTPCGMMergedTrack& trk, bool outward, bool resetCov);
template GPUd() int GPUTrackingRefit::RefitTrack<TrackTPC, TrackParCov>(TrackTPC& trk, bool outward, bool resetCov);
template GPUd() int GPUTrackingRefit::RefitTrack<TrackTPC, GPUTPCGMTrackParam>(TrackTPC& trk, bool outward, bool resetCov);
template GPUd() int GPUTrackingRefit::RefitTrack<GPUTrackingRefit::TrackParCovWithArgs, TrackParCov>(GPUTrackingRefit::TrackParCovWithArgs& trk, bool outward, bool resetCov);
template GPUd() int GPUTrackingRefit::RefitTrack<GPUTrackingRefit::TrackParCovWithArgs, GPUTPCGMTrackParam>(GPUTrackingRefit::TrackParCovWithArgs& trk, bool outward, bool resetCov);

#ifndef GPUCA_GPUCODE
void GPUTrackingRefit::SetPtrsFromGPUConstantMem(const GPUConstantMem* v, MEM_CONSTANT(GPUParam) * p)
{
  mPclusterState = v->ioPtrs.mergedTrackHitStates;
  mPclusterNative = v->ioPtrs.clustersNative;
  mPtrackHits = v->ioPtrs.mergedTrackHits;
  mPfastTransform = v->calibObjects.fastTransform;
  mPmatLUT = v->calibObjects.matLUT;
  mPparam = p ? p : &v->param;
}

void GPUTrackingRefit::SetPropagatorDefault()
{
  mPpropagator = mPparam->GetDefaultO2Propagator(false);
}
#endif
