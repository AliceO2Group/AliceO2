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

/// \file GPUTPCGMTrackParam.cxx
/// \author David Rohr, Sergey Gorbunov

#define GPUCA_CADEBUG 0
#define DEBUG_SINGLE_TRACK -1

#include "GPUTPCDef.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCTracker.h"
#include "GPUdEdx.h"
#include "GPUParam.h"
#include "GPUO2DataTypes.h"
#include "GPUConstantMem.h"
#include "TPCFastTransformPOD.h"
#include "GPUTPCConvertImpl.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUParam.inc"
#include "GPUGetConstexpr.h"

#ifdef GPUCA_CADEBUG_ENABLED
#include "GPUSettings.h"
#include "AliHLTTPCClusterMCData.h"
#endif

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#include <cstdlib>
#endif

using namespace o2::gpu;
using namespace o2::tpc;

GPUd() bool GPUTPCGMTrackParam::Fit(GPUTPCGMMerger& GPUrestrict() merger, int32_t iTrk, int32_t& GPUrestrict() N, int32_t& GPUrestrict() NTolerated, float& GPUrestrict() Alpha, GPUTPCGMMergedTrack& GPUrestrict() track, bool rebuilt)
{
  static constexpr float maxSinPhi = constants::MAX_SIN_PHI;

  const GPUParam& GPUrestrict() param = merger.Param();
  GPUTPCGMMergedTrackHit* GPUrestrict() clusters = merger.Clusters() + track.FirstClusterRef();

  GPUdEdx dEdx, dEdxAlt;
  GPUTPCGMPropagator prop;
  gputpcgmmergertypes::InterpolationErrors interpolation;
  prop.SetMaterialTPC();
  prop.SetPolynomialField(&param.polynomialField);
  prop.SetMaxSinPhi(maxSinPhi);
  if (param.rec.tpc.mergerInterpolateErrors && !rebuilt) {
    for (uint32_t i = 0; i < interpolation.size; i++) { // TODO: Tune the zeroing size
      interpolation.hit[i].errorY = -1;
    }
  }

  const int32_t nWays = param.rec.tpc.nWays;
  const int32_t maxN = N;
  int32_t ihitStart = 0;
  int32_t interpolatedStart = 0;
  float covYYUpd = 0.f;
  float deltaZ = 0.f;

  for (int32_t iWay = rebuilt ? nWays - 1 : 0; iWay < nWays; iWay++) { // TODO DR: Unrolling has no performance improvement on GPU, why?
    int32_t nMissed = 0, nMissed2 = 0;
    float sumInvSqrtCharge = 0.f; // TODO: Compute in first iteration and store!
    int32_t nAvgCharge = 0;

    if (iWay && (iWay & 1) == 0) {
      StoreOuter(&track.OuterParam(), prop.GetAlpha());
    }

    int32_t resetT0 = initResetT0();
    const bool refit = (nWays == 1 || iWay >= 1);
    const bool finalOutInFit = iWay + 2 >= nWays;
    const bool finalFit = iWay == nWays - 1;

    ResetCovariance();
    prop.SetSeedingErrors(!(refit));
    prop.SetFitInProjections(true); // param.rec.fitInProjections == -1 ? (iWay == 0) : param.rec.fitInProjections); // TODO: Reenable once fixed
    prop.SetPropagateBzOnly(param.rec.fitPropagateBzOnly == -1 ? !finalFit : param.rec.fitPropagateBzOnly);
    prop.SetMatLUT((param.rec.useMatLUT && finalFit) ? merger.GetConstantMem()->calibObjects.matLUT : nullptr);
    prop.SetTrack(this, iWay && !rebuilt ? prop.GetAlpha() : Alpha);
    ConstrainSinPhi(iWay == 0 ? 0.95f : constants::MAX_SIN_PHI_LOW);
    CADEBUG(printf("Fitting track %d way %d (sector %d, alpha %f) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", iTrk, iWay, CAMath::Float2IntRn(prop.GetAlpha() / GPUTPCGeometry::kSectAngle()) + (mP[1] < 0 ? 18 : 0), prop.GetAlpha()));

    N = 0;
    uint8_t lastUpdateRow = 255, lastPropagateRow = 255, lastSector = 255;
    float lastUpdateX = -1;
    const bool inFlyDirection = iWay & 1;
    const int32_t wayDirection = (iWay & 1) ? -1 : 1;

    for (int32_t ihit = ihitStart, interpolationIndex = interpolatedStart - wayDirection; ihit >= 0 && ihit < maxN; ihit += wayDirection) {
      if (!param.rec.tpc.rebuildTrackInFit || rebuilt) {
        if ((param.rec.tpc.trackFitRejectMode > 0 && nMissed >= param.rec.tpc.trackFitRejectMode) || nMissed2 >= param.rec.tpc.trackFitMaxRowMissedHard || (clusters[ihit].state & GPUTPCGMMergedTrackHit::flagReject) || (rebuilt && (clusters[ihit].state & GPUTPCGMMergedTrackHit::flagHighIncl))) {
          CADEBUG(printf("\tSkipping hit %d, %d hits rejected, flag %X\n", ihit, nMissed, (int32_t)clusters[ihit].state));
          if (rebuilt && (clusters[ihit].state & GPUTPCGMMergedTrackHit::flagHighIncl)) {
            NTolerated++;
          }
          if (finalOutInFit && !(clusters[ihit].state & (GPUTPCGMMergedTrackHit::flagReject | GPUTPCGMMergedTrackHit::flagHighIncl))) {
            clusters[ihit].state |= GPUTPCGMMergedTrackHit::flagRejectErr;
          }
          continue;
        }
      }

      const bool allowChangeClusters = finalOutInFit && (nWays == 1 || ((iWay & 1) ? (ihit <= CAMath::Max(maxN / 2, maxN - 30)) : (ihit >= CAMath::Min(maxN / 2, 30))));

      int32_t ihitMergeFirst = ihit;
      interpolationIndex += wayDirection;
      uint8_t clusterState = clusters[ihit].state;
      const float clAlpha = param.Alpha(clusters[ihit].sector);
      float xx, yy, zz;
      const int32_t currentClusterStatus = MergeDoubleRowClusters(ihit, wayDirection, clusters, merger, prop, xx, yy, zz, maxN, clAlpha, clusterState, param.rec.tpc.rebuildTrackInFit ? rebuilt : allowChangeClusters);
      // TODO: Check about tracks who have clusters in the same row multiple times in different sectors

      const auto& cluster = clusters[ihit];
      CADEBUG(printf("\tSector %2d %11sTrack   Alpha %8.3f %s, X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", (int32_t)cluster.sector, "", prop.GetAlpha(), (CAMath::Abs(prop.GetAlpha() - clAlpha) < 0.01 ? "   " : " R!"), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
      CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f (Missed %d)\n", ihit, maxN, (int32_t)clusters[ihit].row, clAlpha, (int32_t)clusters[ihit].sector, xx, yy, zz, nMissed));

      uint8_t dEdxSubThresholdRow = 255;
      bool doInterpolate = param.rec.tpc.rebuildTrackInFit && (iWay == nWays - 3 || iWay == nWays - 2);
      if (lastPropagateRow != 255 && CAMath::Abs(cluster.row - lastPropagateRow) > 1) {
        bool dodEdx = param.dodEdxEnabled && param.rec.tpc.adddEdxSubThresholdClusters && finalFit && CAMath::Abs(cluster.row - lastUpdateRow) == 2 && cluster.sector == lastSector && currentClusterStatus == 0;
        bool doAttach = allowChangeClusters && !param.rec.tpc.rebuildTrackInFit && !(merger.Param().rec.tpc.disableRefitAttachment & 2);
        if (dodEdx || doAttach || doInterpolate) {
          int32_t step = cluster.row > lastPropagateRow ? 1 : -1;
          for (int32_t iRow = lastPropagateRow + step; iRow != cluster.row; iRow += step) {
            float tmpX, tmpY, tmpZ;
            if (prop.GetPropagatedYZ(mX - GPUTPCGeometry::Row2X(iRow - step) + GPUTPCGeometry::Row2X(iRow), tmpY, tmpZ)) {
              break;
            }
            merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoX(cluster.sector, iRow, tmpY, tmpZ, tmpX);
            if (prop.PropagateToXAlpha(tmpX, prop.GetAlpha(), inFlyDirection)) {
              break;
            }
            FitAddRow(iRow, cluster.sector, iTrk, track, prop, inFlyDirection, merger, &dEdxSubThresholdRow, dodEdx, doAttach, doInterpolate);
          }
        }
        interpolationIndex += (CAMath::Abs(cluster.row - lastPropagateRow) - 1) * wayDirection;
      }
      lastPropagateRow = cluster.row;

      int32_t retValProp = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
      if ((retValProp == -2 &&                                                  // Rotation failed, try to bring to new x with old alpha first, rotate, and then propagate to x, alpha
           (prop.PropagateToXAlpha(xx, prop.GetAlpha(), inFlyDirection) != 0 || // Cannot rotate to new alpha at all
            prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection))) ||            // propagation fails nonetheless
          retValProp) {                                                         // failed for other reason but rotation
        CADEBUG(printf(" --- break-prop\n"));
        MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagHighIncl);
        nMissed2++;
        NTolerated++;
        continue;
      }
      // clang-format off
      CADEBUG(if (!CheckCov()){printf("INVALID COV AFTER PROPAGATE!!!\n");});
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   PErr %d\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[0] - yy, mP[1] - zz, sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], retValProp));
      // clang-format on
      if (mNDF >= 0 && (mC[0] > param.rec.tpc.trackFitCovLimit || mC[2] > param.rec.tpc.trackFitCovLimit)) {
        break; // bad chi2 for the whole track, stop the fit
      }

      if ((uint32_t)interpolationIndex >= interpolation.size) {
        merger.raiseError(GPUErrors::ERROR_MERGER_INTERPOLATION_OVERFLOW, interpolationIndex, interpolation.size);
        break;
      }
      auto& inter = interpolation.hit[interpolationIndex];

      float uncorrectedY = -1e6f;
      if (param.rec.tpc.rebuildTrackInFit) {
        if (iWay == nWays - 2) {
          uncorrectedY = FindBestInterpolatedHit(merger, inter, cluster.sector, cluster.row, deltaZ, sumInvSqrtCharge, nAvgCharge, prop, iTrk);
        }
        if (allowChangeClusters) {
          AttachClusters(merger, cluster.sector, cluster.row, iTrk, track.Leg() == 0, prop); // TODO: Do this during FindBestInterpolatedHit
        }
      } else if (allowChangeClusters) {
        uncorrectedY = AttachClusters(merger, cluster.sector, cluster.row, iTrk, track.Leg() == 0, prop);
      } else if (param.rec.tpc.rejectEdgeClustersInTrackFit) {
        float tmpZ;
        merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(cluster.sector, cluster.row, mP[0], mP[1], uncorrectedY, tmpZ);
      }

      HandleCrossCE(param, cluster.sector, lastSector);
      lastSector = cluster.sector;

      if (param.rec.tpc.mergerInterpolateErrors && iWay == nWays - 3) {
        prop.InterpolateFill(&inter);
      }

      if (currentClusterStatus) {
        nMissed++;
        nMissed2++;
        continue;
      }

      int32_t retValHit = FitHit(merger, iTrk, track, xx, yy, zz, clusterState, clAlpha, iWay, inFlyDirection, deltaZ, lastUpdateX, clusters, prop, inter, dEdx, dEdxAlt, sumInvSqrtCharge, nAvgCharge, ihit, ihitMergeFirst, allowChangeClusters, refit, finalFit, nMissed, nMissed2, resetT0, uncorrectedY);
      if (retValHit == 0) {
        DodEdx(dEdx, dEdxAlt, merger, finalFit, ihit, ihitMergeFirst, wayDirection, clusters, clusterState, zz, dEdxSubThresholdRow);
        ihitStart = ihit;
        interpolatedStart = interpolationIndex;
        N++;
        covYYUpd = mC[0];
      } else if (retValHit == 1) {
        break;
      } else if (retValHit == 2) {
        NTolerated++;
        continue;
      }

      lastUpdateRow = cluster.row;
      assert(!param.rec.tpc.mergerInterpolateErrors || rebuilt || iWay != nWays - 2 || ihit || interpolationIndex == 0);
    }
    if (finalOutInFit && !(param.rec.tpc.disableRefitAttachment & 4) && lastUpdateRow != 255) {
      StoreLoopPropagation(merger, lastSector, lastUpdateRow, iTrk, lastUpdateRow > clusters[(iWay & 1) ? (maxN - 1) : 0].row, prop.GetAlpha());
      CADEBUG(printf("\t\tSTORING %d lastUpdateRow %d row %d out %d\n", iTrk, (int)lastUpdateRow, (int)clusters[(iWay & 1) ? (maxN - 1) : 0].row, lastUpdateRow > clusters[(iWay & 1) ? (maxN - 1) : 0].row));
    }
    if (!(iWay & 1) && !finalFit && !track.CCE() && !track.Looper()) {
      deltaZ = ShiftZ(clusters, merger, maxN);
    } else {
      deltaZ = 0.f;
    }

    if (param.rec.tpc.rebuildTrackInFit && iWay == nWays - 2) {
      Alpha = prop.GetAlpha();
      if (ihitStart != 0) {
        MarkClusters(clusters, 0, ihitStart - 1, 1, GPUTPCGMMergedTrackHit::flagHighIncl);
      }
      return true;
    }
  }
  ConstrainSinPhi();

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamUpdateTrack, iTrk)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_accept_track", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("debug_accept_track").data() << "iTrk=" << iTrk << "outerParam=" << track.OuterParam() << "track=" << this << "ihitStart=" << ihitStart << "\n";
  })

  if (!(N + NTolerated >= (int32_t)param.tpcMinHitsB5(mP[4] * param.qptB5Scaler) && 2 * NTolerated <= CAMath::Max(10, N) && CheckNumericalQuality(covYYUpd))) {
    return false; // TODO: NTolerated should never become that large, check what is going wrong!
  }
  if (param.rec.tpc.minNClustersFinalTrack != -1 && N + NTolerated < param.rec.tpc.minNClustersFinalTrack) {
    return false;
  }

  if (param.par.dodEdx && param.dodEdxEnabled) {
    dEdx.computedEdx(merger.MergedTracksdEdx()[iTrk], param);
    if GPUCA_RTC_CONSTEXPR (GPUCA_GET_CONSTEXPR(param.rec.tpc, dEdxClusterRejectionFlagMask) != GPUCA_GET_CONSTEXPR(param.rec.tpc, dEdxClusterRejectionFlagMaskAlt)) {
      dEdxAlt.computedEdx(merger.MergedTracksdEdxAlt()[iTrk], param);
    }
  }
  Alpha = prop.GetAlpha();
  MoveToReference(prop, param, Alpha);
  NormalizeAlpha(Alpha);

  return true;
}

GPUdii() void GPUTPCGMTrackParam::FitAddRow(const int32_t iRow, const uint8_t sector, const int32_t iTrk, const GPUTPCGMMergedTrack& GPUrestrict() track, GPUTPCGMPropagator& GPUrestrict() prop, const bool inFlyDirection, GPUTPCGMMerger& GPUrestrict() merger, uint8_t* GPUrestrict() dEdxSubThresholdRow, const bool dodEdx, const bool doAttach, const bool doInterpolate)
{
  if (CAMath::Abs(mP[2]) > constants::MAX_SIN_PHI || CAMath::Abs(mP[0]) > CAMath::Abs(mX) * CAMath::Tan(GPUTPCGeometry::kSectAngle() / 2.f)) {
    return;
  }
  const GPUParam& GPUrestrict() param = merger.Param();
  if GPUCA_RTC_CONSTEXPR (GPUCA_GET_CONSTEXPR(param.par, dodEdx)) {
    if (dodEdx) {
      float yUncorrected, zUncorrected;
      merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(sector, iRow, mP[0], mP[1], yUncorrected, zUncorrected);
      uint32_t pad = CAMath::Float2UIntRn(GPUTPCGeometry::LinearY2Pad(sector, iRow, yUncorrected));
      if (!(pad >= GPUTPCGeometry::NPads(iRow) || (merger.GetConstantMem()->calibObjects.dEdxCalibContainer && merger.GetConstantMem()->calibObjects.dEdxCalibContainer->isDead(sector, iRow, pad)))) {
        *dEdxSubThresholdRow = iRow;
      }
    }
  }
  if (doAttach) {
    AttachClusters(merger, sector, iRow, iTrk, track.Leg() == 0, prop);
  }
}

GPUdii() void GPUTPCGMTrackParam::HandleCrossCE(const GPUParam& GPUrestrict() param, const uint8_t sector, const uint8_t& lastSector)
{
  const bool crossCE = lastSector != 255 && ((lastSector < 18) ^ (sector < 18));
  if (crossCE) {
    if (param.rec.tpc.addErrorsCECrossing) {
      if (param.rec.tpc.addErrorsCECrossing >= 2) {
        AddCovDiagErrorsWithCorrelations(param.rec.tpc.errorsCECrossing);
      } else {
        AddCovDiagErrors(param.rec.tpc.errorsCECrossing);
      }
    } else if (mC[2] < 0.5f) {
      mC[2] = 0.5f;
    }
  }
}

GPUdii() int32_t GPUTPCGMTrackParam::FitHit(GPUTPCGMMerger& GPUrestrict() merger, const int32_t iTrk, const GPUTPCGMMergedTrack& GPUrestrict() track, const float xx, const float yy, const float zz, const uint8_t clusterState, const float clAlpha, const int32_t iWay, const bool inFlyDirection, float& GPUrestrict() deltaZ, float& GPUrestrict() lastUpdateX, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, GPUTPCGMPropagator& GPUrestrict() prop, gputpcgmmergertypes::InterpolationErrorHit& GPUrestrict() inter, GPUdEdx& GPUrestrict() dEdx, GPUdEdx& GPUrestrict() dEdxAlt, float& GPUrestrict() sumInvSqrtCharge, int32_t& GPUrestrict() nAvgCharge, const int32_t ihit, const int32_t ihitMergeFirst, const bool allowChangeClusters, const bool refit, const bool finalFit, int32_t& GPUrestrict() nMissed, int32_t& GPUrestrict() nMissed2, int32_t& GPUrestrict() resetT0, float uncorrectedY)
{
  const GPUParam& GPUrestrict() param = merger.Param();
  const int32_t nWays = param.rec.tpc.nWays;
  const int32_t wayDirection = (iWay & 1) ? -1 : 1;
  const auto& cluster = clusters[ihit];

  const float maxSinForUpdate = CAMath::Sin(70.f * CAMath::Deg2Rad());
  if (mNDF > 0 && CAMath::Abs(prop.GetSinPhi0()) >= maxSinForUpdate) {
    MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagHighIncl);
    nMissed2++;
    CADEBUG(printf(" --- break-sinphi\n"));
    return 2; // Propagate failed or high incl angle
  }

  int32_t retValUpd = 0, retValInt = 0;
  float threshold = 3.f + (lastUpdateX >= 0 ? (CAMath::Abs(mX - lastUpdateX) / 2) : 0.f);
  if (mNDF > (int32_t)param.rec.tpc.mergerNonInterpolateRejectMinNDF && (CAMath::Abs(yy - mP[0]) > threshold || CAMath::Abs(zz - mP[1]) > threshold)) {
    retValUpd = GPUTPCGMPropagator::updateErrorClusterRejectedDistance;
    if (param.rec.tpc.rebuildTrackInFit) {
      MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectDistance); // Will enable rejectChi2 in further rounds
    }
  } else {
    float err2Y, err2Z;
    const float time = merger.GetConstantMem()->ioPtrs.clustersNative ? merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[cluster.num].getTime() : -1.f;
    const float invSqrtCharge = merger.GetConstantMem()->ioPtrs.clustersNative ? CAMath::InvSqrt(merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[cluster.num].qMax) : 0.f;
    const float invCharge = merger.GetConstantMem()->ioPtrs.clustersNative ? (1.f / merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[cluster.num].qMax) : 0.f;
    float invAvgCharge = (sumInvSqrtCharge += invSqrtCharge) / ++nAvgCharge;
    invAvgCharge *= invAvgCharge;

    prop.GetErr2(err2Y, err2Z, param, zz, cluster.row, clusterState, cluster.sector, time, invAvgCharge, invCharge);

    bool rejectChi2 = (clusterState & GPUTPCGMMergedTrackHit::flagReject);
    if (param.rec.tpc.mergerInterpolateErrors) {
      if (iWay == nWays - 2) {
        if (!param.rec.tpc.rebuildTrackInFit) {
          if (inter.errorY < (GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE_A)0) {
            rejectChi2 = true;
          } else {
            retValInt = prop.InterpolateReject(param, yy, zz, clusterState, &inter, err2Y, err2Z, deltaZ);
          }
        }
      } else if (iWay == nWays - 1) {
        if (param.rec.tpc.mergerInterpolateRejectAlsoOnCurrentPosition && GetNDF() > (int32_t)param.rec.tpc.mergerNonInterpolateRejectMinNDF) {
          rejectChi2 = true;
        }
      }
    } else {
      rejectChi2 = allowChangeClusters;
    }

    if (param.rec.tpc.rejectEdgeClustersInTrackFit && uncorrectedY > -1e6f && param.rejectEdgeClusterByY(uncorrectedY, cluster.row, CAMath::Sqrt(mC[0]))) {
      retValUpd = GPUTPCGMPropagator::updateErrorClusterRejectedEdge;
    } else {
      retValUpd = prop.Update(yy, zz, cluster.row, param, clusterState, rejectChi2, refit, err2Y, err2Z);
    }
    GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamUpdateTrack, iTrk)) {
      merger.DebugStreamerUpdate(iTrk, ihit, xx, yy, zz, cluster, merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[cluster.num], *this, prop, inter, rejectChi2, refit, retValUpd, sumInvSqrtCharge / nAvgCharge * sumInvSqrtCharge / nAvgCharge, yy, zz, clusterState, retValInt, err2Y, err2Z);
    });
  }
  // clang-format off
  CADEBUG(if (!CheckCov()) GPUError("INVALID COV AFTER UPDATE!!!"));
  CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   FErr %d %d\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[3], "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], retValUpd, retValInt));
  // clang-format on

  ConstrainSinPhi();            // TODO: Limit using ConstrainSinPhi everywhere!
  if (!retValUpd && !retValInt) // track is updated
  {
    lastUpdateX = mX;
    nMissed = nMissed2 = 0;
    UnmarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagHighIncl);
    float dy = mP[0] - prop.Model().Y();
    float dz = mP[1] - prop.Model().Z();
    if (CAMath::Abs(mP[4]) * param.qptB5Scaler > 10 && --resetT0 <= 0 && CAMath::Abs(mP[2]) < 0.15f && dy * dy + dz * dz > 1) {
      CADEBUG(printf("Reinit linearization\n"));
      prop.SetTrack(this, prop.GetAlpha());
    }
    return 0;                                                                            // ok
  } else if (retValInt || retValUpd >= GPUTPCGMPropagator::updateErrorClusterRejected) { // cluster far away form the track
    if (retValInt || allowChangeClusters) {
      MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectDistance);
    } else if (finalFit) {
      MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, retValUpd >= GPUTPCGMPropagator::updateErrorClusterRejected ? GPUTPCGMMergedTrackHit::flagRejectDistance : GPUTPCGMMergedTrackHit::flagRejectErr);
    }
    if (!retValInt) {
      nMissed++;
      nMissed2++;
    }
    return -1; // cluster rejected
  } else {
    return 1; // bad chi2 for the whole track, stop the fit
  }
}

GPUdii() float GPUTPCGMTrackParam::FindBestInterpolatedHit(GPUTPCGMMerger& GPUrestrict() merger, gputpcgmmergertypes::InterpolationErrorHit& GPUrestrict() inter, const uint8_t sector, const uint8_t row, const float deltaZ, const float sumInvSqrtCharge, const int nAvgCharge, const GPUTPCGMPropagator& GPUrestrict() prop, const int32_t iTrk)
{
  const GPUParam& GPUrestrict() param = merger.Param();
  const GPUTPCTracker& GPUrestrict() tracker = *(merger.GetConstantMem()->tpcTrackers + sector);
  const GPUTPCRow& GPUrestrict() rowData = tracker.Row(row);
  GPUglobalref() const cahit2* hits = tracker.HitData(rowData);
  GPUglobalref() const calink* firsthit = tracker.FirstHitInBin(rowData);
  float uncorrectedY = -1e6f, uncorrectedZ;
  if (rowData.NHits() && inter.errorY >= (GPUCA_PAR_MERGER_INTERPOLATION_ERROR_TYPE_A)0) {
    const float zOffset = param.par.continuousTracking ? merger.GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(sector, mTOffset, param.continuousMaxTimeBin) : 0;
    const float y0 = rowData.Grid().YMin();
    const float stepY = rowData.HstepY();
    const float z0 = rowData.Grid().ZMin() - zOffset; // We can use our own ZOffset, since this is only used temporarily anyway
    const float stepZ = rowData.HstepZ();
    int32_t bin, ny, nz;

    float err2Y, err2Z;
    param.GetClusterErrors2(sector, row, mP[1], mP[2], mP[3], -1.f, 0.f, 0.f, err2Y, err2Z); // TODO: Use correct time/avgCharge

    const float Iz0 = inter.posY - mP[0];
    const float Iz1 = inter.posZ + deltaZ - mP[1];
    const float Iw0 = 1.f / (mC[0] + (float)inter.errorY);
    const float Iw2 = 1.f / (mC[2] + (float)inter.errorZ);
    const float Ik00 = mC[0] * Iw0;
    const float Ik11 = mC[2] * Iw2;
    const float ImP0 = mP[0] + Ik00 * Iz0;
    const float ImP1 = mP[1] + Ik11 * Iz1;
    const float ImC0 = mC[0] - Ik00 * mC[0];
    const float ImC2 = mC[2] - Ik11 * mC[2];

    merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(sector, row, ImP0, ImP1, uncorrectedY, uncorrectedZ);

    int32_t nCandidates = 0;
    while (nCandidates < param.rec.tpc.rebuildTrackInFitClusterCandidates && merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + nCandidates].id > 1) {
      nCandidates++;
    }
    if (CAMath::Abs(uncorrectedY) <= rowData.getTPCMaxY()) {
      const float kFactor = tracker.GetChiSeedFactor();
      const float sy2 = 4 * CAMath::Min(param.rec.tpc.hitSearchArea2, kFactor * (err2Y + CAMath::Abs(mC[0]))); // TODO: is 4 a good factor??
      const float sz2 = 4 * CAMath::Min(param.rec.tpc.hitSearchArea2, kFactor * (err2Z + CAMath::Abs(mC[2])));
      const float tubeY = CAMath::Sqrt(sy2);
      const float tubeZ = CAMath::Sqrt(sz2);
      rowData.Grid().GetBinArea(uncorrectedY, uncorrectedZ + zOffset, tubeY, tubeZ, bin, ny, nz);

      const int32_t nBinsY = rowData.Grid().Ny();
      const int32_t idOffset = tracker.Data().ClusterIdOffset();
      const int32_t* ids = &(tracker.Data().ClusterDataIndex()[rowData.HitNumberOffset()]);
      for (int32_t k = 0; k <= nz; k++) {
        const int32_t mybin = bin + k * nBinsY;
        const uint32_t hitFst = firsthit[mybin];
        const uint32_t hitLst = firsthit[mybin + ny + 1];
        for (uint32_t ih = hitFst; ih < hitLst; ih++) {
          const cahit2 hh = hits[ih];
          const float y = y0 + hh.x * stepY;
          const float z = z0 + hh.y * stepZ;
          const float dy = y - uncorrectedY;
          const float dz = z - uncorrectedZ;

          if (dy * dy < sy2 && dz * dz < sz2) {
            float err2YA, err2ZA;
            const ClusterNative& GPUrestrict() cl = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[idOffset + ids[ih]];
            const auto clflags = cl.getFlags() & GPUTPCGMMergedTrackHit::clustererAndSharedFlags;
            const float time = cl.getTime();
            const float invSqrtCharge = CAMath::InvSqrt(cl.qMax);
            const float invCharge = 1.f / cl.qMax;
            float invAvgCharge = (sumInvSqrtCharge + invSqrtCharge) / (nAvgCharge + 1);
            invAvgCharge *= invAvgCharge;

            prop.GetErr2(err2YA, err2ZA, param, mP[1], row, clflags, sector, time, invAvgCharge, invCharge);
            const float Jw0 = 1.f / (ImC0 + err2YA);
            const float Jw2 = 1.f / (ImC2 + err2ZA);
            const float chi2Y = Jw0 * dy * dy;
            const float chi2Z = Jw2 * dz * dz;
            bool ok = !prop.RejectCluster(chi2Y * param.rec.tpc.clusterRejectChi2TolleranceY, chi2Z * param.rec.tpc.clusterRejectChi2TolleranceZ, clflags);
            float err = dy * dy + dz * dz;
            if (ok) {
              int32_t insert = nCandidates;
              for (int32_t c = 0; c < nCandidates; c++) {
                if (err < merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + c].error) {
                  insert = c;
                  break;
                }
              }
              if (insert < param.rec.tpc.rebuildTrackInFitClusterCandidates) {
                for (int32_t c = CAMath::Min(nCandidates, param.rec.tpc.rebuildTrackInFitClusterCandidates - 1); c > insert; c--) {
                  merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + c] = merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + c - 1];
                }
                merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + insert] = {.id = (uint32_t)(idOffset + ids[ih] + 2), .row = row, .sector = sector, .error = err, .weight = 0, .best = 0};
                nCandidates += (nCandidates < param.rec.tpc.rebuildTrackInFitClusterCandidates);
              }
            }
          }
        }
      }
      CADEBUG(const auto* dbgCand = &merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates]; for (int dbgi = 0; dbgi < nCandidates; dbgi++) { if (dbgCand[dbgi].id > 1) printf("\t\t\tiTrk %d Row %d Candidate %d hit %d err %f\n", iTrk, (int)row, dbgi, dbgCand[dbgi].id - 2, dbgCand[dbgi].error); else break; });
    }
    if (nCandidates == 0) {
      merger.ClusterCandidates()[(iTrk * GPUTPCGeometry::NROWS + row) * param.rec.tpc.rebuildTrackInFitClusterCandidates + 0].id = 1;
    }
  }
  return uncorrectedY;
}

GPUdii() void GPUTPCGMTrackParam::DodEdx(GPUdEdx& GPUrestrict() dEdx, GPUdEdx& GPUrestrict() dEdxAlt, GPUTPCGMMerger& GPUrestrict() merger, bool finalFit, int ihit, int ihitMergeFirst, int wayDirection, const GPUTPCGMMergedTrackHit* GPUrestrict() clusters, uint8_t clusterState, float zz, uint8_t dEdxSubThresholdRow)
{
  const GPUParam& GPUrestrict() param = merger.Param();
  if GPUCA_RTC_CONSTEXPR (GPUCA_GET_CONSTEXPR(param.par, dodEdx)) {
    const GPUCalibObjectsConst& GPUrestrict() calib = merger.GetConstantMem()->calibObjects;
    const ClusterNative* GPUrestrict() clustersArray = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear;
    if (param.dodEdxEnabled && finalFit) { // TODO: Costimize flag to remove, and option to remove double-clusters
      bool acc = (clusterState & param.rec.tpc.dEdxClusterRejectionFlagMask) == 0, accAlt = (clusterState & param.rec.tpc.dEdxClusterRejectionFlagMaskAlt) == 0;
      if (acc || accAlt) {
        float qtot = 0, qmax = 0, pad = 0, relTime = 0;
        const int32_t clusterCount = CAMath::Abs(ihit - ihitMergeFirst) + 1;
        for (int32_t iTmp = ihitMergeFirst; iTmp != ihit + wayDirection; iTmp += wayDirection) {
          const ClusterNative& cl = clustersArray[clusters[iTmp].num];
          qtot += cl.qTot;
          qmax = CAMath::Max<float>(qmax, cl.qMax);
          pad += cl.getPad();
          relTime += cl.getTime();
        }
        qtot /= clusterCount; // TODO: Weighted Average
        pad /= clusterCount;
        relTime /= clusterCount;
        relTime = relTime - CAMath::Round(relTime);
        const auto& cluster = clusters[ihit];
        if (acc) {
          dEdx.fillCluster(qtot, qmax, cluster.row, cluster.sector, mP[2], mP[3], calib, zz, pad, relTime);
          if (dEdxSubThresholdRow) {
            dEdx.fillSubThreshold(dEdxSubThresholdRow);
          }
        }
        if GPUCA_RTC_CONSTEXPR (GPUCA_GET_CONSTEXPR(param.rec.tpc, dEdxClusterRejectionFlagMask) != GPUCA_GET_CONSTEXPR(param.rec.tpc, dEdxClusterRejectionFlagMaskAlt)) {
          if (accAlt) {
            dEdxAlt.fillCluster(qtot, qmax, cluster.row, cluster.sector, mP[2], mP[3], calib, zz, pad, relTime);
            if (dEdxSubThresholdRow) {
              dEdxAlt.fillSubThreshold(dEdxSubThresholdRow);
            }
          }
        }
      }
    }
  }
}

GPUdni() void GPUTPCGMTrackParam::MoveToReference(GPUTPCGMPropagator& GPUrestrict() prop, const GPUParam& GPUrestrict() param, float& GPUrestrict() Alpha)
{
  if (param.rec.tpc.trackReferenceX <= 500) {
    GPUTPCGMTrackParam save = *this;
    float saveAlpha = Alpha;
    for (int32_t attempt = 0; attempt < 3; attempt++) {
      float dAngle = CAMath::Round(CAMath::ATan2(mP[0], mX) / CAMath::Deg2Rad() / 20.f) * GPUTPCGeometry::kSectAngle();
      Alpha += dAngle;
      if (prop.PropagateToXAlpha(param.rec.tpc.trackReferenceX, Alpha, 0)) {
        break;
      }
      ConstrainSinPhi();
      if (CAMath::Abs(mP[0]) <= mX * CAMath::Tan(GPUTPCGeometry::kSectAngle() / 2.f)) {
        return;
      }
    }
    *this = save;
    Alpha = saveAlpha;
  }
  if (CAMath::Abs(mP[0]) > mX * CAMath::Tan(GPUTPCGeometry::kSectAngle() / 2.f)) {
    float dAngle = CAMath::Round(CAMath::ATan2(mP[0], mX) / CAMath::Deg2Rad() / 20.f) * GPUTPCGeometry::kSectAngle();
    Rotate(dAngle);
    ConstrainSinPhi();
    Alpha += dAngle;
  }
}

GPUd() void GPUTPCGMTrackParam::MirrorTo(GPUTPCGMPropagator& GPUrestrict() prop, float toY, float toZ, bool inFlyDirection, const GPUParam& GPUrestrict() param, uint8_t row, uint8_t clusterState, bool mirrorParameters, int8_t sector)
{
  if (mirrorParameters) {
    prop.Mirror(inFlyDirection);
  }
  float err2Y, err2Z;
  prop.GetErr2(err2Y, err2Z, param, toZ, row, clusterState, sector, -1.f, 0.f, 0.f); // Use correct time / avgCharge
  prop.Model().Y() = mP[0] = toY;
  prop.Model().Z() = mP[1] = toZ;
  if (mC[0] < err2Y) {
    mC[0] = err2Y;
  }
  if (mC[2] < err2Z) {
    mC[2] = err2Z;
  }
  if (CAMath::Abs(mC[5]) < 0.1f) {
    mC[5] = mC[5] > 0 ? 0.1f : -0.1f;
  }
  if (mC[9] < 1.f) {
    mC[9] = 1.f;
  }
  mC[1] = mC[4] = mC[6] = mC[8] = mC[11] = mC[13] = 0;
  prop.SetTrack(this, prop.GetAlpha());
  mNDF = -3;
  mChi2 = 0;
}

GPUd() int32_t GPUTPCGMTrackParam::MergeDoubleRowClusters(int32_t& ihit, int32_t wayDirection, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, const GPUTPCGMMerger& GPUrestrict() merger, GPUTPCGMPropagator& GPUrestrict() prop, float& GPUrestrict() xx, float& GPUrestrict() yy, float& GPUrestrict() zz, int32_t maxN, float clAlpha, uint8_t& GPUrestrict() clusterState, const bool markReject)
{
  const int32_t ihitFirst = ihit;
  {
    const ClusterNative& GPUrestrict() cl = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
    merger.GetConstantMem()->calibObjects.fastTransform->Transform(clusters[ihit].sector, clusters[ihit].row, cl.getPad(), cl.getTime(), xx, yy, zz, mTOffset);
  }
  if (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].sector == clusters[ihit + wayDirection].sector) {
    float maxDistY2, maxDistZ2;
    bool noReject = false; // Cannot reject if simple estimation of y/z fails (extremely unlike case)
    if (CAMath::Abs(clAlpha - prop.GetAlpha()) > 1.e-4f) {
      noReject = prop.RotateToAlpha(clAlpha);
    }
    float projY = 0, projZ = 0;
    if (!noReject) {
      noReject |= prop.GetPropagatedYZ(xx, projY, projZ);
    }
    prop.GetErr2(maxDistY2, maxDistZ2, merger.Param(), zz, clusters[ihit].row, 0, clusters[ihit].sector, -1.f, 0.f, 0.f); // TODO: Use correct time, avgCharge
    const float kFactor = merger.GetConstantMem()->tpcTrackers[0].GetChiSeedFactor() * 4.f;
    maxDistY2 = (maxDistY2 + mC[0]) * kFactor;
    maxDistZ2 = (maxDistZ2 + mC[2]) * kFactor;
    auto chkFunction = [clusters, markReject, maxDistY2, maxDistZ2, projY, projZ, noReject CADEBUG(, this)](int32_t ih, float y, float z) {
      float dy = y - projY;
      float dz = z - projZ;
      if (!noReject && (dy * dy > maxDistY2 || dz * dz > maxDistZ2)) {
        CADEBUG(printf("\t\t\tRejecting double-row cluster: dy %f, dz %f, chiY %f, chiZ %f (Y: trk %f prj %f cl %f - Z: trk %f prj %f cl %f)\n", dy, dz, sqrtf(maxDistY2), sqrtf(maxDistZ2), mP[0], projY, y, mP[1], projZ, z));
        if (markReject) {
          clusters[ih].state |= GPUTPCGMMergedTrackHit::flagRejectDistance;
        }
        return false;
      } else {
        CADEBUG(printf("\t\t\tMerging hit row %d Y %f Z %f (dy %f, dz %f, chiY %f, chiZ %f)\n", clusters[ih].row, y, z, dy, dz, sqrtf(maxDistY2), sqrtf(maxDistZ2)));
        return true;
      }
    };
    const float tmpX = xx;
    float count;
    if (chkFunction(ihit, yy, zz)) {
      const ClusterNative& GPUrestrict() cl = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
      const float clamp = cl.qTot;
      xx *= clamp;
      yy *= clamp;
      zz *= clamp;
      clusterState = clusters[ihit].state;
      count = clamp;
    } else {
      xx = yy = zz = count = 0.f;
      clusterState = 0;
    }
    do {
      ihit += wayDirection;
      const ClusterNative& GPUrestrict() cl = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
      const float clamp = cl.qTot;
      float clx, cly, clz;
      merger.GetConstantMem()->calibObjects.fastTransform->Transform(clusters[ihit].sector, clusters[ihit].row, cl.getPad(), cl.getTime(), clx, cly, clz, mTOffset);
      if (chkFunction(ihit, cly, clz)) {
        xx += clx * clamp;
        yy += cly * clamp;
        zz += clz * clamp;
        clusterState |= clusters[ihit].state;
        count += clamp;
      }
    } while (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].sector == clusters[ihit + wayDirection].sector);
    if (count < 0.1f) {
      CADEBUG(printf("\t\tNo matching cluster in double-row, skipping\n"));
      xx = tmpX;
      return -1;
    }
    xx /= count;
    yy /= count;
    zz /= count;
  }
  if (merger.Param().rec.tpc.rejectIFCLowRadiusCluster) {
    const float r2 = xx * xx + yy * yy;
    const float rmax2 = CAMath::Square(83.5f + merger.Param().rec.tpc.sysClusErrorMinDist);
    if (r2 < rmax2) {
      if (markReject) {
        MarkClusters(clusters, ihitFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectErr);
      }
      return -1;
    }
  }
  return 0;
}

GPUd() float GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger& GPUrestrict() merger, int32_t sector, int32_t iRow, int32_t iTrack, bool goodLeg, GPUTPCGMPropagator& prop)
{
  float Y, Z;
  float X = 0;
  merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoX(sector, iRow, mP[0], mP[1], X);
  if (prop.GetPropagatedYZ(X, Y, Z)) {
    Y = mP[0];
    Z = mP[1];
  }
  return AttachClusters(merger, sector, iRow, iTrack, goodLeg, Y, Z);
}

GPUd() float GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger& GPUrestrict() merger, int32_t sector, int32_t iRow, int32_t iTrack, bool goodLeg, float Y, float Z)
{
  const GPUParam& GPUrestrict() param = merger.Param();
  if (param.rec.tpc.disableRefitAttachment & 1) {
    return -1e6f;
  }
  const GPUTPCTracker& GPUrestrict() tracker = *(merger.GetConstantMem()->tpcTrackers + sector);
  const GPUTPCRow& GPUrestrict() row = tracker.Row(iRow);
  GPUglobalref() const cahit2* hits = tracker.HitData(row);
  GPUglobalref() const calink* firsthit = tracker.FirstHitInBin(row);
  if (row.NHits() == 0) {
    return -1e6f;
  }

  const float zOffset = param.par.continuousTracking ? merger.GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(sector, mTOffset, param.continuousMaxTimeBin) : 0; // TODO: do some validatiomns for the transform conv functions...
  const float y0 = row.Grid().YMin();
  const float stepY = row.HstepY();
  const float z0 = row.Grid().ZMin() - zOffset; // We can use our own ZOffset, since this is only used temporarily anyway
  const float stepZ = row.HstepZ();
  int32_t bin, ny, nz;

  float uncorrectedY, uncorrectedZ;
  merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformYZtoNominalYZ(sector, iRow, Y, Z, uncorrectedY, uncorrectedZ);
  if (CAMath::Abs(uncorrectedY) > row.getTPCMaxY()) {
    return uncorrectedY;
  }

  bool protect = CAMath::Abs(GetQPt() * param.qptB5Scaler) <= param.rec.tpc.rejectQPtB5 && goodLeg;
  float err2Y, err2Z;
  param.GetClusterErrors2(sector, iRow, Z, mP[2], mP[3], -1.f, 0.f, 0.f, err2Y, err2Z); // TODO: Use correct time/avgCharge
  const float tubeMaxSize2 = protect ? param.rec.tpc.tubeProtectMaxSize2 : param.rec.tpc.tubeRemoveMaxSize2;
  const float tubeMinSize2 = protect ? param.rec.tpc.tubeProtectMinSize2 : 0.f;
  float tubeSigma2 = protect ? param.rec.tpc.tubeProtectSigma2 : param.rec.tpc.tubeRemoveSigma2;
  uint32_t pad = CAMath::Float2UIntRn(GPUTPCGeometry::LinearY2Pad(sector, iRow, uncorrectedY));
  float time = merger.GetConstantMem()->calibObjects.fastTransform->InverseTransformInTimeFrame(sector, uncorrectedZ + (param.par.continuousTracking ? merger.GetConstantMem()->calibObjects.fastTransform->convVertexTimeToZOffset(sector, mTOffset, param.continuousMaxTimeBin) : 0), param.continuousMaxTimeBin); // TODO: Simplify this call in TPCFastTransform
  if (iRow < param.rec.tpc.tubeExtraProtectMinRow ||
      pad < param.rec.tpc.tubeExtraProtectEdgePads || pad >= (uint32_t)(GPUTPCGeometry::NPads(iRow) - param.rec.tpc.tubeExtraProtectEdgePads) ||
      param.GetUnscaledMult(time) / GPUTPCGeometry::Row2X(iRow) > param.rec.tpc.tubeExtraProtectMinOccupancy) {
    tubeSigma2 *= protect ? 2 : 0.5;
  }
  const float sy2 = CAMath::Max(tubeMinSize2, CAMath::Min(tubeMaxSize2, tubeSigma2 * (err2Y + CAMath::Abs(mC[0])))); // Cov can be bogus when following circle
  const float sz2 = CAMath::Max(tubeMinSize2, CAMath::Min(tubeMaxSize2, tubeSigma2 * (err2Z + CAMath::Abs(mC[2])))); // In that case we should provide the track error externally
  const float tubeY = CAMath::Sqrt(sy2);
  const float tubeZ = CAMath::Sqrt(sz2);
  const float sy21 = 1.f / sy2;
  const float sz21 = 1.f / sz2;

  row.Grid().GetBinArea(uncorrectedY, uncorrectedZ + zOffset, tubeY, tubeZ, bin, ny, nz);
  const int32_t nBinsY = row.Grid().Ny();
  const int32_t idOffset = tracker.Data().ClusterIdOffset();
  const int32_t* ids = &(tracker.Data().ClusterDataIndex()[row.HitNumberOffset()]);
  uint32_t myWeight = merger.TrackOrderAttach()[iTrack] | gputpcgmmergertypes::attachAttached | gputpcgmmergertypes::attachTube;
  GPUAtomic(uint32_t)* const weights = merger.ClusterAttachment();
  if (goodLeg) {
    myWeight |= gputpcgmmergertypes::attachGoodLeg;
  }
  if (protect) {
    myWeight |= gputpcgmmergertypes::attachProtect;
  }

  for (int32_t k = 0; k <= nz; k++) {
    const int32_t mybin = bin + k * nBinsY;
    const uint32_t hitFst = firsthit[mybin];
    const uint32_t hitLst = firsthit[mybin + ny + 1];
    for (uint32_t ih = hitFst; ih < hitLst; ih++) {
      int32_t id = idOffset + ids[ih];
      GPUAtomic(uint32_t)* const weight = weights + id;
      if constexpr (GPUCA_PAR_NO_ATOMIC_PRECHECK == 0) {
        if (myWeight <= *weight) {
          continue;
        }
      }
      const cahit2 hh = hits[ih];
      const float y = y0 + hh.x * stepY;
      const float z = z0 + hh.y * stepZ;
      const float dy = y - uncorrectedY;
      const float dz = z - uncorrectedZ;
      if (dy * dy * sy21 + dz * dz * sz21 <= CAMath::Sqrt(2.f)) {
        // CADEBUG(printf("Found Y %f Z %f\n", y, z));
        CAMath::AtomicMax(weight, myWeight);
      }
    }
  }
  return uncorrectedY;
}

GPUdii() void GPUTPCGMTrackParam::StoreOuter(gputpcgmmergertypes::GPUTPCOuterParam* outerParam, float alpha)
{
  CADEBUG(printf("\t%21sStorO   Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f, SP %5.2f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f\n", "", alpha, mX, mP[0], mP[1], mP[4], mP[2], sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14])));
  for (int32_t i = 0; i < 5; i++) {
    outerParam->P[i] = mP[i];
  }
  for (int32_t i = 0; i < 15; i++) {
    outerParam->C[i] = mC[i];
  }
  outerParam->X = mX;
  outerParam->alpha = alpha;
}

GPUdic(0, 1) void GPUTPCGMTrackParam::StoreLoopPropagation(const GPUTPCGMMerger& GPUrestrict() merger, int32_t sector, int32_t iRow, int32_t iTrack, bool outwards, float alpha)
{
  if (iRow == 0 || iRow == GPUTPCGeometry::NROWS - 1) {
    return;
  }
  if (CAMath::Abs(mP[2]) >= constants::MAX_SIN_PHI) { // TODO: How can we avoid this?
    return;
  }
  if (CAMath::Abs(mP[2]) < 0.75) {
    return;
  }
  if ((mP[2] * mP[4] < 0) ^ outwards) {
    return;
  }

  uint32_t nLoopData = CAMath::AtomicAdd(&merger.Memory()->nLoopData, 1u);
  if (nLoopData >= merger.NMaxTracks()) {
    merger.raiseError(GPUErrors::ERROR_MERGER_LOOPER_OVERFLOW, nLoopData, merger.NMaxTracks());
    CAMath::AtomicExch(&merger.Memory()->nLoopData, merger.NMaxTracks());
    return;
  }
  GPUTPCGMLoopData data;
  data.param = *this;
  data.track = iTrack;
  data.alpha = alpha;
  data.sector = sector;
  data.row = iRow;
  data.outwards = outwards;
  merger.LoopData()[nLoopData] = data;
}

GPUdii() void GPUTPCGMTrackParam::PropagateLooper(const GPUTPCGMMerger& GPUrestrict() merger, int32_t loopIdx)
{
  GPUTPCGMPropagator prop;
  prop.SetMaterialTPC();
  prop.SetPolynomialField(&merger.Param().polynomialField);
  prop.SetMaxSinPhi(constants::MAX_SIN_PHI);
  prop.SetMatLUT(merger.Param().rec.useMatLUT ? merger.GetConstantMem()->calibObjects.matLUT : nullptr);
  prop.SetSeedingErrors(false);
  prop.SetFitInProjections(true);
  prop.SetPropagateBzOnly(false);

  GPUTPCGMLoopData& data = merger.LoopData()[loopIdx];
  prop.SetTrack(&data.param, data.alpha);
  if (merger.Param().rec.tpc.looperFollowMode == 1) {
    data.param.AttachClustersLooperFollow(merger, prop, data.sector, data.track, data.outwards);
  } else {
    data.param.AttachClustersLooper(merger, data.sector, data.row, data.track, data.outwards, prop);
  }
}

GPUdi() void GPUTPCGMTrackParam::AttachClustersLooperFollow(const GPUTPCGMMerger& GPUrestrict() merger, GPUTPCGMPropagator& GPUrestrict() prop, int32_t sector, const int32_t iTrack, const bool up)
{
  float toX = mX;
  bool inFlyDirection = (merger.MergedTracks()[iTrack].Leg() & 1) ^ up;

  const GPUParam& GPUrestrict() param = merger.Param();
  bool right = (mP[2] < 0) ^ up;
  const int32_t sectorSide = sector >= (int32_t)(GPUTPCGeometry::NSECTORS / 2) ? (GPUTPCGeometry::NSECTORS / 2) : 0;
  float lrFactor = right ^ !up ? 1.f : -1.f;
  // clang-format off
  CADEBUG(printf("\nCIRCLE Track %d: Sector %d Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f QPt %f - Right %d Up %d lrFactor %f\n", iTrack, sector, prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3], mP[4], (int32_t)right, (int32_t)up, lrFactor));
  // clang-format on

  if (prop.RotateToAlpha(prop.GetAlpha() + (CAMath::Pi() / 2.f) * lrFactor)) {
    return;
  }
  CADEBUG(printf("\tRotated: X %f Y %f Z %f SinPhi %f (Alpha %f / %f)\n", mP[0], mX, mP[1], mP[2], prop.GetAlpha(), prop.GetAlpha() + CAMath::Pi() / 2.f));
  uint32_t maxTries = 100;
  while (true) {
    while (CAMath::Abs(mX) <= CAMath::Abs(mP[0]) * CAMath::Tan(GPUTPCGeometry::kSectAngle() / 2.f) + 0.1f) {
      if (maxTries-- == 0) {
        return;
      }
      if (CAMath::Abs(mP[2]) > 0.7f) {
        return;
      }
      if (up ? (-mP[0] * lrFactor > GPUTPCGeometry::Row2X(GPUTPCGeometry::NROWS - 1)) : (-mP[0] * lrFactor < GPUTPCGeometry::Row2X(0))) {
        return;
      }
      if (!((up ? (-mP[0] * lrFactor >= toX) : (-mP[0] * lrFactor <= toX)) || (right ^ (mP[2] > 0)))) {
        return;
      }
      int32_t err = prop.PropagateToXAlpha(mX + (up ? 1.f : -1.f), prop.GetAlpha(), inFlyDirection);
      if (err) {
        CADEBUG(printf("\t\tpropagation error (%d)\n", err));
        return;
      }
      CADEBUG(printf("\tPropagated to y = %f: X %f Z %f SinPhi %f\n", mX, mP[0], mP[1], mP[2]));
      for (uint32_t j = 0; j < GPUTPCGeometry::NROWS; j++) { // TODO: Avoid iterating over all rows
        float rowX = GPUTPCGeometry::Row2X(j);
        if (CAMath::Abs(rowX - (-mP[0] * lrFactor)) < 1.5f) {
          CADEBUG(printf("\t\tAttempt row %d (X %f Y %f Z %f)\n", j, rowX, mX * lrFactor, mP[1]));
          AttachClusters(merger, sector, j, iTrack, false, mX * lrFactor, mP[1]);
        }
      }
    }
    if (maxTries-- == 0) {
      return;
    }
    if (right) {
      if (++sector >= sectorSide + 18) {
        sector -= 18;
      }
    } else {
      if (--sector < sectorSide) {
        sector += 18;
      }
    }
    CADEBUG(printf("\tRotating to sector %d: %f --> %f\n", sector, prop.GetAlpha(), param.Alpha(sector) + (CAMath::Pi() / 2.f) * lrFactor));
    int32_t err = prop.RotateToAlpha(param.Alpha(sector) + (CAMath::Pi() / 2.f) * lrFactor);
    if (err) {
      CADEBUG(printf("Rotation Error %d\n", err));
      return;
    }
    CADEBUG(printf("\tAfter Rotating Alpha %f Position X %f Y %f Z %f SinPhi %f\n", prop.GetAlpha(), mP[0], mX, mP[1], mP[2]));
  }
}

GPUdi() void GPUTPCGMTrackParam::AttachClustersLooper(const GPUTPCGMMerger& GPUrestrict() merger, int32_t sector, int32_t iRow, const int32_t iTrack, const bool up, const GPUTPCGMPropagator& GPUrestrict() prop)
{
  // Note that the coordinate system is rotated by 90 degree swapping X and Y!
  float X = mP[2] > 0 ? mP[0] : -mP[0];
  float Y = mP[2] > 0 ? -mX : mX;
  float Z = mP[1];
  float SinPhi = CAMath::Sqrt(1 - mP[2] * mP[2]) * (mP[2] > 0 ? -1 : 1);
  float b = prop.GetBz(prop.GetAlpha(), mX, mP[0], mP[1]);

  float dx = up ? 1.f : -1.f;
  const float myRowX = GPUTPCGeometry::Row2X(iRow);
  // printf("\nAttachMirror sector %d row %d outwards %d\n", (int)sector, (int)iRow, (int)outwards);
  // printf("X %f Y %f Z %f SinPhi %f -->\n", mX, mP[0], mP[1], mP[2]);
  // printf("X %f Y %f Z %f SinPhi %f, dx %f\n", X, Y, Z, SinPhi, dx);
  uint32_t maxTries = 100;
  while (maxTries--) {
    float ex = CAMath::Sqrt(1 - SinPhi * SinPhi);
    float exi = 1.f / ex;
    float dxBzQ = dx * -b * mP[4];
    float newSinPhi = SinPhi + dxBzQ;
    if (CAMath::Abs(newSinPhi) > constants::MAX_SIN_PHI_LOW) {
      // printf("Abort, newSinPhi %f\n", newSinPhi);
      return;
    }
    if (mP[2] > 0 ? (newSinPhi > 0.5) : (newSinPhi < -0.5)) {
      // printf("Finished, newSinPhi %f\n", newSinPhi);
      return;
    }
    float dS = dx * exi;
    float h2 = dS * exi * exi;
    float h4 = .5f * h2 * dxBzQ;

    X += dx;
    Y += dS * SinPhi + h4;
    Z += dS * mP[3];
    SinPhi = newSinPhi;
    if (CAMath::Abs(X) > CAMath::Abs(Y) * CAMath::Tan(GPUTPCGeometry::kSectAngle() / 2.f)) {
      // printf("Abort, sector edge\n");
      return;
    }

    // printf("count %d: At X %f Y %f Z %f SinPhi %f\n", maxTries, mP[2] > 0 ? -Y : Y, mP[2] > 0 ? X : -X, Z, SinPhi);
    float paramX = mP[2] > 0 ? -Y : Y;
    int32_t step = up ? 1 : -1;
    int32_t found = 0;
    for (int32_t j = iRow; j >= 0 && j < (int32_t)GPUTPCGeometry::NROWS && found < 3; j += step) {
      float rowX = mX + GPUTPCGeometry::Row2X(j) - myRowX;
      if (CAMath::Abs(rowX - paramX) < 1.5f) {
        // printf("Attempt row %d at y %f\n", j, X);
        AttachClusters(merger, sector, j, iTrack, false, mP[2] > 0 ? X : -X, Z);
      }
    }
  }
}

GPUd() float GPUTPCGMTrackParam::ShiftZ(const GPUTPCGMMergedTrackHit* GPUrestrict() clusters, const GPUTPCGMMerger& GPUrestrict() merger, int32_t N)
{
  if (N == 0) {
    N = 1;
  }
  const auto& GPUrestrict() cls = merger.GetConstantMem()->ioPtrs.clustersNative->clustersLinear;
  float z0 = cls[clusters[0].num].getTime(), zn = cls[clusters[N - 1].num].getTime();
  const auto tmp = zn > z0 ? std::array<float, 3>{zn, z0, GPUTPCGeometry::Row2X(clusters[N - 1].row)} : std::array<float, 3>{z0, zn, GPUTPCGeometry::Row2X(clusters[0].row)};
  return ShiftZ(merger, clusters[0].sector, tmp[0], tmp[1], tmp[2]);
}

GPUd() float GPUTPCGMTrackParam::ShiftZ(const GPUTPCGMMerger& GPUrestrict() merger, uint32_t sector, float cltmax, float cltmin, float clx)
{
  const GPUParam& GPUrestrict() param = merger.Param();
  if (!param.par.continuousTracking) {
    return 0.f;
  }
  float deltaZ = 0.f;
  bool beamlineReached = false;
  const float r1 = CAMath::Max(0.0001f, CAMath::Abs(mP[4] * param.polynomialField.GetNominalBz()));
  if (r1 < 0.01501) { // 100 MeV @ 0.5T ~ 0.66m cutof
    const float dist2 = mX * mX + mP[0] * mP[0];
    const float dist1r2 = dist2 * r1 * r1;
    if (dist1r2 < 4) {
      const float alpha = CAMath::ACos(1 - 0.5f * dist1r2); // Angle of a circle, such that |(cosa, sina) - (1,0)| == dist
      const float beta = CAMath::ATan2(mP[0], mX);
      const int32_t comp = mP[2] > CAMath::Sin(beta);
      const float sinab = CAMath::Sin((comp ? 0.5f : -0.5f) * alpha + beta); // Angle of circle through origin and track position, to be compared to Snp
      const float res = CAMath::Abs(sinab - mP[2]);

      if (res < 0.2) {
        const float r = 1.f / r1;
        const float dS = alpha * r;
        float z0 = dS * mP[3];
        if (CAMath::Abs(z0) > GPUTPCGeometry::TPCLength()) {
          z0 = z0 > 0 ? GPUTPCGeometry::TPCLength() : -GPUTPCGeometry::TPCLength();
        }
        deltaZ = mP[1] - z0;
        beamlineReached = true;

        // printf("X %9.3f Y %9.3f QPt %9.3f R %9.3f --> Alpha %9.3f Snp %9.3f Snab %9.3f Res %9.3f dS %9.3f z0 %9.3f\n", mX, mP[0], mP[4], r, alpha / 3.1415 * 180, mP[2], sinab, res, dS, z0);
      }
    }
  }

  if (!beamlineReached) {
    float refZ = ((sector < GPUTPCGeometry::NSECTORS / 2) ? param.rec.tpc.defaultZOffsetOverR : -param.rec.tpc.defaultZOffsetOverR) * clx;
    float basez;
    merger.GetConstantMem()->calibObjects.fastTransform->TransformIdealZ(sector, cltmax, basez, mTOffset);
    deltaZ = basez - refZ;
  }
  {
    float deltaT = merger.GetConstantMem()->calibObjects.fastTransform->convDeltaZtoDeltaTimeInTimeFrame(sector, deltaZ);
    mTOffset += deltaT;
    const float maxT = cltmin - merger.GetConstantMem()->calibObjects.fastTransform->getT0();
    const float minT = cltmax - merger.GetConstantMem()->calibObjects.fastTransform->getMaxDriftTime(sector);
    // printf("T Check: Clusters %f %f, min %f max %f vtx %f\n", tz1, tz2, minT, maxT, mTOffset);
    deltaT = 0.f;
    if (mTOffset < minT) {
      deltaT = minT - mTOffset;
    }
    if (mTOffset + deltaT > maxT) {
      deltaT = maxT - mTOffset;
    }
    if (deltaT != 0.f) {
      deltaZ += merger.GetConstantMem()->calibObjects.fastTransform->convDeltaTimeToDeltaZinTimeFrame(sector, deltaT);
      // printf("Moving clusters to TPC Range: QPt %f, New mTOffset %f, t1 %f, t2 %f, Shift %f in Z: %f to %f --> %f to %f in T\n", mP[4], mTOffset + deltaT, tz1, tz2, deltaZ, tz2 - mTOffset, tz1 - mTOffset, tz2 - mTOffset - deltaT, tz1 - mTOffset - deltaT);
      mTOffset += deltaT;
    }
    mP[1] -= deltaZ;
  }
  // printf("\n");
  return -deltaZ;
}

GPUd() bool GPUTPCGMTrackParam::CheckCov() const
{
  const float* c = mC;
  bool ok = c[0] >= 0 && c[2] >= 0 && c[5] >= 0 && c[9] >= 0 && c[14] >= 0 && (c[1] * c[1] <= c[2] * c[0]) && (c[3] * c[3] <= c[5] * c[0]) && (c[4] * c[4] <= c[5] * c[2]) && (c[6] * c[6] <= c[9] * c[0]) && (c[7] * c[7] <= c[9] * c[2]) && (c[8] * c[8] <= c[9] * c[5]) &&
            (c[10] * c[10] <= c[14] * c[0]) && (c[11] * c[11] <= c[14] * c[2]) && (c[12] * c[12] <= c[14] * c[5]) && (c[13] * c[13] <= c[14] * c[9]);
  return ok;
}

GPUd() bool GPUTPCGMTrackParam::CheckNumericalQuality(float overrideCovYY) const
{
  //* Check that the track parameters and covariance matrix are reasonable
  bool ok = CAMath::Finite(mX) && CAMath::Finite(mChi2);
  // CADEBUG(printf("OK %d - %f - ", (int32_t)ok, mX); for (int32_t i = 0; i < 5; i++) { printf("%f ", mP[i]); } printf(" - "); for (int32_t i = 0; i < 15; i++) { printf("%f ", mC[i]); } printf("\n"));
  const float* c = mC;
  for (int32_t i = 0; i < 15; i++) {
    ok = ok && CAMath::Finite(c[i]);
  }
  for (int32_t i = 0; i < 5; i++) {
    ok = ok && CAMath::Finite(mP[i]);
  }
  if ((overrideCovYY > 0 ? overrideCovYY : c[0]) > 4.f * 4.f || c[2] > 4.f * 4.f || c[5] > 2.f * 2.f || c[9] > 2.f * 2.f) {
    ok = 0;
  }
  if (CAMath::Abs(mP[2]) > constants::MAX_SIN_PHI) {
    ok = 0;
  }
  if (!CheckCov()) {
    ok = false;
  }
  return ok;
}

GPUdii() void GPUTPCGMTrackParam::RefitTrack(GPUTPCGMMergedTrack& GPUrestrict() track, int32_t iTrk, GPUTPCGMMerger& GPUrestrict() merger, bool rebuilt) // VS: GPUd changed to GPUdii. No change in output and no performance penalty.
{
  if (!track.OK()) {
    return;
  }

  CADEBUG(if (DEBUG_SINGLE_TRACK != -1 && iTrk != ((DEBUG_SINGLE_TRACK == -2 && getenv("DEBUG_TRACK")) ? atoi(getenv("DEBUG_TRACK")) : DEBUG_SINGLE_TRACK)) { track.SetNClusters(0); track.SetOK(0); return; });

  int32_t nTrackHits = track.NClusters();
  int32_t NTolerated = 0; // Clusters not fit but tollerated for track length cut
  GPUTPCGMTrackParam t = track.Param();
  float Alpha = track.Alpha();
  bool ok = t.Fit(merger, iTrk, nTrackHits, NTolerated, Alpha, track, rebuilt);
  CADEBUG(if (!merger.Param().rec.tpc.rebuildTrackInFit || rebuilt) printf("Finished Fit Track %7d --- OUTPUT hits %d -> %d+%d = %d, QPt %f -> %f, SP %f, OK %d chi2 %f chi2ndf %f\n", iTrk, track.NClusters(), nTrackHits, NTolerated, nTrackHits + NTolerated, track.GetParam().GetQPt(), t.QPt(), t.SinPhi(), (int32_t)ok, t.Chi2(), t.Chi2() / CAMath::Max(1, nTrackHits)));

  if (CAMath::Abs(t.QPt()) < 1.e-4f) {
    t.QPt() = CAMath::Copysign(1.e-4f, t.QPt());
  }

  CADEBUG(if (t.GetX() > 250) { printf("ERROR, Track %d at impossible X %f, Pt %f, Looper %d\n", iTrk, t.GetX(), CAMath::Abs(1.f / t.QPt()), (int32_t)merger.MergedTracks()[iTrk].Looper()); });

  track.SetOK(ok);                                                               // TODO: Should we recover tracks who failed the fit in iWay0/1 for the rebuild?
  if (t.GetNDF() <= 0 && !rebuilt && merger.Param().rec.tpc.rebuildTrackInFit) { // TODO: Better handling of NDF<0 tracks, how do we want to do cluster rejection?
    track.Param().NDF() = 0;
  } else {
    track.Param() = t;
    track.Alpha() = Alpha;
  }
  if (!merger.Param().rec.tpc.rebuildTrackInFit || rebuilt) {
    track.SetNClustersFitted(nTrackHits);
  }

  // if (track.OK()) merger.DebugRefitMergedTrack(track);
}

GPUd() void GPUTPCGMTrackParam::Rotate(float alpha)
{
  float cA, sA;
  CAMath::SinCos(alpha, sA, cA);
  float x0 = mX;
  float sinPhi0 = mP[2], cosPhi0 = CAMath::Sqrt(1 - mP[2] * mP[2]);
  float cosPhi = cosPhi0 * cA + sinPhi0 * sA;
  float sinPhi = -cosPhi0 * sA + sinPhi0 * cA;
  float j0 = cosPhi0 / cosPhi;
  float j2 = cosPhi / cosPhi0;
  mX = x0 * cA + mP[0] * sA;
  mP[0] = -x0 * sA + mP[0] * cA;
  mP[2] = sinPhi;
  mC[0] *= j0 * j0;
  mC[1] *= j0;
  mC[3] *= j0;
  mC[6] *= j0;
  mC[10] *= j0;

  mC[3] *= j2;
  mC[4] *= j2;
  mC[5] *= j2 * j2;
  mC[8] *= j2;
  mC[12] *= j2;
  if (cosPhi < 0) { // change direction ( t0 direction is already changed in t0.UpdateValues(); )
    SinPhi() = -SinPhi();
    DzDs() = -DzDs();
    QPt() = -QPt();
    mC[3] = -mC[3];
    mC[4] = -mC[4];
    mC[6] = -mC[6];
    mC[7] = -mC[7];
    mC[10] = -mC[10];
    mC[11] = -mC[11];
  }
}

GPUd() void GPUTPCGMTrackParam::AddCovDiagErrors(const float* GPUrestrict() errors2)
{
  mC[0] += errors2[0];
  mC[2] += errors2[1];
  mC[5] += errors2[2];
  mC[9] += errors2[3];
  mC[14] += errors2[4];
}

GPUd() void GPUTPCGMTrackParam::AddCovDiagErrorsWithCorrelations(const float* GPUrestrict() errors2)
{
  const int32_t diagMap[5] = {0, 2, 5, 9, 14};
  const float oldDiag[5] = {mC[0], mC[2], mC[5], mC[9], mC[14]};
  for (int32_t i = 0; i < 5; i++) {
    mC[diagMap[i]] += errors2[i];
    for (int32_t j = 0; j < i; j++) {
      mC[diagMap[i - 1] + j + 1] *= gpu::CAMath::Sqrt(mC[diagMap[i]] * mC[diagMap[j]] / (oldDiag[i] * oldDiag[j]));
    }
  }
}
