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
#define EXTRACT_RESIDUALS 0

#if EXTRACT_RESIDUALS == 1
#include "GPUROOTDump.h"
#endif

#include "GPUTPCDef.h"
#include "GPUTPCGMTrackParam.h"
#include "GPUTPCGMPhysicalTrackModel.h"
#include "GPUTPCGMPropagator.h"
#include "GPUTPCGMBorderTrack.h"
#include "GPUTPCGMMergedTrack.h"
#include "GPUTPCGMPolynomialField.h"
#include "GPUTPCGMMerger.h"
#include "GPUTPCTracker.h"
#include "GPUTPCClusterData.h"
#include "GPUdEdx.h"
#include "GPUParam.h"
#include "GPUO2DataTypes.h"
#include "GPUConstantMem.h"
#include "TPCFastTransform.h"
#include "CorrectionMapsHelper.h"
#include "GPUTPCConvertImpl.h"
#include "GPUTPCGMMergerTypes.h"
#include "GPUParam.inc"

#ifdef GPUCA_ALIROOT_LIB
#include "AliExternalTrackParam.h"
#endif

#ifdef GPUCA_CADEBUG_ENABLED
#include "../utils/qconfig.h"
#include "AliHLTTPCClusterMCData.h"
#endif

#ifndef GPUCA_GPUCODE_DEVICE
#include <cmath>
#include <cstdlib>
#endif

using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;

static constexpr float kDeg2Rad = M_PI / 180.f;
static constexpr float kSectAngle = 2 * M_PI / 18.f;

GPUd() bool GPUTPCGMTrackParam::Fit(GPUTPCGMMerger* GPUrestrict() merger, int iTrk, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, GPUTPCGMMergedTrackHitXYZ* GPUrestrict() clustersXYZ, int& GPUrestrict() N, int& GPUrestrict() NTolerated, float& GPUrestrict() Alpha, int attempt, float maxSinPhi, gputpcgmmergertypes::GPUTPCOuterParam* GPUrestrict() outerParam)
{
  const GPUParam& GPUrestrict() param = merger->Param();

  GPUdEdx dEdx;
  GPUTPCGMPropagator prop;
  gputpcgmmergertypes::InterpolationErrors interpolation;
  prop.SetMaterialTPC();
  prop.SetPolynomialField(&merger->Param().polynomialField);
  prop.SetMaxSinPhi(maxSinPhi);
  prop.SetToyMCEventsFlag(param.par.toyMCEventsFlag);
  if ((clusters[0].slice < 18) == (clusters[N - 1].slice < 18)) {
    ShiftZ2(clusters, clustersXYZ, merger, N);
  }
  if (param.rec.tpc.mergerInterpolateErrors) {
    for (int i = 0; i < N; i++) {
      interpolation.hit[i].errorY = -1;
    }
  }

  const int nWays = param.rec.tpc.nWays;
  const int maxN = N;
  int ihitStart = 0;
  float covYYUpd = 0.f;
  float lastUpdateX = -1.f;
  unsigned char lastRow = 255;
  unsigned char lastSlice = 255;
  unsigned char storeOuter = 0;

  for (int iWay = 0; iWay < nWays; iWay++) {
    int nMissed = 0, nMissed2 = 0;
    if (iWay && storeOuter != 255 && param.rec.tpc.nWaysOuter && outerParam) {
      storeOuter = 0;
      if (iWay == nWays - 1) {
        StoreOuter(outerParam, prop, 0);
        if (merger->OutputTracks()[iTrk].Looper()) {
          storeOuter = 1;
        }
      } else if (iWay == nWays - 2 && merger->OutputTracks()[iTrk].Looper()) {
        storeOuter = 2;
      }
    }

    int resetT0 = initResetT0();
    const bool refit = (nWays == 1 || iWay >= 1);
    const float maxSinForUpdate = CAMath::Sin(70.f * kDeg2Rad);

    ResetCovariance();
    prop.SetSeedingErrors(!(refit && attempt == 0));
    prop.SetFitInProjections(param.rec.fitInProjections == -1 ? (iWay != 0) : param.rec.fitInProjections);
    prop.SetPropagateBzOnly(param.rec.fitPropagateBzOnly > iWay);
    prop.SetMatLUT((param.rec.useMatLUT && iWay == nWays - 1) ? merger->GetConstantMem()->calibObjects.matLUT : nullptr);
    prop.SetTrack(this, iWay ? prop.GetAlpha() : Alpha);
    ConstrainSinPhi(prop.GetFitInProjections() ? 0.95f : GPUCA_MAX_SIN_PHI_LOW);
    CADEBUG(printf("Fitting track %d way %d (sector %d, alpha %f)\n", iTrk, iWay, (int)(prop.GetAlpha() / kSectAngle + 0.5) + (mP[1] < 0 ? 18 : 0), prop.GetAlpha()));

    N = 0;
    lastUpdateX = -1;
    const bool inFlyDirection = iWay & 1;
    unsigned char lastLeg = clusters[ihitStart].leg;
    const int wayDirection = (iWay & 1) ? -1 : 1;

    bool noFollowCircle = false, noFollowCircle2 = false;
    int goodRows = 0;
    for (int ihit = ihitStart; ihit >= 0 && ihit < maxN; ihit += wayDirection) {
      const bool crossCE = lastSlice != 255 && ((lastSlice < 18) ^ (clusters[ihit].slice < 18));
      if (crossCE) {
        lastSlice = clusters[ihit].slice;
        noFollowCircle2 = true;
      }

      if (storeOuter == 2 && clusters[ihit].leg == clusters[maxN - 1].leg - 1) {
        if (lastLeg == clusters[maxN - 1].leg) {
          StoreOuter(outerParam, prop, 1);
          storeOuter = 255;
        } else {
          storeOuter = 0;
        }
      }

      unsigned char clusterState = clusters[ihit].state;
      const float clAlpha = param.Alpha(clusters[ihit].slice);
      if ((param.rec.tpc.trackFitRejectMode > 0 && nMissed >= param.rec.tpc.trackFitRejectMode) || nMissed2 >= param.rec.tpc.trackFitMaxRowMissedHard || clusters[ihit].state & GPUTPCGMMergedTrackHit::flagReject) {
        CADEBUG(printf("\tSkipping hit, %d hits rejected, flag %X\n", nMissed, (int)clusters[ihit].state));
        if (iWay + 2 >= nWays && !(clusters[ihit].state & GPUTPCGMMergedTrackHit::flagReject)) {
          clusters[ihit].state |= GPUTPCGMMergedTrackHit::flagRejectErr;
        }
        continue;
      }

      const bool allowModification = refit && (iWay == 0 || (((nWays - iWay) & 1) ? (ihit >= CAMath::Min(maxN / 2, 30)) : (ihit <= CAMath::Max(maxN / 2, maxN - 30))));
      int ihitMergeFirst = ihit;
      prop.SetStatErrorCurCluster(&clusters[ihit]);

      float xx, yy, zz;
      if (merger->Param().par.earlyTpcTransform) {
        const float zOffset = (clusters[ihit].slice < 18) == (clusters[0].slice < 18) ? mTZOffset : -mTZOffset;
        xx = clustersXYZ[ihit].x;
        yy = clustersXYZ[ihit].y;
        zz = clustersXYZ[ihit].z - zOffset;
      } else {
        const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
        merger->GetConstantMem()->calibObjects.fastTransformHelper->Transform(clusters[ihit].slice, clusters[ihit].row, cl.getPad(), cl.getTime(), xx, yy, zz, mTZOffset);
      }
      // clang-format off
      CADEBUG(printf("\tHit %3d/%3d Row %3d: Cluster Alpha %8.3f %3d, X %8.3f - Y %8.3f, Z %8.3f (Missed %d)\n", ihit, maxN, (int)clusters[ihit].row, clAlpha, (int)clusters[ihit].slice, xx, yy, zz, nMissed));
      // CADEBUG(if ((unsigned int)merger->GetTrackingChain()->mIOPtrs.nMCLabelsTPC > clusters[ihit].num))
      // CADEBUG({printf(" MC:"); for (int i = 0; i < 3; i++) {int mcId = merger->GetTrackingChain()->mIOPtrs.mcLabelsTPC[clusters[ihit].num].fClusterID[i].fMCID; if (mcId >= 0) printf(" %d", mcId); } } printf("\n"));
      // clang-format on
      if (MergeDoubleRowClusters(ihit, wayDirection, clusters, clustersXYZ, merger, prop, xx, yy, zz, maxN, clAlpha, clusterState, allowModification) == -1) {
        nMissed++;
        nMissed2++;
        continue;
      }

      bool changeDirection = (clusters[ihit].leg - lastLeg) & 1;
      // clang-format off
      CADEBUG(if (changeDirection) printf("\t\tChange direction\n"));
      CADEBUG(printf("\tLeg %3d Slice %2d %4sTrack   Alpha %8.3f %s, X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", (int)clusters[ihit].leg, (int)clusters[ihit].slice, "", prop.GetAlpha(), (CAMath::Abs(prop.GetAlpha() - clAlpha) < 0.01 ? "   " : " R!"), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
      // clang-format on
      if (allowModification && changeDirection && !noFollowCircle && !noFollowCircle2) {
        bool tryFollow = lastRow != 255;
        if (tryFollow) {
          const GPUTPCGMTrackParam backup = *this;
          const float backupAlpha = prop.GetAlpha();
          if (FollowCircle<0>(merger, prop, lastSlice, lastRow, iTrk, clAlpha, xx, yy, clusters[ihit].slice, clusters[ihit].row, inFlyDirection)) {
            CADEBUG(printf("Error during follow circle, resetting track!\n"));
            *this = backup;
            prop.SetTrack(this, backupAlpha);
            noFollowCircle = true;
            tryFollow = false;
          }
        }
        if (tryFollow) {
          MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].row, clusterState, false, clusters[ihit].slice > 18);
          lastUpdateX = mX;
          lastLeg = clusters[ihit].leg;
          lastSlice = clusters[ihit].slice;
          lastRow = 255;
          N++;
          resetT0 = initResetT0();
          // clang-format off
          CADEBUG(printf("\n"));
          CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
          // clang-format on
          continue;
        }
      } else if (allowModification && lastRow != 255 && CAMath::Abs(clusters[ihit].row - lastRow) > 1) {
        if (merger->Param().par.dodEdx && iWay == nWays - 1 && clusters[ihit].row == lastRow - 2 && clusters[ihit].leg == clusters[maxN - 1].leg) {
          dEdx.fillSubThreshold(lastRow - 1, param);
        }
        AttachClustersPropagate(merger, clusters[ihit].slice, lastRow, clusters[ihit].row, iTrk, clusters[ihit].leg == clusters[maxN - 1].leg, prop, inFlyDirection);
      }

      int err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
      // clang-format off
      CADEBUG(if (!CheckCov()){printf("INVALID COV AFTER PROPAGATE!!!\n");});
      // clang-format on
      if (err == -2) // Rotation failed, try to bring to new x with old alpha first, rotate, and then propagate to x, alpha
      {
        CADEBUG(printf("REROTATE\n"));
        if (prop.PropagateToXAlpha(xx, prop.GetAlpha(), inFlyDirection) == 0) {
          err = prop.PropagateToXAlpha(xx, clAlpha, inFlyDirection);
        }
      }
      if (lastRow == 255 || CAMath::Abs((int)lastRow - (int)clusters[ihit].row) > 5 || lastSlice != clusters[ihit].slice || (param.rec.tpc.trackFitRejectMode < 0 && -nMissed <= param.rec.tpc.trackFitRejectMode)) {
        goodRows = 0;
      } else {
        goodRows++;
      }
      if (err == 0) {
        lastRow = clusters[ihit].row;
        lastSlice = clusters[ihit].slice;
      }
      // clang-format off
      CADEBUG(printf("\t%21sPropaga Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Res %8.3f %8.3f   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[0] - yy, mP[1] - zz, sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], err));
      // clang-format on

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

      if (err == 0 && changeDirection) {
        const float mirrordY = prop.GetMirroredYTrack();
        CADEBUG(printf(" -- MirroredY: %f --> %f", mP[0], mirrordY));
        if (CAMath::Abs(yy - mP[0]) > CAMath::Abs(yy - mirrordY)) {
          CADEBUG(printf(" - Mirroring!!!"));
          if (allowModification) {
            AttachClustersMirror<0>(merger, clusters[ihit].slice, clusters[ihit].row, iTrk, yy, prop); // TODO: Never true, will always call FollowCircle above, really???
          }
          MirrorTo(prop, yy, zz, inFlyDirection, param, clusters[ihit].row, clusterState, true, clusters[ihit].slice > 18);
          noFollowCircle = false;

          lastUpdateX = mX;
          lastLeg = clusters[ihit].leg;
          lastRow = 255;
          N++;
          resetT0 = initResetT0();
          // clang-format off
          CADEBUG(printf("\n"));
          CADEBUG(printf("\t%21sMirror  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f) %28s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10]));
          // clang-format on
          continue;
        }
      }

      if (allowModification) {
        AttachClusters(merger, clusters[ihit].slice, clusters[ihit].row, iTrk, clusters[ihit].leg == clusters[maxN - 1].leg, prop);
      }

      const int err2 = mNDF > 0 && CAMath::Abs(prop.GetSinPhi0()) >= maxSinForUpdate;
      if (err || err2) {
        if (mC[0] > param.rec.tpc.trackFitCovLimit || mC[2] > param.rec.tpc.trackFitCovLimit) {
          break;
        }
        MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagNotFit);
        nMissed2++;
        NTolerated++;
        CADEBUG(printf(" --- break (%d, %d)\n", err, err2));
        continue;
      }
      CADEBUG(printf("\n"));

      int retVal;
      float threshold = 3.f + (lastUpdateX >= 0 ? (CAMath::Abs(mX - lastUpdateX) / 2) : 0.f);
      if (mNDF > 5 && (CAMath::Abs(yy - mP[0]) > threshold || CAMath::Abs(zz - mP[1]) > threshold)) {
        retVal = 2;
      } else {
        char rejectChi2 = attempt ? 0 : ((param.rec.tpc.mergerInterpolateErrors && CAMath::Abs(ihit - ihitMergeFirst) <= 1) ? (refit ? (2 + ((nWays - iWay) & 1)) : 0) : (allowModification && goodRows > 5));
#if EXTRACT_RESIDUALS == 1
        if (iWay == nWays - 1 && interpolation.hit[ihit].errorY > (GPUCA_MERGER_INTERPOLATION_ERROR_TYPE)0) {
          const float Iz0 = interpolation.hit[ihit].posY - mP[0];
          const float Iz1 = interpolation.hit[ihit].posZ - mP[1];
          float Iw0 = mC[2] + (float)interpolation.hit[ihit].errorZ;
          float Iw2 = mC[0] + (float)interpolation.hit[ihit].errorY;
          float Idet1 = 1.f / CAMath::Max(1e-10f, Iw0 * Iw2 - mC[1] * mC[1]);
          const float Ik00 = (mC[0] * Iw0 + mC[1] * mC[1]) * Idet1;
          const float Ik01 = (mC[0] * mC[1] + mC[1] * Iw2) * Idet1;
          const float Ik10 = (mC[1] * Iw0 + mC[2] * mC[1]) * Idet1;
          const float Ik11 = (mC[1] * mC[1] + mC[2] * Iw2) * Idet1;
          const float ImP0 = mP[0] + Ik00 * Iz0 + Ik01 * Iz1;
          const float ImP1 = mP[1] + Ik10 * Iz0 + Ik11 * Iz1;
          const float ImC0 = mC[0] - Ik00 * mC[0] + Ik01 * mC[1];
          const float ImC2 = mC[2] - Ik10 * mC[1] + Ik11 * mC[2];
          auto& tup = GPUROOTDump<TNtuple>::get("clusterres", "row:clX:clY:clZ:angle:trkX:trkY:trkZ:trkSinPhi:trkDzDs:trkQPt:trkSigmaY2:trkSigmaZ2trkSigmaQPt2");
          tup.Fill((float)clusters[ihit].row, xx, yy, zz, clAlpha, mX, ImP0, ImP1, mP[2], mP[3], mP[4], ImC0, ImC2, mC[14]);
        }
#endif
        retVal = prop.Update(yy, zz, clusters[ihit].row, param, clusterState, rejectChi2, &interpolation.hit[ihit], refit, clusters[ihit].slice > 18 GPUCA_DEBUG_STREAMER_CHECK(, iTrk));
        GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamUpdateTrack, iTrk)) {
          o2::utils::DebugStreamer::instance()->getStreamer("debug_update_track", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_update_track").data() << "iTrk=" << iTrk << "ihit=" << ihit << "yy=" << yy << "zz=" << zz << "cluster=" << clusters[ihit] << "track=" << this << "rejectChi2=" << rejectChi2 << "interpolationhit=" << interpolation.hit[ihit] << "refit=" << refit << "retVal=" << retVal << "\n";
        })
      }
      // clang-format off
      CADEBUG(if (!CheckCov()) GPUError("INVALID COV AFTER UPDATE!!!"));
      CADEBUG(printf("\t%21sFit     Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f), DzDs %5.2f %16s    ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f   -   YPt %8.3f   -   Err %d\n", "", prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), mP[3], "", sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14]), mC[10], retVal));
      // clang-format on

      ConstrainSinPhi();
      if (retVal == 0) // track is updated
      {
        if (storeOuter == 1 && clusters[ihit].leg == clusters[maxN - 1].leg) {
          StoreOuter(outerParam, prop, 2);
          storeOuter = 255;
        }
        noFollowCircle2 = false;
        lastUpdateX = mX;
        covYYUpd = mC[0];
        nMissed = nMissed2 = 0;
        UnmarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagNotFit);
        N++;
        ihitStart = ihit;
        float dy = mP[0] - prop.Model().Y();
        float dz = mP[1] - prop.Model().Z();
        if (CAMath::Abs(mP[4]) * merger->Param().qptB5Scaler > 10 && --resetT0 <= 0 && CAMath::Abs(mP[2]) < 0.15f && dy * dy + dz * dz > 1) {
          CADEBUG(printf("Reinit linearization\n"));
          prop.SetTrack(this, prop.GetAlpha());
        }
        if (merger->Param().par.dodEdx && iWay == nWays - 1 && clusters[ihit].leg == clusters[maxN - 1].leg) {
          float qtot, qmax, pad, relTime;
          if (merger->GetConstantMem()->ioPtrs.clustersNative == nullptr) {
            qtot = clustersXYZ[ihit].amp;
            qmax = 0;
            pad = 0;
            relTime = 0;
          } else {
            const ClusterNative& cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
            qtot = cl.qTot;
            qmax = cl.qMax;
            pad = cl.getPad();
            relTime = cl.getTime() - int(cl.getTime() + 0.5f);
          }
          dEdx.fillCluster(qtot, qmax, clusters[ihit].row, clusters[ihit].slice, mP[2], mP[3], param, merger->GetConstantMem()->calibObjects, zz, pad, relTime);
        }
      } else if (retVal == 2) { // cluster far away form the track
        if (allowModification) {
          MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectDistance);
        } else if (iWay == nWays - 1) {
          MarkClusters(clusters, ihitMergeFirst, ihit, wayDirection, GPUTPCGMMergedTrackHit::flagRejectErr);
        }
        nMissed++;
        nMissed2++;
      } else {
        break; // bad chi2 for the whole track, stop the fit
      }
    }
    if (((nWays - iWay) & 1) && (clusters[0].slice < 18) == (clusters[maxN - 1].slice < 18)) {
      ShiftZ2(clusters, clustersXYZ, merger, maxN);
    }
  }
  ConstrainSinPhi();

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamUpdateTrack, iTrk)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_accept_track", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("debug_accept_track").data() << "iTrk=" << iTrk << "outerParam=" << *outerParam << "track=" << this << "ihitStart=" << ihitStart << "\n";
  })

  if (!(N + NTolerated >= GPUCA_TRACKLET_SELECTOR_MIN_HITS_B5(mP[4] * merger->Param().qptB5Scaler) && 2 * NTolerated <= CAMath::Max(10, N) && CheckNumericalQuality(covYYUpd))) {
    return false; // TODO: NTolerated should never become that large, check what is going wrong!
  }
  if (merger->Param().rec.tpc.minNClustersFinalTrack != -1 && N + NTolerated < merger->Param().rec.tpc.minNClustersFinalTrack) {
    return false;
  }

  // TODO: we have looping tracks here with 0 accepted clusters in the primary leg. In that case we should refit the track using only the primary leg.

  if (merger->Param().par.dodEdx) {
    dEdx.computedEdx(merger->OutputTracksdEdx()[iTrk], param);
  }
  Alpha = prop.GetAlpha();
  MoveToReference(prop, param, Alpha);
  NormalizeAlpha(Alpha);

  return true;
}

GPUdni() void GPUTPCGMTrackParam::MoveToReference(GPUTPCGMPropagator& prop, const GPUParam& param, float& Alpha)
{
  if (param.rec.tpc.trackReferenceX <= 500) {
    GPUTPCGMTrackParam save = *this;
    float saveAlpha = Alpha;
    for (int attempt = 0; attempt < 3; attempt++) {
      float dAngle = floor(CAMath::ATan2(mP[0], mX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
      Alpha += dAngle;
      if (prop.PropagateToXAlpha(param.rec.tpc.trackReferenceX, Alpha, 0)) {
        break;
      }
      ConstrainSinPhi();
      if (CAMath::Abs(mP[0]) <= mX * CAMath::Tan(kSectAngle / 2.f)) {
        return;
      }
    }
    *this = save;
    Alpha = saveAlpha;
  }
  if (CAMath::Abs(mP[0]) > mX * CAMath::Tan(kSectAngle / 2.f)) {
    float dAngle = floor(CAMath::ATan2(mP[0], mX) / kDeg2Rad / 20.f + 0.5f) * kSectAngle;
    Rotate(dAngle);
    ConstrainSinPhi();
    Alpha += dAngle;
  }
}

GPUd() void GPUTPCGMTrackParam::MirrorTo(GPUTPCGMPropagator& GPUrestrict() prop, float toY, float toZ, bool inFlyDirection, const GPUParam& param, unsigned char row, unsigned char clusterState, bool mirrorParameters, bool cSide)
{
  if (mirrorParameters) {
    prop.Mirror(inFlyDirection);
  }
  float err2Y, err2Z;
  prop.GetErr2(err2Y, err2Z, param, toZ, row, clusterState, cSide);
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

GPUd() int GPUTPCGMTrackParam::MergeDoubleRowClusters(int& ihit, int wayDirection, GPUTPCGMMergedTrackHit* GPUrestrict() clusters, GPUTPCGMMergedTrackHitXYZ* clustersXYZ, const GPUTPCGMMerger* GPUrestrict() merger, GPUTPCGMPropagator& GPUrestrict() prop, float& GPUrestrict() xx, float& GPUrestrict() yy, float& GPUrestrict() zz, int maxN, float clAlpha, unsigned char& GPUrestrict() clusterState, bool rejectChi2)
{
  if (ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].slice == clusters[ihit + wayDirection].slice && clusters[ihit].leg == clusters[ihit + wayDirection].leg) {
    float maxDistY, maxDistZ;
    prop.GetErr2(maxDistY, maxDistZ, merger->Param(), zz, clusters[ihit].row, 0, clusters[ihit].slice > 18);
    maxDistY = (maxDistY + mC[0]) * 20.f;
    maxDistZ = (maxDistZ + mC[2]) * 20.f;
    int noReject = 0; // Cannot reject if simple estimation of y/z fails (extremely unlike case)
    if (CAMath::Abs(clAlpha - prop.GetAlpha()) > 1.e-4f) {
      noReject = prop.RotateToAlpha(clAlpha);
    }
    float projY = 0, projZ = 0;
    if (noReject == 0) {
      noReject |= prop.GetPropagatedYZ(xx, projY, projZ);
    }
    float count = 0.f;
    xx = yy = zz = 0.f;
    clusterState = 0;
    while (true) {
      float clx, cly, clz, clamp;
      if (merger->Param().par.earlyTpcTransform) {
        const float zOffset = (clusters[ihit].slice < 18) == (clusters[0].slice < 18) ? mTZOffset : -mTZOffset;
        clx = clustersXYZ[ihit].x;
        cly = clustersXYZ[ihit].y;
        clz = clustersXYZ[ihit].z - zOffset;
        clamp = clustersXYZ[ihit].amp;
      } else {
        const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[clusters[ihit].num];
        clamp = cl.qTot;
        merger->GetConstantMem()->calibObjects.fastTransformHelper->Transform(clusters[ihit].slice, clusters[ihit].row, cl.getPad(), cl.getTime(), clx, cly, clz, mTZOffset);
      }
      float dy = cly - projY;
      float dz = clz - projZ;
      if (noReject == 0 && (dy * dy > maxDistY || dz * dz > maxDistZ)) {
        CADEBUG(printf("\t\tRejecting double-row cluster: dy %f, dz %f, chiY %f, chiZ %f (Y: trk %f prj %f cl %f - Z: trk %f prj %f cl %f)\n", dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ), mP[0], projY, cly, mP[1], projZ, clz));
        if (rejectChi2) {
          clusters[ihit].state |= GPUTPCGMMergedTrackHit::flagRejectDistance;
        }
      } else {
        CADEBUG(printf("\t\tMerging hit row %d X %f Y %f Z %f (dy %f, dz %f, chiY %f, chiZ %f)\n", clusters[ihit].row, clx, cly, clz, dy, dz, sqrtf(maxDistY), sqrtf(maxDistZ)));
        xx += clx * clamp;
        yy += cly * clamp;
        zz += clz * clamp;
        clusterState |= clusters[ihit].state;
        count += clamp;
      }
      if (!(ihit + wayDirection >= 0 && ihit + wayDirection < maxN && clusters[ihit].row == clusters[ihit + wayDirection].row && clusters[ihit].slice == clusters[ihit + wayDirection].slice && clusters[ihit].leg == clusters[ihit + wayDirection].leg)) {
        break;
      }
      ihit += wayDirection;
    }
    if (count < 0.1f) {
      CADEBUG(printf("\t\tNo matching cluster in double-row, skipping\n"));
      return -1;
    }
    xx /= count;
    yy /= count;
    zz /= count;
  }
  return 0;
}

GPUd() void GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, bool goodLeg, GPUTPCGMPropagator& prop)
{
  float Y, Z;
  if (Merger->Param().par.earlyTpcTransform) {
    Y = mP[0];
    Z = mP[1];
  } else {
    float X = 0;
    Merger->GetConstantMem()->calibObjects.fastTransformHelper->InverseTransformYZtoX(slice, iRow, mP[0], mP[1], X);
    if (prop.GetPropagatedYZ(X, Y, Z)) {
      Y = mP[0];
      Z = mP[1];
    }
  }
  AttachClusters(Merger, slice, iRow, iTrack, goodLeg, Y, Z);
}

GPUd() void GPUTPCGMTrackParam::AttachClusters(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, bool goodLeg, float Y, float Z)
{
  if (Merger->Param().rec.tpc.disableRefitAttachment & 1) {
    return;
  }
  const GPUTPCTracker& GPUrestrict() tracker = *(Merger->GetConstantMem()->tpcTrackers + slice);
  const GPUTPCRow& GPUrestrict() row = tracker.Row(iRow);
#ifndef GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  GPUglobalref() const cahit2* hits = tracker.HitData(row);
  GPUglobalref() const calink* firsthit = tracker.FirstHitInBin(row);
#endif //! GPUCA_TEXTURE_FETCH_CONSTRUCTOR
  if (row.NHits() == 0) {
    return;
  }

  const float zOffset = Merger->Param().par.earlyTpcTransform ? ((Merger->OutputTracks()[iTrack].CSide() ^ (slice >= 18)) ? -mTZOffset : mTZOffset) : Merger->GetConstantMem()->calibObjects.fastTransformHelper->getCorrMap()->convVertexTimeToZOffset(slice, mTZOffset, Merger->Param().par.continuousMaxTimeBin);
  const float y0 = row.Grid().YMin();
  const float stepY = row.HstepY();
  const float z0 = row.Grid().ZMin() - zOffset; // We can use our own ZOffset, since this is only used temporarily anyway
  const float stepZ = row.HstepZ();
  int bin, ny, nz;

  float err2Y, err2Z;
  Merger->Param().GetClusterErrors2(iRow, Z, mP[2], mP[3], err2Y, err2Z);
  const float sy2 = CAMath::Min(Merger->Param().rec.tpc.tubeMaxSize2, Merger->Param().rec.tpc.tubeChi2 * (err2Y + CAMath::Abs(mC[0]))); // Cov can be bogus when following circle
  const float sz2 = CAMath::Min(Merger->Param().rec.tpc.tubeMaxSize2, Merger->Param().rec.tpc.tubeChi2 * (err2Z + CAMath::Abs(mC[2]))); // In that case we should provide the track error externally
  const float tubeY = CAMath::Sqrt(sy2);
  const float tubeZ = CAMath::Sqrt(sz2);
  const float sy21 = 1.f / sy2;
  const float sz21 = 1.f / sz2;
  float nY, nZ;
  if (Merger->Param().par.earlyTpcTransform) {
    nY = Y;
    nZ = Z;
  } else {
    Merger->GetConstantMem()->calibObjects.fastTransformHelper->InverseTransformYZtoNominalYZ(slice, iRow, Y, Z, nY, nZ);
  }

  if (CAMath::Abs(nY) > row.getTPCMaxY()) {
    return;
  }
  row.Grid().GetBinArea(nY, nZ + zOffset, tubeY, tubeZ, bin, ny, nz);

  const int nBinsY = row.Grid().Ny();
  const int idOffset = tracker.Data().ClusterIdOffset();
  const int* ids = &(tracker.Data().ClusterDataIndex()[row.HitNumberOffset()]);
  unsigned int myWeight = Merger->TrackOrderAttach()[iTrack] | gputpcgmmergertypes::attachAttached | gputpcgmmergertypes::attachTube;
  GPUAtomic(unsigned int)* const weights = Merger->ClusterAttachment();
  if (goodLeg) {
    myWeight |= gputpcgmmergertypes::attachGoodLeg;
  }
  for (int k = 0; k <= nz; k++) {
    const int mybin = bin + k * nBinsY;
    const unsigned int hitFst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin);
    const unsigned int hitLst = CA_TEXTURE_FETCH(calink, gAliTexRefu, firsthit, mybin + ny + 1);
    for (unsigned int ih = hitFst; ih < hitLst; ih++) {
      int id = idOffset + ids[ih];
      GPUAtomic(unsigned int)* const weight = weights + id;
#if !defined(GPUCA_NO_ATOMIC_PRECHECK) && GPUCA_NO_ATOMIC_PRECHECK < 1
      if (myWeight <= *weight) {
        continue;
      }
#endif
      const cahit2 hh = CA_TEXTURE_FETCH(cahit2, gAliTexRefu2, hits, ih);
      const float y = y0 + hh.x * stepY;
      const float z = z0 + hh.y * stepZ;
      const float dy = y - nY;
      const float dz = z - nZ;
      if (dy * dy * sy21 + dz * dz * sz21 <= CAMath::Sqrt(2.f)) {
        // CADEBUG(printf("Found Y %f Z %f\n", y, z));
        CAMath::AtomicMax(weight, myWeight);
      }
    }
  }
}

GPUd() void GPUTPCGMTrackParam::AttachClustersPropagate(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int lastRow, int toRow, int iTrack, bool goodLeg, GPUTPCGMPropagator& GPUrestrict() prop, bool inFlyDirection, float maxSinPhi)
{
  if (Merger->Param().rec.tpc.disableRefitAttachment & 2) {
    return;
  }
  if (CAMath::Abs(lastRow - toRow) < 2) {
    return;
  }
  int step = toRow > lastRow ? 1 : -1;
  float xx = mX - Merger->Param().tpcGeometry.Row2X(lastRow);
  for (int iRow = lastRow + step; iRow != toRow; iRow += step) {
    if (CAMath::Abs(mP[2]) > maxSinPhi) {
      return;
    }
    if (CAMath::Abs(mX) > CAMath::Abs(mP[0]) * CAMath::Tan(kSectAngle / 2.f)) {
      return;
    }
    int err = prop.PropagateToXAlpha(xx + Merger->Param().tpcGeometry.Row2X(iRow), prop.GetAlpha(), inFlyDirection);
    if (err) {
      return;
    }
    CADEBUG(printf("Attaching in row %d\n", iRow));
    AttachClusters(Merger, slice, iRow, iTrack, goodLeg, prop);
  }
}

GPUd() bool GPUTPCGMTrackParam::FollowCircleChk(float lrFactor, float toY, float toX, bool up, bool right)
{
  return CAMath::Abs(mX * lrFactor - toY) > 1.f &&                                                                       // transport further in Y
         CAMath::Abs(mP[2]) < 0.7f &&                                                                                    // rotate back
         (up ? (-mP[0] * lrFactor > toX || (right ^ (mP[2] > 0))) : (-mP[0] * lrFactor < toX || (right ^ (mP[2] < 0)))); // don't overshoot in X
}

GPUdii() void GPUTPCGMTrackParam::StoreOuter(gputpcgmmergertypes::GPUTPCOuterParam* outerParam, const GPUTPCGMPropagator& prop, int phase)
{
  CADEBUG(printf("\t%21sStorO%d  Alpha %8.3f    , X %8.3f - Y %8.3f, Z %8.3f   -   QPt %7.2f (%7.2f), SP %5.2f (%5.2f)   ---   Cov sY %8.3f sZ %8.3f sSP %8.3f sPt %8.3f\n", "", phase, prop.GetAlpha(), mX, mP[0], mP[1], mP[4], prop.GetQPt0(), mP[2], prop.GetSinPhi0(), sqrtf(mC[0]), sqrtf(mC[2]), sqrtf(mC[5]), sqrtf(mC[14])));
  for (int i = 0; i < 5; i++) {
    outerParam->P[i] = mP[i];
  }
  for (int i = 0; i < 15; i++) {
    outerParam->C[i] = mC[i];
  }
  outerParam->X = mX;
  outerParam->alpha = prop.GetAlpha();
}

GPUdic(0, 1) void GPUTPCGMTrackParam::StoreAttachMirror(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, float toAlpha, float toY, float toX, int toSlice, int toRow, bool inFlyDirection, float alpha)
{
  unsigned int nLoopData = CAMath::AtomicAdd(&Merger->Memory()->nLoopData, 1u);
  if (nLoopData >= Merger->NMaxTracks()) {
    Merger->raiseError(GPUErrors::ERROR_MERGER_LOOPER_OVERFLOW, nLoopData, Merger->NMaxTracks());
    CAMath::AtomicExch(&Merger->Memory()->nLoopData, Merger->NMaxTracks());
    return;
  }
  GPUTPCGMLoopData data;
  data.param = *this;
  data.alpha = alpha;
  data.track = iTrack;
  data.toAlpha = toAlpha;
  data.toY = toY;
  data.toX = toX;
  data.slice = slice;
  data.row = iRow;
  data.toSlice = toSlice;
  data.toRow = toRow;
  data.inFlyDirection = inFlyDirection;
  Merger->LoopData()[nLoopData] = data;
}

GPUdii() void GPUTPCGMTrackParam::RefitLoop(const GPUTPCGMMerger* GPUrestrict() Merger, int loopIdx)
{
  GPUTPCGMPropagator prop;
  prop.SetMaterialTPC();
  prop.SetPolynomialField(&Merger->Param().polynomialField);
  prop.SetMaxSinPhi(GPUCA_MAX_SIN_PHI);
  prop.SetToyMCEventsFlag(Merger->Param().par.toyMCEventsFlag);
  prop.SetMatLUT(Merger->Param().rec.useMatLUT ? Merger->GetConstantMem()->calibObjects.matLUT : nullptr);
  prop.SetSeedingErrors(false);
  prop.SetFitInProjections(true);
  prop.SetPropagateBzOnly(false);

  GPUTPCGMLoopData& data = Merger->LoopData()[loopIdx];
  prop.SetTrack(&data.param, data.alpha);
  if (data.toSlice == -1) {
    data.param.AttachClustersMirror<1>(Merger, data.slice, data.row, data.track, data.toY, prop, true);
  } else {
    data.param.FollowCircle<1>(Merger, prop, data.slice, data.row, data.track, data.toAlpha, data.toX, data.toY, data.toSlice, data.toRow, data.inFlyDirection, true);
  }
}

template <int I>
GPUdic(0, 1) int GPUTPCGMTrackParam::FollowCircle(const GPUTPCGMMerger* GPUrestrict() Merger, GPUTPCGMPropagator& GPUrestrict() prop, int slice, int iRow, int iTrack, float toAlpha, float toX, float toY, int toSlice, int toRow, bool inFlyDirection, bool phase2)
{
  if (Merger->Param().rec.tpc.disableRefitAttachment & 4) {
    return 1;
  }
  if (Merger->Param().rec.tpc.loopInterpolationInExtraPass && phase2 == false) {
    StoreAttachMirror(Merger, slice, iRow, iTrack, toAlpha, toY, toX, toSlice, toRow, inFlyDirection, prop.GetAlpha());
    return 1;
  }
  const GPUParam& GPUrestrict() param = Merger->Param();
  bool right;
  float dAlpha = toAlpha - prop.GetAlpha();
  int sliceSide = slice >= (GPUCA_NSLICES / 2) ? (GPUCA_NSLICES / 2) : 0;
  if (CAMath::Abs(dAlpha) > 0.001f) {
    right = CAMath::Abs(dAlpha) < CAMath::Pi() ? (dAlpha > 0) : (dAlpha < 0);
  } else {
    right = toY > mP[0];
  }
  bool up = (mP[2] < 0) ^ right;
  int targetRow = up ? (GPUCA_ROW_COUNT - 1) : 0;
  float lrFactor = mP[2] < 0 ? -1.f : 1.f; // !(right ^ down) // TODO: shouldn't it be "right ? 1.f : -1.f", but that gives worse results...
  // clang-format off
  CADEBUG(printf("CIRCLE Track %d: Slice %d Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f - Next hit: Slice %d Alpha %f X %f Y %f - Right %d Up %d dAlpha %f lrFactor %f\n", iTrack, slice, prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3], toSlice, toAlpha, toX, toY, (int)right, (int)up, dAlpha, lrFactor));
  // clang-format on

  AttachClustersPropagate(Merger, slice, iRow, targetRow, iTrack, false, prop, inFlyDirection, 0.7f);
  if (prop.RotateToAlpha(prop.GetAlpha() + (CAMath::Pi() / 2.f) * lrFactor)) {
    return 1;
  }
  CADEBUG(printf("\tRotated: X %f Y %f Z %f SinPhi %f (Alpha %f / %f)\n", mP[0], mX, mP[1], mP[2], prop.GetAlpha(), prop.GetAlpha() + CAMath::Pi() / 2.f));
  while (slice != toSlice || FollowCircleChk(lrFactor, toY, toX, up, right)) {
    while ((slice != toSlice) ? (CAMath::Abs(mX) <= CAMath::Abs(mP[0]) * CAMath::Tan(kSectAngle / 2.f)) : FollowCircleChk(lrFactor, toY, toX, up, right)) {
      int err = prop.PropagateToXAlpha(mX + 1.f, prop.GetAlpha(), inFlyDirection);
      if (err) {
        CADEBUG(printf("\t\tpropagation error (%d)\n", err));
        prop.RotateToAlpha(prop.GetAlpha() - (CAMath::Pi() / 2.f) * lrFactor);
        return 1;
      }
      CADEBUG(printf("\tPropagated to y = %f: X %f Z %f SinPhi %f\n", mX, mP[0], mP[1], mP[2]));
      for (int j = 0; j < GPUCA_ROW_COUNT; j++) {
        float rowX = Merger->Param().tpcGeometry.Row2X(j);
        if (CAMath::Abs(rowX - (-mP[0] * lrFactor)) < 1.5f) {
          CADEBUG(printf("\t\tAttempt row %d (Y %f Z %f)\n", j, mX * lrFactor, mP[1]));
          AttachClusters(Merger, slice, j, iTrack, false, mX * lrFactor, mP[1]);
        }
      }
    }
    if (slice != toSlice) {
      if (right) {
        if (++slice >= sliceSide + 18) {
          slice -= 18;
        }
      } else {
        if (--slice < sliceSide) {
          slice += 18;
        }
      }
      CADEBUG(printf("\tRotating to slice %d\n", slice));
      if (prop.RotateToAlpha(param.Alpha(slice) + (CAMath::Pi() / 2.f) * lrFactor)) {
        CADEBUG(printf("\t\trotation error\n"));
        prop.RotateToAlpha(prop.GetAlpha() - (CAMath::Pi() / 2.f) * lrFactor);
        return 1;
      }
      CADEBUG(printf("\tAfter Rotatin Alpha %f Position X %f Y %f Z %f SinPhi %f\n", prop.GetAlpha(), mP[0], mX, mP[1], mP[2]));
    }
  }
  CADEBUG(printf("\tRotating back\n"));
  for (int i = 0; i < 2; i++) {
    if (prop.RotateToAlpha(prop.GetAlpha() + (CAMath::Pi() / 2.f) * lrFactor) == 0) {
      break;
    }
    if (i) {
      CADEBUG(printf("Final rotation failed\n"));
      return 1;
    }
    CADEBUG(printf("\tresetting physical model\n"));
    prop.SetTrack(this, prop.GetAlpha());
  }
  prop.Rotate180();
  CADEBUG(printf("\tMirrored position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3]));
  iRow = toRow;
  float dx = toX - Merger->Param().tpcGeometry.Row2X(toRow);
  if (up ^ (toX > mX)) {
    if (up) {
      while (iRow < GPUCA_ROW_COUNT - 2 && Merger->Param().tpcGeometry.Row2X(iRow + 1) + dx <= mX) {
        iRow++;
      }
    } else {
      while (iRow > 1 && Merger->Param().tpcGeometry.Row2X(iRow - 1) + dx >= mX) {
        iRow--;
      }
    }
    prop.PropagateToXAlpha(Merger->Param().tpcGeometry.Row2X(iRow) + dx, prop.GetAlpha(), inFlyDirection);
    AttachClustersPropagate(Merger, slice, iRow, toRow, iTrack, false, prop, inFlyDirection);
  }
  if (prop.PropagateToXAlpha(toX, prop.GetAlpha(), inFlyDirection)) {
    mX = toX;
  }
  CADEBUG(printf("Final position: Alpha %f X %f Y %f Z %f SinPhi %f DzDs %f\n", prop.GetAlpha(), mX, mP[0], mP[1], mP[2], mP[3]));
  return (0);
}

template <int I>
GPUdni() void GPUTPCGMTrackParam::AttachClustersMirror(const GPUTPCGMMerger* GPUrestrict() Merger, int slice, int iRow, int iTrack, float toY, GPUTPCGMPropagator& GPUrestrict() prop, bool phase2)
{
  if (Merger->Param().rec.tpc.disableRefitAttachment & 8) {
    return;
  }
  if (Merger->Param().rec.tpc.loopInterpolationInExtraPass && phase2 == false) {
    StoreAttachMirror(Merger, slice, iRow, iTrack, 0, toY, 0, -1, 0, 0, prop.GetAlpha());
    return;
  }
  float X = mP[2] > 0 ? mP[0] : -mP[0];
  float toX = mP[2] > 0 ? toY : -toY;
  float Y = mP[2] > 0 ? -mX : mX;
  float Z = mP[1];
  if (CAMath::Abs(mP[2]) >= GPUCA_MAX_SIN_PHI_LOW) {
    return;
  }
  float SinPhi = CAMath::Sqrt(1 - mP[2] * mP[2]) * (mP[2] > 0 ? -1 : 1);
  if (CAMath::Abs(SinPhi) >= GPUCA_MAX_SIN_PHI_LOW) {
    return;
  }
  float b = prop.GetBz(prop.GetAlpha(), mX, mP[0], mP[1]);

  int count = CAMath::Abs((toX - X) / 0.5f) + 0.5f;
  if (count == 0) {
    return;
  }
  float dx = (toX - X) / count;
  const float myRowX = Merger->Param().tpcGeometry.Row2X(iRow);
  // printf("AttachMirror\n");
  // printf("X %f Y %f Z %f SinPhi %f toY %f -->\n", mX, mP[0], mP[1], mP[2], toY);
  // printf("X %f Y %f Z %f SinPhi %f, count %d dx %f (to: %f)\n", X, Y, Z, SinPhi, count, dx, X + count * dx);
  while (count--) {
    float ex = CAMath::Sqrt(1 - SinPhi * SinPhi);
    float exi = 1.f / ex;
    float dxBzQ = dx * -b * mP[4];
    float newSinPhi = SinPhi + dxBzQ;
    if (CAMath::Abs(newSinPhi) > GPUCA_MAX_SIN_PHI_LOW) {
      return;
    }
    float dS = dx * exi;
    float h2 = dS * exi * exi;
    float h4 = .5f * h2 * dxBzQ;

    X += dx;
    Y += dS * SinPhi + h4;
    Z += dS * mP[3];
    SinPhi = newSinPhi;
    if (CAMath::Abs(X) > CAMath::Abs(Y) * CAMath::Tan(kSectAngle / 2.f)) {
      continue;
    }

    // printf("count %d: At X %f Y %f Z %f SinPhi %f\n", count, mP[2] > 0 ? -Y : Y, mP[2] > 0 ? X : -X, Z, SinPhi);

    float paramX = mP[2] > 0 ? -Y : Y;
    int step = paramX >= mX ? 1 : -1;
    int found = 0;
    for (int j = iRow; j >= 0 && j < GPUCA_ROW_COUNT && found < 3; j += step) {
      float rowX = mX + Merger->Param().tpcGeometry.Row2X(j) - myRowX;
      if (CAMath::Abs(rowX - paramX) < 1.5f) {
        // printf("Attempt row %d\n", j);
        AttachClusters(Merger, slice, j, iTrack, false, mP[2] > 0 ? X : -X, Z);
      }
    }
  }
}

GPUd() void GPUTPCGMTrackParam::ShiftZ2(const GPUTPCGMMergedTrackHit* clusters, GPUTPCGMMergedTrackHitXYZ* clustersXYZ, const GPUTPCGMMerger* merger, int N)
{
  float tzInner, tzOuter;
  float xInner, xOuter;
  if (N == 0) {
    N = 1;
  }
  if (merger->Param().par.earlyTpcTransform) {
    tzInner = clustersXYZ[N - 1].z;
    tzOuter = clustersXYZ[0].z;
    xInner = clustersXYZ[N - 1].x;
    xOuter = clustersXYZ[0].x;
  } else {
    const auto& GPUrestrict() cls = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear;
    tzInner = cls[clusters[N - 1].num].getTime();
    tzOuter = cls[clusters[0].num].getTime();
    xInner = merger->Param().tpcGeometry.Row2X(clusters[N - 1].row);
    xOuter = merger->Param().tpcGeometry.Row2X(clusters[0].row);
  }
  ShiftZ(merger, clusters[0].slice, tzInner, tzOuter, xInner, xOuter);
}

GPUd() void GPUTPCGMTrackParam::ShiftZ(const GPUTPCGMMerger* GPUrestrict() merger, int slice, float tz1, float tz2, float x1, float x2)
{
  if (!merger->Param().par.continuousTracking) {
    return;
  }
  const float r1 = CAMath::Max(0.0001f, CAMath::Abs(mP[4] * merger->Param().polynomialField.GetNominalBz()));

  const float dist2 = mX * mX + mP[0] * mP[0];
  const float dist1r2 = dist2 * r1 * r1;
  float deltaZ = 0.f;
  bool beamlineReached = false;
  if (dist1r2 < 4) {
    const float alpha = CAMath::ACos(1 - 0.5f * dist1r2); // Angle of a circle, such that |(cosa, sina) - (1,0)| == dist
    const float beta = CAMath::ATan2(mP[0], mX);
    const int comp = mP[2] > CAMath::Sin(beta);
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

  if (!beamlineReached) {
    if (merger->Param().par.earlyTpcTransform) {
      float basez, basex;
      if (CAMath::Abs(tz1) < CAMath::Abs(tz2)) {
        basez = tz1;
        basex = x1;
      } else {
        basez = tz2;
        basex = x2;
      }
      float refZ = ((basez > 0) ? merger->Param().rec.tpc.defaultZOffsetOverR : -merger->Param().rec.tpc.defaultZOffsetOverR) * basex;
      deltaZ = basez - refZ - mTZOffset;
    } else {
      float baset, basex;
      if (CAMath::Abs(tz1) > CAMath::Abs(tz2)) {
        baset = tz1;
        basex = x1;
      } else {
        baset = tz2;
        basex = x2;
      }
      float refZ = ((slice < GPUCA_NSLICES / 2) ? merger->Param().rec.tpc.defaultZOffsetOverR : -merger->Param().rec.tpc.defaultZOffsetOverR) * basex;
      float basez;
      merger->GetConstantMem()->calibObjects.fastTransformHelper->getCorrMap()->TransformIdealZ(slice, baset, basez, mTZOffset);
      deltaZ = basez - refZ;
    }
  }
  if (merger->Param().par.earlyTpcTransform) {
    mTZOffset += deltaZ;
    mP[1] -= deltaZ;
    deltaZ = 0;
    float zMax = CAMath::Max(tz1, tz2);
    float zMin = CAMath::Min(tz1, tz2);
    // printf("Z Check: Clusters %f %f, min %f max %f vtx %f\n", tz1, tz2, zMin, zMax, mTZOffset);
    if (zMin < 0 && zMin - mTZOffset < -GPUTPCGeometry::TPCLength()) {
      deltaZ = zMin - mTZOffset + GPUTPCGeometry::TPCLength();
    } else if (zMax > 0 && zMax - mTZOffset > GPUTPCGeometry::TPCLength()) {
      deltaZ = zMax - mTZOffset - GPUTPCGeometry::TPCLength();
    }
    if (zMin < 0 && zMax - (mTZOffset + deltaZ) > 0) {
      deltaZ = zMax - mTZOffset;
    } else if (zMax > 0 && zMin - (mTZOffset + deltaZ) < 0) {
      deltaZ = zMin - mTZOffset;
    }
    // if (deltaZ != 0) printf("Moving clusters to TPC Range: Shift %f in Z: %f to %f --> %f to %f in Z\n", deltaZ, tz2 - mTZOffset, tz1 - mTZOffset, tz2 - mTZOffset - deltaZ, tz1 - mTZOffset - deltaZ);
    mTZOffset += deltaZ;
    mP[1] -= deltaZ;
  } else {
    float deltaT = merger->GetConstantMem()->calibObjects.fastTransformHelper->getCorrMap()->convDeltaZtoDeltaTimeInTimeFrame(slice, deltaZ);
    mTZOffset += deltaT;
    mP[1] -= deltaZ;
    const float maxT = CAMath::Min(tz1, tz2);
    const float minT = CAMath::Max(tz1, tz2) - merger->GetConstantMem()->calibObjects.fastTransformHelper->getCorrMap()->getMaxDriftTime(slice);
    // printf("T Check: Clusters %f %f, min %f max %f vtx %f\n", tz1, tz2, minT, maxT, mTZOffset);
    deltaT = 0.f;
    if (mTZOffset < minT) {
      deltaT = minT - mTZOffset;
    }
    if (mTZOffset + deltaT > maxT) {
      deltaT = maxT - mTZOffset;
    }
    if (deltaT != 0.f) {
      deltaZ = merger->GetConstantMem()->calibObjects.fastTransformHelper->getCorrMap()->convDeltaTimeToDeltaZinTimeFrame(slice, deltaT);
      // printf("Moving clusters to TPC Range: QPt %f, New mTZOffset %f, t1 %f, t2 %f, Shift %f in Z: %f to %f --> %f to %f in T\n", mP[4], mTZOffset + deltaT, tz1, tz2, deltaZ, tz2 - mTZOffset, tz1 - mTZOffset, tz2 - mTZOffset - deltaT, tz1 - mTZOffset - deltaT);
      mTZOffset += deltaT;
      mP[1] -= deltaZ;
    }
  }
  // printf("\n");
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
  CADEBUG(
    printf("OK %d - %f - ", (int)ok, mX); for (int i = 0; i < 5; i++) { printf("%f ", mP[i]); } printf(" - "); for (int i = 0; i < 15; i++) { printf("%f ", mC[i]); } printf("\n"));
  const float* c = mC;
  for (int i = 0; i < 15; i++) {
    ok = ok && CAMath::Finite(c[i]);
  }
  CADEBUG(printf("OK1 %d\n", (int)ok));
  for (int i = 0; i < 5; i++) {
    ok = ok && CAMath::Finite(mP[i]);
  }
  CADEBUG(printf("OK2 %d\n", (int)ok));
  if ((overrideCovYY > 0 ? overrideCovYY : c[0]) > 4.f * 4.f || c[2] > 4.f * 4.f || c[5] > 2.f * 2.f || c[9] > 2.f * 2.f) {
    ok = 0;
  }
  CADEBUG(printf("OK3 %d\n", (int)ok));
  if (CAMath::Abs(mP[2]) > GPUCA_MAX_SIN_PHI) {
    ok = 0;
  }
  CADEBUG(printf("OK4 %d\n", (int)ok));
  if (!CheckCov()) {
    ok = false;
  }
  CADEBUG(printf("OK5 %d\n", (int)ok));
  return ok;
}

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)
bool GPUTPCGMTrackParam::GetExtParam(AliExternalTrackParam& T, double alpha) const
{
  //* Convert from GPUTPCGMTrackParam to AliExternalTrackParam parameterisation,
  //* the angle alpha is the global angle of the local X axis

  bool ok = CheckNumericalQuality();

  double par[5], cov[15];
  for (int i = 0; i < 5; i++) {
    par[i] = mP[i];
  }
  for (int i = 0; i < 15; i++) {
    cov[i] = mC[i];
  }

  if (par[2] > GPUCA_MAX_SIN_PHI) {
    par[2] = GPUCA_MAX_SIN_PHI;
  }
  if (par[2] < -GPUCA_MAX_SIN_PHI) {
    par[2] = -GPUCA_MAX_SIN_PHI;
  }

  if (CAMath::Abs(par[4]) < 1.e-5) {
    par[4] = 1.e-5; // some other software will crash if q/Pt==0
  }
  if (CAMath::Abs(par[4]) > 1. / 0.08) {
    ok = 0; // some other software will crash if q/Pt is too big
  }
  T.Set((double)mX, alpha, par, cov);
  return ok;
}

void GPUTPCGMTrackParam::SetExtParam(const AliExternalTrackParam& T)
{
  //* Convert from AliExternalTrackParam parameterisation

  for (int i = 0; i < 5; i++) {
    mP[i] = T.GetParameter()[i];
  }
  for (int i = 0; i < 15; i++) {
    mC[i] = T.GetCovariance()[i];
  }
  mX = T.GetX();
  if (mP[2] > GPUCA_MAX_SIN_PHI) {
    mP[2] = GPUCA_MAX_SIN_PHI;
  }
  if (mP[2] < -GPUCA_MAX_SIN_PHI) {
    mP[2] = -GPUCA_MAX_SIN_PHI;
  }
}
#endif

GPUd() void GPUTPCGMTrackParam::RefitTrack(GPUTPCGMMergedTrack& GPUrestrict() track, int iTrk, GPUTPCGMMerger* GPUrestrict() merger, int attempt) // TODO: Inline me, once __forceinline__ is fixed by HIP
{
  if (!track.OK()) {
    return;
  }

  // clang-format off
  CADEBUG(if (DEBUG_SINGLE_TRACK >= 0 && iTrk != DEBUG_SINGLE_TRACK) { track.SetNClusters(0); track.SetOK(0); return; } );
  // clang-format on

  int nTrackHits = track.NClusters();
  int NTolerated = 0; // Clusters not fit but tollerated for track length cut
  GPUTPCGMTrackParam t = track.Param();
  float Alpha = track.Alpha();
  CADEBUG(int nTrackHitsOld = nTrackHits; float ptOld = t.QPt());
  bool ok = t.Fit(merger, iTrk, merger->Clusters() + track.FirstClusterRef(), merger->ClustersXYZ() + track.FirstClusterRef(), nTrackHits, NTolerated, Alpha, attempt, GPUCA_MAX_SIN_PHI, &track.OuterParam());
  CADEBUG(printf("Finished Fit Track %d\n", iTrk));
  CADEBUG(printf("OUTPUT hits %d -> %d+%d = %d, QPt %f -> %f, SP %f, ok %d chi2 %f chi2ndf %f\n", nTrackHitsOld, nTrackHits, NTolerated, nTrackHits + NTolerated, ptOld, t.QPt(), t.SinPhi(), (int)ok, t.Chi2(), t.Chi2() / CAMath::Max(1, nTrackHits)));

  if (!ok && attempt == 0 && merger->Param().rec.tpc.retryRefit) {
    for (unsigned int i = 0; i < track.NClusters(); i++) {
      merger->Clusters()[track.FirstClusterRef() + i].state &= GPUTPCGMMergedTrackHit::clustererAndSharedFlags;
    }
    CADEBUG(printf("Track rejected, marking for retry\n"));
    if (merger->Param().rec.tpc.retryRefit == 2) {
      nTrackHits = track.NClusters();
      NTolerated = 0; // Clusters not fit but tollerated for track length cut
      t = track.Param();
      Alpha = track.Alpha();
      ok = t.Fit(merger, iTrk, merger->Clusters() + track.FirstClusterRef(), merger->ClustersXYZ() + track.FirstClusterRef(), nTrackHits, NTolerated, Alpha, 1, GPUCA_MAX_SIN_PHI, &track.OuterParam());
    } else {
      unsigned int nRefit = CAMath::AtomicAdd(&merger->Memory()->nRetryRefit, 1u);
      merger->RetryRefitIds()[nRefit] = iTrk;
      return;
    }
  }
  if (CAMath::Abs(t.QPt()) < 1.e-4f) {
    t.QPt() = 1.e-4f;
  }

  CADEBUG(if (t.GetX() > 250) { printf("ERROR, Track %d at impossible X %f, Pt %f, Looper %d\n", iTrk, t.GetX(), CAMath::Abs(1.f / t.QPt()), (int)merger->OutputTracks()[iTrk].Looper()); });

  track.SetOK(ok);
  track.SetNClustersFitted(nTrackHits);
  track.Param() = t;
  track.Alpha() = Alpha;

  if (track.OK()) {
    int ind = track.FirstClusterRef();
    const GPUParam& GPUrestrict() param = merger->Param();
    float alphaa = param.Alpha(merger->Clusters()[ind].slice);
    float xx, yy, zz;
    if (merger->Param().par.earlyTpcTransform) {
      xx = merger->ClustersXYZ()[ind].x;
      yy = merger->ClustersXYZ()[ind].y;
      zz = merger->ClustersXYZ()[ind].z - track.Param().GetTZOffset();
    } else {
      const ClusterNative& GPUrestrict() cl = merger->GetConstantMem()->ioPtrs.clustersNative->clustersLinear[merger->Clusters()[ind].num];
      merger->GetConstantMem()->calibObjects.fastTransformHelper->Transform(merger->Clusters()[ind].slice, merger->Clusters()[ind].row, cl.getPad(), cl.getTime(), xx, yy, zz, track.Param().GetTZOffset());
    }
    float sinA, cosA;
    CAMath::SinCos(alphaa - track.Alpha(), sinA, cosA);
    track.SetLastX(xx * cosA - yy * sinA);
    track.SetLastY(xx * sinA + yy * cosA);
    track.SetLastZ(zz);
  }
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
  mP[2] = sinPhi + j2;
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
  const int diagMap[5] = {0, 2, 5, 9, 14};
  const float oldDiag[5] = {mC[0], mC[2], mC[5], mC[9], mC[14]};
  for (int i = 0; i < 5; i++) {
    mC[diagMap[i]] += errors2[i];
    for (int j = 0; j < i; j++) {
      mC[diagMap[i - 1] + j + 1] *= gpu::CAMath::Sqrt(mC[diagMap[i]] * mC[diagMap[j]] / (oldDiag[i] * oldDiag[j]));
    }
  }
}
