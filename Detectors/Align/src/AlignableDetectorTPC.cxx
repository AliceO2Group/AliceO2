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

/// @file   AlignableDetectorTPC.h
/// @author ruben.shahoyan@cern.ch
/// @brief  TPC detector wrapper

#include "Align/AlignableDetectorTPC.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorTPC.h"
#include "Align/Controller.h"
#include "Align/AlignConfig.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include <TMath.h>
#include <TGeoManager.h>
#include "GPUParam.inc"

namespace o2
{
namespace align
{
using namespace TMath;

//____________________________________________
AlignableDetectorTPC::AlignableDetectorTPC(Controller* ctr) : AlignableDetector(DetID::TPC, ctr)
{
  // default c-tor
}

//____________________________________________
void AlignableDetectorTPC::defineVolumes()
{
  // define fictious TPC envelope and sector volumes
  AlignableVolume* volTPC = nullptr;
  int labDet = getDetLabel();
  const int NSectors = o2::tpc::constants::MAXSECTOR / 2;

  addVolume(volTPC = new AlignableVolume("TPC_envelope", getDetLabel(), mController));
  volTPC->setDummyEnvelope();

  for (int isec = 0; isec < o2::tpc::constants::MAXSECTOR; isec++) {
    int isecSide = isec % NSectors;
    const char* symname = Form("TPC/sec%02d", isec);
    AlignableSensorTPC* sector = new AlignableSensorTPC(symname, o2::base::GeometryManager::getSensID(mDetID, isec), getSensLabel(isec), isec, mController);
    sector->setParent(volTPC);
    addVolume(sector);
  }
}

//____________________________________________
void AlignableDetectorTPC::Print(const Option_t* opt) const
{
  // print info
  AlignableDetector::Print(opt);
}

//____________________________________________
int AlignableDetectorTPC::processPoints(GIndex gid, int npntCut, bool inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  //
  const auto& algConf = AlignConfig::Instance();
  const auto recoData = mController->getRecoContainer();
  const auto& trk = recoData->getTrack<o2::tpc::TrackTPC>(gid);
  gsl::span<const unsigned char> TPCShMap = recoData->clusterShMapTPC;

  int nClus = trk.getNClusters();
  if (nClus < npntCut) {
    return -1;
  }
  int npointsIni = mNPoints;
  auto prop = o2::base::Propagator::Instance(); // float version!
  constexpr float TAN10 = 0.17632698;
  const auto clusterIdxStruct = recoData->getTPCTracksClusterRefs();
  const auto clusterNativeAccess = recoData->inputsTPCclusters->clusterIndex;
  bool fail = false;
  auto algTrack = mController->getAlgTrack();

  o2::track::TrackParCov trkParam = inv ? trk : trk.getOuterParam(); // we refit outer param inward
  trkParam.resetCovariance();
  float bzkG = prop->getNominalBz(), qptB5Scale = std::abs(bzkG) > 0.1 ? std::abs(bzkG) / 5.006680f : 1.f;
  float q2pt2 = trkParam.getQ2Pt() * trkParam.getQ2Pt(), q2pt2Wgh = q2pt2 * qptB5Scale * qptB5Scale;
  float err2 = (100.f + q2pt2Wgh) / (1.f + q2pt2Wgh) * q2pt2; // -> 100 for high pTs, -> 1 for low pTs.
  trkParam.setCov(err2, 14);                                  // 100% error

  int direction = inv ? -1 : 1;
  int start = inv ? nClus - 1 : 0;
  int stop = inv ? -1 : nClus;
  const o2::tpc::ClusterNative* cl = nullptr;
  uint8_t sector, row = 0, currentSector = 0, currentRow = 0;
  short clusterState = 0, nextState;
  float tOffset = mTrackTimeStamp / (o2::constants::lhc::LHCBunchSpacingMUS * 8);
  bool stopLoop = false;
  int npoints = 0;
  for (int i = start; i != stop; i += cl ? 0 : direction) {
    float x, y, z, xTmp, yTmp, zTmp, charge = 0.f;
    int clusters = 0;
    double combRow = 0;

    while (true) {
      if (!cl) {
        auto clTmp = &trk.getCluster(clusterIdxStruct, i, clusterNativeAccess, sector, row);
        if (row < algConf.minTPCPadRow) {
          if (!inv) { // inward refit: all other clusters padrow will be <= minumum (with outward refit the following clusters have a chance to be accepted)
            stopLoop = true;
          }
          break;
        } else if (row > algConf.maxTPCPadRow) {
          if (inv) { // outward refit: all other clusters padrow will be >= maximum (with inward refit the following clusters have a chance to be accepted)
            stopLoop = true;
          }
          break;
        }
        if (algConf.discardEdgePadrows > 0 && getDistanceToStackEdge(row) < algConf.discardEdgePadrows) {
          if (i + direction != stop) {
            i += direction;
            continue;
          } else {
            stopLoop = true;
            break;
          }
        }
        mController->getTPCCorrMaps()->Transform(sector, row, cl->getPad(), cl->getTime(), xTmp, yTmp, zTmp, tOffset);
        if (algConf.discardSectorEdgeDepth > 0) {
          if (std::abs(yTmp) + algConf.discardSectorEdgeDepth > xTmp * TAN10) {
            if (i + direction != stop) {
              i += direction;
              continue;
            } else {
              stopLoop = true;
              break;
            }
          }
        }

        cl = clTmp;
        nextState = TPCShMap[cl - clusterNativeAccess.clustersLinear];
      }
      if (clusters == 0 || (sector == currentSector && std::abs(row - currentRow) < algConf.maxTPCRowsCombined)) {
        if (clusters == 1) {
          x *= charge;
          y *= charge;
          z *= charge;
          combRow *= charge;
        }
        if (clusters == 0) {
          x = xTmp;
          y = yTmp;
          z = zTmp;
          // mController->getTPCCorrMaps()->Transform(sector, row, cl->getPad(), cl->getTime(), x, y, z, tOffset);
          currentRow = row;
          currentSector = sector;
          charge = cl->qTot;
          clusterState = nextState;
          combRow = row;
          LOGP(debug, "starting a supercluster at row {} of sector {} -> {},{},{}", currentRow, currentSector, x, y, z);
        } else {
          // float xx, yy, zz;
          // mController->getTPCCorrMaps()->Transform(sector, row, cl->getPad(), cl->getTime(), xx, yy, zz, tOffset);
          x += xTmp * cl->qTot;
          y += yTmp * cl->qTot;
          z += zTmp * cl->qTot;
          combRow += row * cl->qTot;
          charge += cl->qTot;
          clusterState |= nextState;
          npntCut--;
          LOGP(debug, "merging cluster #{} at row {} to a supercluster starting at row {} ", clusters + 1, row, currentRow);
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
    if (stopLoop) {
      break;
    }
    if (clusters == 0) {
      continue;
    } else if (clusters > 1) {
      x /= charge;
      y /= charge;
      z /= charge;
      currentRow = combRow / charge;
      LOGP(debug, "Combined cluster of {} subclusters: row {} , {},{},{}", clusters, currentRow, x, y, z);
    }

    if (!trkParam.rotate(math_utils::detail::sector2Angle<float>(currentSector)) || !prop->PropagateToXBxByBz(trkParam, x, algConf.maxSnp)) {
      break;
    }
    if (!npoints) {
      trkParam.setZ(z);
    }

    auto* sectSensor = (AlignableSensorTPC*)getSensor(currentSector);
    const auto* sysE = sectSensor->getAddError(); // additional syst error

    gpu::gpustd::array<float, 2> p = {y, z};
    gpu::gpustd::array<float, 3> c = {0, 0, 0};
    mController->getTPCParam()->GetClusterErrors2(currentRow, z, trkParam.getSnp(), trkParam.getTgl(), c[0], c[2]);
    if (sysE[0] > 0.f) {
      c[0] += sysE[0] * sysE[0];
    }
    if (sysE[1] > 0.f) {
      c[2] += sysE[1] * sysE[1];
    }

    mController->getTPCParam()->UpdateClusterError2ByState(clusterState, c[0], c[2]);
    if (!trkParam.update(p, c)) {
      break;
    }

    auto& pnt = algTrack->addDetectorPoint();
    pnt.setYZErrTracking(c[0], c[1], c[2]);
    if (getUseErrorParam()) { // errors will be calculated just before using the point in the fit, using track info
      pnt.setNeedUpdateFromTrack();
    }
    pnt.setXYZTracking(x, y, z);
    pnt.setSensor(sectSensor);
    pnt.setAlphaSens(sectSensor->getAlpTracking());
    pnt.setXSens(sectSensor->getXTracking());
    pnt.setDetID(mDetID);
    pnt.setSID(sectSensor->getSID());
    pnt.setContainsMeasurement();
    pnt.setInvDir(inv);
    pnt.init();
    npoints++;
  }
  if (npoints < npntCut) {
    algTrack->suppressLastPoints(npoints);
    mNPoints = npointsIni;
    npoints = -1;
  }
  mNPoints += npoints;

  return npoints;
}

} // namespace align
} // namespace o2
