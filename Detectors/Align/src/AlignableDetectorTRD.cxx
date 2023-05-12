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

/// @file   AlignableDetectorTRD.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  TRD detector wrapper

#include "Align/AlignableDetectorTRD.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorTRD.h"
#include "Align/Controller.h"
#include "Align/AlignConfig.h"
#include "TRDBase/Geometry.h"
#include "TRDBase/TrackletTransformer.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include <TMath.h>
#include <TGeoManager.h>

namespace o2
{
namespace align
{
using namespace TMath;
const char* AlignableDetectorTRD::CalibDOFName[AlignableDetectorTRD::NCalibParams] = {"DZdTglNRC", "DVDriftT"};

//____________________________________________
AlignableDetectorTRD::AlignableDetectorTRD(Controller* ctr) : AlignableDetector(DetID::TRD, ctr)
{
  // default c-tor
}

//____________________________________________
void AlignableDetectorTRD::defineVolumes()
{
  // define TRD volumes
  auto geo = o2::trd::Geometry::instance();
  geo->createPadPlaneArray();
  geo->createClusterMatrixArray(); // ideal T2L matrices

  AlignableSensorTRD* chamb = nullptr;
  AlignableVolume* sect[o2::trd::constants::NSECTOR]{};
  AlignableVolume* volTRD = nullptr; // fictious envelope

  addVolume(volTRD = new AlignableVolume("TRD_envelope", getDetLabel(), mController));
  volTRD->setDummyEnvelope();

  for (int ilr = 0; ilr < o2::trd::constants::NLAYER; ilr++) {                                 // layer
    for (int ich = 0; ich < o2::trd::constants::NSTACK * o2::trd::constants::NSECTOR; ich++) { // chamber
      int isector = ich / o2::trd::constants::NSTACK;
      int istack = ich % o2::trd::constants::NSTACK;
      uint16_t sid = o2::trd::Geometry::getDetector(ilr, istack, isector);
      const char* symname = Form("TRD/sm%02d/st%d/pl%d", isector, istack, ilr);
      addVolume(chamb = new AlignableSensorTRD(symname, o2::base::GeometryManager::getSensID(mDetID, sid), getSensLabel(sid), isector, mController));
      if (!gGeoManager->GetAlignableEntry(symname)) {
        chamb->setDummy(true);
        //        continue;
      }
      if (!sect[isector]) {
        sect[isector] = new AlignableVolume(Form("TRD/sm%02d", isector), getNonSensLabel(isector), mController);
        sect[isector]->setParent(volTRD);
      }
      chamb->setParent(sect[isector]);
    } // chamber
  }   // layer
  //
  for (int isc = 0; isc < o2::trd::constants::NSECTOR; isc++) {
    if (sect[isc]) {
      addVolume(sect[isc]);
    }
  }
}

//____________________________________________
void AlignableDetectorTRD::Print(const Option_t* opt) const
{
  // print info
  AlignableDetector::Print(opt);
  printf("Extra error for RC tracklets: Y:%e Z:%e\n", mExtraErrRC[0], mExtraErrRC[1]);
}

//____________________________________________
const char* AlignableDetectorTRD::getCalibDOFName(int i) const
{
  // return calibration DOF name
  return i < NCalibParams ? CalibDOFName[i] : nullptr;
}

//______________________________________________________
void AlignableDetectorTRD::writePedeInfo(FILE* parOut, const Option_t* opt) const
{
  // contribute to params and constraints template files for PEDE
  AlignableDetector::writePedeInfo(parOut, opt);
  //
  // write calibration parameters
  enum { kOff,
         kOn,
         kOnOn };
  const char* comment[3] = {"  ", "! ", "!!"};
  const char* kKeyParam = "parameter";
  //
  fprintf(parOut, "%s%s %s\t %d calibraction params for %s\n", comment[kOff], kKeyParam, comment[kOnOn],
          getNCalibDOFs(), GetName());
  //
  for (int ip = 0; ip < getNCalibDOFs(); ip++) {
    int cmt = isCondDOF(ip) ? kOff : kOn;
    fprintf(parOut, "%s %9d %+e %+e\t%s %s p%d\n", comment[cmt], getParLab(ip),
            -getParVal(ip), getParErr(ip), comment[kOnOn], isFreeDOF(ip) ? "  " : "FX", ip);
  }
  //
}

//______________________________________________________
void AlignableDetectorTRD::writeLabeledPedeResults(FILE* parOut) const
{
  //
  AlignableDetector::writeLabeledPedeResults(parOut);
  //
  for (int ip = 0; ip < getNCalibDOFs(); ip++) {
    fprintf(parOut, "%9d %+e %+e\t! calib param %d of %s %s %s\n", getParLab(ip), -getParVal(ip), getParErr(ip), ip, GetName(),
            isFreeDOF(ip) ? "   " : "FXU", o2::align::utils::isZeroAbs(getParVal(ip)) ? "FXP" : "   ");
  }
  //
}

//_______________________________________________________
double AlignableDetectorTRD::getCalibDOFVal(int id) const
{
  // return preset value of calibration dof
  double val = 0;
  switch (id) {
    case CalibNRCCorrDzDtgl:
      val = getNonRCCorrDzDtgl();
      break;
    case CalibDVT:
      val = getCorrDVT();
      break;
    default:
      break;
  };
  return val;
}

//_______________________________________________________
double AlignableDetectorTRD::getCalibDOFValWithCal(int id) const
{
  // return preset value of calibration dof + mp correction
  return getCalibDOFVal(id) + getParVal(id);
}

//____________________________________________
int AlignableDetectorTRD::processPoints(GIndex gid, bool inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  //
  const auto& algConf = AlignConfig::Instance();
  const auto recoData = mController->getRecoContainer();
  const auto& trk = recoData->getTrack<o2::trd::TrackTRD>(gid);
  if (trk.getNtracklets() < algConf.minTRDTracklets) {
    return -1;
  }
  auto propagator = o2::base::Propagator::Instance(); // float version!
  static float prevBz = -99999.;
  if (prevBz != propagator->getNominalBz()) {
    prevBz = propagator->getNominalBz();
    mRecoParam.setBfield(prevBz);
  }
  mNPoints = 0;
  const auto* transformer = mController->getTRDTransformer();
  auto algTrack = mController->getAlgTrack();
  mFirstPoint = algTrack->getNPoints();
  const auto trackletsRaw = recoData->getTRDTracklets();
  bool fail = false;
  int nPntIni = algTrack->getNPoints();
  o2::track::TrackPar trkParam = trk.getOuterParam(); // we refit outer param inward to get tracklet coordinates accounting for tilt
  for (int il = o2::trd::constants::NLAYER; il--;) {
    if (trk.getTrackletIndex(il) == -1) {
      continue;
    }
    int trkltId = trk.getTrackletIndex(il);
    const auto& trackletRaw = trackletsRaw[trkltId];
    const auto trackletCalibLoc = transformer->transformTracklet(trackletRaw, false); // calibrated tracket in local frame !!!
    int trkltDet = trackletRaw.getDetector();
    auto* sensor = (AlignableSensorTRD*)getSensor(trkltDet);
    if (sensor->isDummy()) {
      LOGP(error, "Dummy sensor {} is referred by a track", trkltDet);
      fail = true;
      continue;
    }
    double locXYZ[3] = {trackletCalibLoc.getX(), trackletCalibLoc.getY(), trackletCalibLoc.getZ()}, locXYZC[3], traXYZ[3];
    ;
    const auto& matAlg = sensor->getMatrixClAlg(); // local alignment matrix
    matAlg.LocalToMaster(locXYZ, locXYZC);         // aligned point in the local frame
    const auto& mat = sensor->getMatrixT2L();      // RS FIXME check if correct
    mat.MasterToLocal(locXYZC, traXYZ);

    // This is a hack until TRD T2L matrix problem will be solved
    const auto trackletCalib = recoData->getTRDCalibratedTracklets()[trkltId];
    traXYZ[0] = trackletCalib.getX();
    traXYZ[1] = trackletCalib.getY();
    traXYZ[2] = trackletCalib.getZ();

    int trkltSec = sensor->getSector();       // trkltDet / (o2::trd::constants::NLAYER * o2::trd::constants::NSTACK);
    float alpSens = sensor->getAlpTracking(); // o2::math_utils::sector2Angle(trkltSec);
    if (trkltSec != o2::math_utils::angle2Sector(trkParam.getAlpha()) ||
        !trkParam.rotateParam(alpSens) ||
        !propagator->propagateTo(trkParam, traXYZ[0], propagator->getNominalBz(), o2::base::Propagator::MAX_SIN_PHI, 10., o2::base::Propagator::MatCorrType::USEMatCorrNONE)) { // we don't need high precision here
      fail = true;
      break;
    }
    const o2::trd::PadPlane* pad = o2::trd::Geometry::instance()->getPadPlane(trkltDet);
    float tilt = std::tan(TMath::DegToRad() * pad->getTiltingAngle()); // tilt is signed! and returned in degrees
    float tiltCorrUp = tilt * (traXYZ[2] - trkParam.getZ());
    float zPosCorrUp = trkParam.getZ() + getNonRCCorrDzDtglWithCal() * trkParam.getTgl(); // + mRecoParam.getZCorrCoeffNRC() * trkParam.getTgl();
    float padLength = pad->getRowSize(trackletRaw.getPadRow());
    if (std::fabs(traXYZ[2] - trkParam.getZ()) < padLength) { // RS do we need this?
      tiltCorrUp = 0.f;
    }
    std::array<float, 2> trkltPosUp{float(traXYZ[1] - tiltCorrUp), zPosCorrUp};
    std::array<float, 3> trkltCovUp;
    mRecoParam.recalcTrkltCov(tilt, trkParam.getSnp(), pad->getRowSize(trackletRaw.getPadRow()), trkltCovUp);
    // Correction for DVT, equivalent to shift in X at which Y is evaluated: dY = tg_phi * dvt
    {
      auto dvt = getCorrDVTWithCal();
      if (std::abs(dvt) > utils::AlmostZeroF) {
        auto snp = trkParam.getSnp();
        auto slpY = snp / std::sqrt((1.f - snp) * (1.f + snp));
        trkltPosUp[0] += slpY * dvt;
      }
    }
    auto& pnt = algTrack->addDetectorPoint();
    const auto* sysE = sensor->getAddError(); // additional syst error
    pnt.setYZErrTracking(trkltCovUp[0] + sysE[0] * sysE[0], trkltCovUp[1], trkltCovUp[2] + sysE[1] * sysE[1]);
    if (getUseErrorParam()) { // errors will be calculated just before using the point in the fit, using track info
      pnt.setNeedUpdateFromTrack();
    }
    pnt.setXYZTracking(traXYZ[0], trkltPosUp[0], trkltPosUp[1]);
    pnt.setSensor(sensor);
    pnt.setAlphaSens(alpSens);
    pnt.setXSens(sensor->getXTracking());
    pnt.setDetID(mDetID);
    pnt.setSID(sensor->getSID());
    pnt.setContainsMeasurement();
    pnt.init();
    mNPoints++;
  }
  if (fail) { // reset points to original start
    algTrack->suppressLastPoints(mNPoints);
    mNPoints = 0;
  }
  return mNPoints;
}

} // namespace align
} // namespace o2
