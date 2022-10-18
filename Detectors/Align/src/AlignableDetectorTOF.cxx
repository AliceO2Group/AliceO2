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

/// @file   AlignableDetectorTOF.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Wrapper for TOF detector

#include "Align/AlignableDetectorTOF.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorTOF.h"
#include "Align/Controller.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTOF/Cluster.h"
#include "TOFBase/Geo.h"
#include <TGeoManager.h>

ClassImp(o2::align::AlignableDetectorTOF);

namespace o2
{
namespace align
{

//____________________________________________
AlignableDetectorTOF::AlignableDetectorTOF(Controller* ctr) : AlignableDetector(DetID::TOF, ctr)
{
  // default c-tor
}

//____________________________________________
void AlignableDetectorTOF::defineVolumes()
{
  // define TOF volumes
  //
  constexpr int NSect = 18;
  int labDet = getDetLabel();
  AlignableSensorTOF* strip = nullptr;
  //
  //  AddVolume( volTOF = new AlignableVolume("TOF") ); // no main volume, why?
  AlignableVolume* sect[NSect] = {};
  //
  int cnt = 0;
  for (int isc = 0; isc < NSect; isc++) {
    for (int istr = 1; istr <= o2::tof::Geo::NSTRIPXSECTOR; istr++) { // strip
      int modUID = o2::base::GeometryManager::getSensID(DetID::TOF, cnt);
      const char* symname = Form("TOF/sm%02d/strip%02d", isc, istr);
      addVolume(strip = new AlignableSensorTOF(symname, cnt, getSensLabel(cnt), isc, mController));
      if (!gGeoManager->GetAlignableEntry(symname)) {
        strip->setDummy(true);
        //        continue;
      }
      if (!sect[isc]) {
        sect[isc] = new AlignableVolume(Form("TOF/sm%02d", isc), getNonSensLabel(isc), mController);
      }
      strip->setParent(sect[isc]);
    } // strip
  }   // layer
  //
}

//____________________________________________
int AlignableDetectorTOF::processPoints(GIndex gid, bool inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  //

  mNPoints = 0;
  auto algTrack = mController->getAlgTrack();
  auto recoData = mController->getRecoContainer();
  auto TOFClusters = recoData->getTOFClusters();
  if (TOFClusters.empty()) {
    return -1; // source not loaded?
  }
  const auto& clus = TOFClusters[gid.getIndex()];
  int det[5] = {}, ch = clus.getMainContributingChannel();
  o2::tof::Geo::getVolumeIndices(ch, det);
  int sensID = o2::tof::Geo::getStripNumberPerSM(det[1], det[2]) + clus.getSector() * o2::tof::Geo::NSTRIPXSECTOR;
  auto* sensor = (AlignableSensorTOF*)getSensor(sensID);
  if (sensor->isDummy()) {
    LOGP(error, "Dummy sensor {} is referred by a track", sensID);
    return 0;
  }
  float posf[3] = {};
  o2::tof::Geo::getPos(det, posf);
  o2::tof::Geo::rotateToSector(posf, det[0]);
  o2::tof::Geo::rotateToSector(posf, 18);       // twisted sector tracking coordinates?
  double tra[3] = {posf[1], posf[0], -posf[2]}; // sector track coordinates ?
  // rotate to local
  double loc[3] = {}, locCorr[3] = {}, traCorr[3] = {};
  const auto& matT2L = sensor->getMatrixT2L();
  matT2L.LocalToMaster(tra, loc);
  // correct for misalignments
  const auto& matAlg = sensor->getMatrixClAlg();
  matAlg.LocalToMaster(loc, locCorr);
  // rotate back to tracking
  matT2L.MasterToLocal(locCorr, traCorr);
  //
  // alternative method via TOF methods from PR10102
  //  float posS[3] = {};
  //  o2::tof::Geo::getPos(det, posS);
  //  o2::tof::Geo::getPosInStripCoord(ch, posS);

  mFirstPoint = algTrack->getNPoints();
  auto& pnt = algTrack->addDetectorPoint();

  const auto* sysE = sensor->getAddError(); // additional syst error
  pnt.setYZErrTracking(clus.getSigmaY2() + sysE[0] * sysE[0], clus.getSigmaYZ(), clus.getSigmaZ2() + sysE[1] * sysE[1]);
  if (getUseErrorParam()) { // errors will be calculated just before using the point in the fit, using track info
    pnt.setNeedUpdateFromTrack();
  }
  pnt.setXYZTracking(traCorr[0], traCorr[1], traCorr[2]);
  pnt.setSensor(sensor);
  pnt.setAlphaSens(sensor->getAlpTracking());
  pnt.setXSens(sensor->getXTracking());
  pnt.setDetID(mDetID);
  pnt.setSID(sensor->getSID());
  //
  pnt.setContainsMeasurement();
  pnt.init();
  mNPoints++;
  return mNPoints;
  /*

  mNPoints = 0;
  auto algTrack = mController->getAlgTrack();
  auto recoData = mController->getRecoContainer();
  auto TOFClusters = recoData->getTOFClusters();
  if (TOFClusters.empty()) {
    return -1; // source not loaded?
  }
  const auto& clus = TOFClusters[gid.getIndex()];
  int det[5];
  float posf[3];
  int ch = clus.getMainContributingChannel();
  o2::tof::Geo::getVolumeIndices(ch, det);
  int sensID = o2::tof::Geo::getStripNumberPerSM(det[1], det[2]) + clus.getSector() * o2::tof::Geo::NSTRIPXSECTOR;
  o2::tof::Geo::getPos(det, posf);
  o2::tof::Geo::rotateToSector(posf, clus.getSector());
  auto* sensor = (AlignableSensorTOF*)getSensor(sensID);
  if (sensor->isDummy()) {
    LOGP(error, "Dummy sensor {} is referred by a track", sensID);
    return 0;
  }
  const auto& matAlg = sensor->getMatrixClAlg();
  double locXYZ[3]{posf[0], posf[1], posf[2]}, locXYZC[3], traXYZ[3];
  matAlg.LocalToMaster(locXYZ, locXYZC);
  const auto& mat = sensor->getMatrixT2L(); // RS FIXME check if correct
  mat.MasterToLocal(locXYZC, traXYZ);
  mFirstPoint = algTrack->getNPoints();
  auto& pnt = algTrack->addDetectorPoint();

  const auto* sysE = sensor->getAddError(); // additional syst error
  pnt.setYZErrTracking(clus.getSigmaY2() + sysE[0] * sysE[0], clus.getSigmaYZ(), clus.getSigmaZ2() + sysE[1] * sysE[1]);
  if (getUseErrorParam()) { // errors will be calculated just before using the point in the fit, using track info
    pnt.setNeedUpdateFromTrack();
  }
  pnt.setXYZTracking(traXYZ[0], traXYZ[1], traXYZ[2]);
  pnt.setSensor(sensor);
  pnt.setAlphaSens(sensor->getAlpTracking());
  pnt.setXSens(sensor->getXTracking());
  pnt.setDetID(mDetID);
  pnt.setSID(sensor->getSID());
  //
  pnt.setContainsMeasurement();
  pnt.init();
  mNPoints++;
  return mNPoints;

   */
}

} // namespace align
} // namespace o2
