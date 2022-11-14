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
      const char* symname = Form("TOF/sm%02d/strip%02d", isc, istr);
      addVolume(strip = new AlignableSensorTOF(symname, o2::base::GeometryManager::getSensID(DetID::TOF, cnt), getSensLabel(cnt), isc, mController));
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

  for (int isc = 0; isc < NSect; isc++) {
    if (sect[isc]) {
      addVolume(sect[isc]);
    }
  }
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
  double loc[3] = {(det[4] + 0.5) * o2::tof::Geo::XPAD - o2::tof::Geo::XHALFSTRIP, 0., (det[3] - 0.5) * o2::tof::Geo::ZPAD}, locCorr[3] = {}, traCorr[3] = {};
  const auto& matAlg = sensor->getMatrixClAlg();
  matAlg.LocalToMaster(loc, locCorr);
  // rotate to tracking
  const auto& matT2L = sensor->getMatrixT2L();
  matT2L.MasterToLocal(locCorr, traCorr);
  //
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
}

} // namespace align
} // namespace o2
