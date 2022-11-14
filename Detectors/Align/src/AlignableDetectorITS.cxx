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

/// @file   AlignableDetectorITS.cxx
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  ITS detector wrapper

#include "Align/AlignableDetectorITS.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableSensorITS.h"
#include "Align/Controller.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/TrkClusRef.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ITStracking/IOUtils.h"
#include <TMath.h>
#include <cstdio>

using namespace TMath;
using namespace o2::align::utils;

namespace o2
{
namespace align
{

//____________________________________________
AlignableDetectorITS::AlignableDetectorITS(Controller* ctr) : AlignableDetector(DetID::ITS, ctr)
{
  // default c-tor
}

/*
//____________________________________________
void AlignableDetectorITS::initGeom()
{
  if (getInitGeomDone()) {
    return;
  }
  defineVolumes();
  AlignableDetector::initGeom();
}
*/
//____________________________________________
void AlignableDetectorITS::defineVolumes()
{
  // define ITS volumes
  //
  auto geom = o2::its::GeometryTGeo::Instance();

  AlignableVolume *volITS = nullptr, *volLr = nullptr, *volHB = nullptr, *volSt = nullptr, *volHSt = nullptr, *volMod = nullptr;
  AlignableSensorITS* sens = nullptr;
  //
  std::unordered_map<std::string, AlignableVolume*> sym2vol;
  addVolume(volITS = new AlignableVolume(geom->composeSymNameITS(), getDetLabel(), mController));
  sym2vol[volITS->getSymName()] = volITS;
  //
  int nonSensCnt = 0;
  for (int ilr = 0; ilr < geom->getNumberOfLayers(); ilr++) {
    for (int ihb = 0; ihb < geom->getNumberOfHalfBarrels(); ihb++) {
      addVolume(volLr = new AlignableVolume(geom->composeSymNameHalfBarrel(ilr, ihb), getNonSensLabel(nonSensCnt++), mController));
      sym2vol[volLr->getSymName()] = volLr;
      volLr->setParent(volITS);
      int nstavesHB = geom->getNumberOfStaves(ilr) / 2;
      for (int ist = 0; ist < nstavesHB; ist++) {
        addVolume(volSt = new AlignableVolume(geom->composeSymNameStave(ilr, ihb, ist), getNonSensLabel(nonSensCnt++), mController));
        sym2vol[volSt->getSymName()] = volSt;
        volSt->setParent(volLr);
        for (int ihst = 0; ihst < geom->getNumberOfHalfStaves(ilr); ihst++) {
          addVolume(volHSt = new AlignableVolume(geom->composeSymNameHalfStave(ilr, ihb, ist, ihst), getNonSensLabel(nonSensCnt++), mController));
          sym2vol[volHSt->getSymName()] = volHSt;
          volHSt->setParent(volSt);
          for (int imd = 0; imd < geom->getNumberOfModules(ilr); imd++) {
            addVolume(volMod = new AlignableVolume(geom->composeSymNameModule(ilr, ihb, ist, ihst, imd), getNonSensLabel(nonSensCnt++), mController));
            sym2vol[volMod->getSymName()] = volMod;
            volMod->setParent(volHSt);
          } // module
        }   //halfstave
      }     // stave
    }       // layer halfBarrel
  }         // layer

  for (int ich = 0; ich < geom->getNumberOfChips(); ich++) {
    int chID = o2::base::GeometryManager::getSensID(mDetID, ich);
    if (ich != chID) {
      throw std::runtime_error(fmt::format("mismatch between counter {} and composed {} chip IDs", ich, chID));
    }
    addVolume(sens = new AlignableSensorITS(o2::base::GeometryManager::getSymbolicName(mDetID, ich), chID, getSensLabel(ich), mController));
    int lay = 0, hba, sta = 0, ssta = 0, modd = 0, chip = 0;
    geom->getChipId(chID, lay, hba, sta, ssta, modd, chip);
    AlignableVolume* parVol = sym2vol[modd < 0 ? geom->composeSymNameStave(lay, hba, sta) : geom->composeSymNameModule(lay, hba, sta, ssta, modd)];
    if (!parVol) {
      throw std::runtime_error(fmt::format("did not find parent for chip {}", chID));
    }
    sens->setParent(parVol);
  }
  //
}

//____________________________________________
int AlignableDetectorITS::processPoints(GIndex gid, bool inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  //
  mNPoints = 0;
  auto algTrack = mController->getAlgTrack();
  auto recoData = mController->getRecoContainer();

  auto procClus = [this, &algTrack](const ClusterD& clus) {
    auto* sensor = this->getSensor(clus.getSensorID());
    auto& pnt = algTrack->addDetectorPoint();
    const auto* sysE = sensor->getAddError(); // additional syst error
    pnt.setYZErrTracking(clus.getSigmaY2() + sysE[0] * sysE[0], clus.getSigmaYZ(), clus.getSigmaZ2() + sysE[1] * sysE[1]);
    if (this->getUseErrorParam()) { // errors will be calculated just before using the point in the fit, using track info
      pnt.setNeedUpdateFromTrack();
    }
    pnt.setXYZTracking(clus.getX(), clus.getY(), clus.getZ());
    pnt.setSensor(sensor);
    pnt.setAlphaSens(sensor->getAlpTracking());
    pnt.setXSens(sensor->getXTracking());
    pnt.setDetID(this->mDetID);
    pnt.setSID(sensor->getSID());
    pnt.setContainsMeasurement();
    pnt.init();
    mNPoints++;
  };
  if (gid.getSource() == GIndex::ITS) {
    const auto tracks = recoData->getITSTracks();
    if (tracks.empty()) {
      return -1; // source not loaded?
    }
    const auto& track = tracks[gid.getIndex()];
    const auto& clusIdx = recoData->getITSTracksClusterRefs();
    // do we want to apply some cuts?
    int clEntry = track.getFirstClusterEntry();
    mFirstPoint = algTrack->getNPoints();
    for (int icl = track.getNumberOfClusters(); icl--;) {
      const auto& clus = mITSClustersArray[clusIdx[clEntry++]];
      procClus(clus);
    }
  } else { // ITSAB
    const auto& trkITSABref = recoData->getITSABRefs()[gid.getIndex()];
    const auto& ABTrackClusIdx = recoData->getITSABClusterRefs();
    int nCl = trkITSABref.getNClusters();
    int clEntry = trkITSABref.getFirstEntry();
    for (int icl = 0; icl < nCl; icl++) { // clusters are stored from inner to outer layers
      const auto& clus = mITSClustersArray[ABTrackClusIdx[clEntry + icl]];
      procClus(clus);
    }
  }
  return mNPoints;
}

//____________________________________________
bool AlignableDetectorITS::prepareDetectorData()
{
  // prepare TF data for processing: convert clusters
  auto recoData = mController->getRecoContainer();
  const auto clusITS = recoData->getITSClusters();
  const auto patterns = recoData->getITSClustersPatterns();
  auto pattIt = patterns.begin();
  mITSClustersArray.clear();
  mITSClustersArray.reserve(clusITS.size());

  for (auto& c : clusITS) {
    auto* sensor = getSensor(c.getSensorID());
    double sigmaY2, sigmaZ2, sigmaYZ = 0, locXYZC[3], traXYZ[3];
    auto locXYZ = o2::its::ioutils::extractClusterDataA(c, pattIt, mITSDict, sigmaY2, sigmaZ2); // local ideal coordinates
    const auto& matAlg = sensor->getMatrixClAlg();                                              // local alignment matrix !!! RS FIXME
    matAlg.LocalToMaster(locXYZ.data(), locXYZC);                                               // aligned point in the local frame
    const auto& mat = sensor->getMatrixT2L();                                                   // RS FIXME check if correct
    mat.MasterToLocal(locXYZC, traXYZ);
    /*
    if (applyMisalignment) {
      auto lrID = chmap.getLayer(c.getSensorID());
      sigmaY2 += conf.sysErrY2[lrID];
      sigmaZ2 += conf.sysErrZ2[lrID];
    }
    */
    auto& cl3d = mITSClustersArray.emplace_back(c.getSensorID(), traXYZ[0], traXYZ[1], traXYZ[2], sigmaY2, sigmaZ2, sigmaYZ); // local --> tracking
  }

  return true;
}

//____________________________________________
void AlignableDetectorITS::Print(const Option_t* opt) const
{
  AlignableDetector::Print(opt);
}

//____________________________________________
void AlignableDetectorITS::SetAddErrorLr(int ilr, double sigY, double sigZ)
{
  // set syst. errors for specific layer
  auto geom = o2::its::GeometryTGeo::Instance();
  int chMin = geom->getFirstChipIndex(ilr), chMax = geom->getLastChipIndex(ilr);
  for (int isn = chMin; isn <= chMax; isn++) {
    getSensor(isn)->setAddError(sigY, sigZ);
  }
}

//____________________________________________
void AlignableDetectorITS::SetSkipLr(int ilr)
{
  // exclude sensor of the layer from alignment
  auto geom = o2::its::GeometryTGeo::Instance();
  int chMin = geom->getFirstChipIndex(ilr), chMax = geom->getLastChipIndex(ilr);
  for (int isn = chMin; isn <= chMax; isn++) {
    getSensor(isn)->setSkip();
  }
}

//_________________________________________________
void AlignableDetectorITS::setUseErrorParam(int v)
{
  // set type of points error parameterization // RS DO WE NEED THIS?
  mUseErrorParam = v;
}

//_________________________________________________
void AlignableDetectorITS::updatePointByTrackInfo(AlignmentPoint* pnt, const trackParam_t* t) const
{
  // update point using specific error parameterization
  // the track must be in the detector tracking frame
  //TODO RS
  /*
  const AlignableSensor* sens = pnt->getSensor();
  int vid = sens->getVolID();
  double angPol = ATan(t.getTgl());
  double angAz = ASin(t.getSnp());
  double errY, errZ;
  GetErrorParamAngle(lr, angPol, angAz, errY, errZ);
  const double* sysE = sens->getAddError(); // additional syst error
  //
  pnt->setYZErrTracking(errY * errY + sysE[0] * sysE[0], 0, errZ * errZ + sysE[1] * sysE[1]);
  pnt->init();
  */
  //
}

} // namespace align
} // namespace o2
