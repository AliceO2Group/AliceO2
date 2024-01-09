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
#include "Align/AlignConfig.h"
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
  o2::itsmft::ChipMappingITS mp;
  mOverlaps = mp.getOverlapsInfo();
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
int AlignableDetectorITS::processPoints(GIndex gid, int npntCut, bool inv)
{
  // Extract the points corresponding to this detector, recalibrate/realign them to the
  // level of the "starting point" for the alignment/calibration session.
  // If inv==true, the track propagates in direction of decreasing tracking X
  // (i.e. upper leg of cosmic track)
  //
  auto algTrack = mController->getAlgTrack();
  auto recoData = mController->getRecoContainer();
  const auto& algConf = AlignConfig::Instance();
  int npoints = 0;
  auto procClus = [this, inv, &npoints, &algTrack](const ClusterD& clus) {
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
    pnt.setInvDir(inv);
    pnt.init();
    npoints++;
  };
  std::array<int, 7> clusIDs{};
  int nOverlaps = 0;
  if (gid.getSource() == GIndex::ITS) {
    const auto tracks = recoData->getITSTracks();
    if (tracks.empty()) {
      return -1; // source not loaded?
    }
    const auto& track = tracks[gid.getIndex()];
    if (track.getNClusters() < npntCut) {
      return -1;
    }
    const auto& clusIdx = recoData->getITSTracksClusterRefs();
    // do we want to apply some cuts?
    int clEntry = track.getFirstClusterEntry();
    for (int icl = track.getNumberOfClusters(); icl--;) { // clusters refs are stored from outer to inner layers, we loop in inner -> outer direction
      const auto& clus = mITSClustersArray[(clusIDs[npoints] = clusIdx[clEntry + icl])];
      if (clus.getBits()) { // overlapping clusters will have bit set
        if (clus.isBitSet(EdgeFlags::Biased)) {
          continue;
        }
        if (clus.getCount()) {
          nOverlaps++;
        }
      }
      procClus(clus);
    }
  } else { // ITSAB
    const auto& trkITSABref = recoData->getITSABRefs()[gid.getIndex()];
    const auto& ABTrackClusIdx = recoData->getITSABClusterRefs();
    int nCl = trkITSABref.getNClusters();
    int clEntry = trkITSABref.getFirstEntry();
    for (int icl = 0; icl < nCl; icl++) { // clusters are stored from inner to outer layers
      const auto& clus = mITSClustersArray[(clusIDs[npoints] = ABTrackClusIdx[clEntry + icl])];
      if (clus.getBits()) { // overlapping clusters will have bit set
        if (clus.isBitSet(EdgeFlags::Biased)) {
          continue;
        }
        if (clus.getCount()) {
          nOverlaps++;
        }
      }
      procClus(clus);
    }
  }
  if (npoints < npntCut) { // reset points to original start
    algTrack->suppressLastPoints(npoints);
    npoints = 0;
    return 0;
  }

  // do we need to process overlaps?
  if (nOverlaps) {
    auto trcProp = mController->getRecoContainer()->getTrackParam(mController->getAlgTrack()->getCurrentTrackID());
    trcProp.resetCovariance(10);
    auto propagator = o2::base::Propagator::Instance();
    for (int icl = 0; icl < npoints; icl++) {
      const auto& clus = mITSClustersArray[clusIDs[icl]];
      float alp = getSensor(clus.getSensorID())->getAlpTracking();
      if (!trcProp.rotate(alp) ||
          !propagator->propagateToX(trcProp, clus.getX(), propagator->getNominalBz(), algConf.maxSnp, algConf.maxStep, base::Propagator::MatCorrType(algConf.matCorType)) ||
          !trcProp.update(clus)) {
        break;
      }

      if (clus.getCount()) { // there is an overlap, find best matching cluster
        nOverlaps--;
        int bestClusID = -1, clusIDtoCheck = mOverlapClusRef[clusIDs[icl]];
        float bestChi2 = algConf.ITSOverlapMaxChi2;
        auto trPropOvl = trcProp;
        for (int iov = 0; iov < clus.getCount(); iov++) {
          int clusOvlID = mOverlapCandidateID[clusIDtoCheck];
          const auto& clusOvl = mITSClustersArray[clusOvlID];
          if (iov == 0) {
            if (!trPropOvl.rotate(getSensor(clusOvl.getSensorID())->getAlpTracking()) ||
                !propagator->propagateToX(trPropOvl, clusOvl.getX(), propagator->getNominalBz(), algConf.maxSnp, algConf.maxStep, base::Propagator::MatCorrType::USEMatCorrNONE)) {
              break;
            }
          }
          auto chi2 = trPropOvl.getPredictedChi2(clusOvl);
          if (chi2 < bestChi2) {
            bestChi2 = chi2;
            bestClusID = clusOvlID;
          }
        }
        if (bestClusID != -1) { // account overlapping cluster
          procClus(mITSClustersArray[bestClusID]);
        }
      }
      if (!nOverlaps) {
        break;
      }
    }
  }
  mNPoints += npoints;
  return npoints;
}

//____________________________________________
bool AlignableDetectorITS::prepareDetectorData()
{
  // prepare TF data for processing: convert clusters
  const auto& algConf = AlignConfig::Instance();
  auto recoData = mController->getRecoContainer();
  const auto clusITS = recoData->getITSClusters();
  const auto clusITSROF = recoData->getITSClustersROFRecords();
  const auto patterns = recoData->getITSClustersPatterns();
  auto pattIt = patterns.begin();
  mITSClustersArray.clear();
  mITSClustersArray.reserve(clusITS.size());
  if (algConf.ITSOverlapMargin > 0) {
    mOverlapClusRef.clear();
    mOverlapClusRef.resize(clusITS.size(), -1);

    mOverlapCandidateID.clear();
    mOverlapCandidateID.reserve(clusITS.size());
  }
  static std::vector<int> edgeClusters;
  int ROFCount = 0;
  int16_t curSensID = -1;
  struct ROFChipEntry {
    int16_t rofCount = -1;
    int chipFirstEntry = -1;
  };
  std::array<ROFChipEntry, o2::itsmft::ChipMappingITS::getNChips()> chipROFStart{}; // fill only for clusters with overlaps

  for (const auto& rof : clusITSROF) {
    int maxic = rof.getFirstEntry() + rof.getNEntries();
    edgeClusters.clear();
    for (int ic = rof.getFirstEntry(); ic < maxic; ic++) {
      const auto& c = clusITS[ic];
      int16_t sensID = c.getSensorID();
      auto* sensor = getSensor(sensID);
      double sigmaY2, sigmaZ2, sigmaYZ = 0, locXYZC[3], traXYZ[3];
      auto pattItCopy = pattIt;
      auto locXYZ = o2::its::ioutils::extractClusterDataA(c, pattIt, mITSDict, sigmaY2, sigmaZ2); // local ideal coordinates
      const auto& matAlg = sensor->getMatrixClAlg();                                              // local alignment matrix !!! RS FIXME
      matAlg.LocalToMaster(locXYZ.data(), locXYZC);                                               // aligned point in the local frame
      const auto& mat = sensor->getMatrixT2L();                                                   // RS FIXME check if correct
      mat.MasterToLocal(locXYZC, traXYZ);
      auto& cl3d = mITSClustersArray.emplace_back(sensID, traXYZ[0], traXYZ[1], traXYZ[2], sigmaY2, sigmaZ2, sigmaYZ); // local --> tracking

      if (algConf.ITSOverlapMargin > 0) {
        // fill chips overlaps info for clusters whose center is within of the algConf.ITSOverlapMargin distance from the chip min or max row edge
        // but the pixel closest to this edge has distance of at least algConf.ITSOverlapEdgeRows from the edge
        int row = 0, col = 0;
        o2::itsmft::SegmentationAlpide::localToDetectorUnchecked(locXYZ[0], locXYZ[2], row, col);
        int drow = row < o2::itsmft::SegmentationAlpide::NRows / 2 ? row : o2::itsmft::SegmentationAlpide::NRows - row - 1; // distance to the edge
        if (drow * o2::itsmft::SegmentationAlpide::PitchRow < algConf.ITSOverlapMargin) {                                   // rough check is passed, check if the edge cluster is indeed good
          cl3d.setBit(row < o2::itsmft::SegmentationAlpide::NRows / 2 ? EdgeFlags::LowRow : EdgeFlags::HighRow);            // flag that this is an edge cluster and indicate the low/high row side
          // check if it is not too close to the edge (to be biased)
          if (algConf.ITSOverlapEdgeRows > 0) { // is there a restriction?
            auto pattID = c.getPatternID();
            drow = c.getRow();
            if (pattID != itsmft::CompCluster::InvalidPatternID) {
              if (!mITSDict->isGroup(pattID)) {
                const auto& patt = mITSDict->getPattern(pattID); // reference pixel is min row/col corner
                if (row > o2::itsmft::SegmentationAlpide::NRows / 2) {
                  drow = o2::itsmft::SegmentationAlpide::NRows - 1 - (drow + patt.getRowSpan() - 1);
                }
              } else { // group: reference pixel is the one containing the COG
                o2::itsmft::ClusterPattern patt(pattItCopy);
                drow = row < o2::itsmft::SegmentationAlpide::NRows / 2 ? drow - patt.getRowSpan() / 2 : o2::itsmft::SegmentationAlpide::NRows - 1 - (drow + patt.getRowSpan() / 2 - 1);
              }
            } else {
              o2::itsmft::ClusterPattern patt(pattItCopy); // reference pixel is min row/col corner
              if (row > o2::itsmft::SegmentationAlpide::NRows / 2) {
                drow = o2::itsmft::SegmentationAlpide::NRows - 1 - (drow + patt.getRowSpan() - 1);
              }
            }
            if (drow < algConf.ITSOverlapEdgeRows) { // too close to the edge, flag this
              cl3d.setBit(EdgeFlags::Biased);
            }
          }
          if (!cl3d.isBitSet(EdgeFlags::Biased)) {
            if (chipROFStart[sensID].rofCount != ROFCount) { // remember 1st entry
              chipROFStart[sensID].rofCount = ROFCount;
              chipROFStart[sensID].chipFirstEntry = edgeClusters.size(); // remember 1st entry of edge cluster for this chip
            }
            edgeClusters.push_back(ic);
          }
        }
      }
    } // clusters of ROF
    // relate edge clusters of ROF to each other
    int prevSensID = -1;
    for (auto ic : edgeClusters) {
      auto& cl = mITSClustersArray[ic];
      int sensID = cl.getSensorID();
      auto ovl = mOverlaps[sensID];
      int ovlCount = 0;
      for (int ir = 0; ir < OVL::NSides; ir++) {
        if (ovl.rowSide[ir] == OVL::NONE) { // no overlap from this row side
          continue;
        }
        int chipOvl = ovl.rowSide[ir]; // look for overlaps with this chip
        // are there clusters with overlaps on chipOvl?
        if (chipROFStart[chipOvl].rofCount == ROFCount) {
          auto oClusID = edgeClusters[chipROFStart[chipOvl].chipFirstEntry];
          while (oClusID < int(mITSClustersArray.size())) {
            auto oClus = mITSClustersArray[oClusID];
            if (oClus.getSensorID() != sensID) {
              break; // no more clusters on the overlapping chip
            }
            if (oClus.isBitSet(ovl.rowSideOverlap[ir]) &&                       // make sure that the edge cluster is on the right side of the row
                !oClus.isBitSet(EdgeFlags::Biased) &&                           // not too close to the edge
                std::abs(oClus.getZ() - cl.getZ()) < algConf.ITSOverlapMaxDZ) { // apply fiducial cut on Z distance of 2 clusters
              // register overlaping cluster
              if (!ovlCount) { // 1st overlap
                mOverlapClusRef[ic] = mOverlapCandidateID.size();
              }
              mOverlapCandidateID.push_back(oClusID);
              ovlCount++;
            }
            oClusID++;
          }
        }
      }
      cl.setCount(std::min(127, ovlCount));
    }

    ROFCount++;
  } // loop over ROFs
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
