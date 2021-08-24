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
#include <TMath.h>
#include <cstdio>

using namespace TMath;
using namespace o2::align::utils;

namespace o2
{
namespace align
{

const char* AlignableDetectorITS::fgkHitsSel[AlignableDetectorITS::kNSPDSelTypes] =
  {"SPDNoSel", "SPDBoth", "SPDAny", "SPD0", "SPD1"};

//____________________________________________
AlignableDetectorITS::AlignableDetectorITS(Controller* ctr) : AlignableDetector(DetID::ITS, ctr)
{
  // default c-tor
  defineVolumes();
  setUseErrorParam();
  SetITSSelPatternColl();
  SetITSSelPatternCosm();
}

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
    addVolume(volLr = new AlignableVolume(geom->composeSymNameLayer(ilr), getNonSensLabel(nonSensCnt++), mController));
    sym2vol[volLr->getSymName()] = volLr;
    volLr->setParent(volITS);
    for (int ihb = 0; ihb < 2; ihb++) {
      addVolume(volHB = new AlignableVolume(geom->composeSymNameHalfBarrel(ilr, ihb), getNonSensLabel(nonSensCnt++), mController));
      volHB->setParent(volLr);
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
    }       // halfBarrel
  }         // layer

  for (int ich = 0; ich < geom->getNumberOfChips(); ich++) {
    int chID = o2::base::GeometryManager::getSensID(mDetID, ich);
    if (ich != chID) {
      throw std::runtime_error(fmt::format("mismatch between counter {} and composed {} chip IDs", ich, chID));
    }
    addVolume(sens = new AlignableSensorITS(o2::base::GeometryManager::getSymbolicName(mDetID, ich), chID, getSensLabel(chID), mController));
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
void AlignableDetectorITS::Print(const Option_t* opt) const
{
  AlignableDetector::Print(opt);
  printf("Sel.pattern   Collisions: %7s | Cosmic: %7s\n",
         GetITSPattName(fITSPatt[Coll]), GetITSPattName(fITSPatt[Cosm]));
}

/*
// RSTODO
//____________________________________________
bool AlignableDetectorITS::AcceptTrack(const AliESDtrack* trc, int trtype) const
{
  // test if detector had seed this track
  if (!CheckFlags(trc, trtype))
    return false;
  if (trc->GetNcls(0) < mNPointsSel[trtype])
    return false;
  if (!CheckHitPattern(trc, GetITSSelPattern(trtype)))
    return false;
  //
  return true;
}
*/

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
  // set type of points error parameterization
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
