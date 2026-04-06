// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>

#include "Framework/Logger.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITS3Align/AlignmentSensors.h"
#include "ITSBase/GeometryTGeo.h"

namespace o2::its3::align
{

AlignableVolume::Ptr buildHierarchyITS(AlignableVolume::SensorMapping& sensorMap)
{
  uint32_t gLbl{0}, det{0};
  auto geom = o2::its::GeometryTGeo::Instance();
  AlignableVolume *volHB{nullptr}, *volSt{nullptr}, *volHSt{nullptr}, *volMod{nullptr};
  std::unordered_map<std::string, AlignableVolume*> sym2vol;

  auto root = std::make_unique<AlignableVolume>(geom->composeSymNameITS(), gLbl++, det, false);
  sym2vol[root->getSymName()] = root.get();
  for (int ilr = 0; ilr < geom->getNumberOfLayers(); ilr++) {
    for (int ihb = 0; ihb < geom->getNumberOfHalfBarrels(); ihb++) {
      volHB = root->addChild(geom->composeSymNameHalfBarrel(ilr, ihb), gLbl++, det, false);
      sym2vol[volHB->getSymName()] = volHB;
      int nstavesHB = geom->getNumberOfStaves(ilr) / 2;
      for (int ist = 0; ist < nstavesHB; ist++) {
        volSt = volHB->addChild(geom->composeSymNameStave(ilr, ihb, ist), gLbl++, det, false);
        sym2vol[volSt->getSymName()] = volSt;
        for (int ihst = 0; ihst < geom->getNumberOfHalfStaves(ilr); ihst++) {
          volHSt = volSt->addChild(geom->composeSymNameHalfStave(ilr, ihb, ist, ihst), gLbl++, det, false);
          sym2vol[volHSt->getSymName()] = volHSt;
          for (int imd = 0; imd < geom->getNumberOfModules(ilr); imd++) {
            volMod = volHSt->addChild(geom->composeSymNameModule(ilr, ihb, ist, ihst, imd), gLbl++, det, false);
            sym2vol[volMod->getSymName()] = volMod;
          }
        }
      }
    }
  }

  // NOTE: for ITS sensors the local x and y are swapped
  int lay = 0, hba = 0, sta = 0, ssta = 0, modd = 0, chip = 0;
  for (int ich = 0; ich < geom->getNumberOfChips(); ich++) {
    geom->getChipId(ich, lay, hba, sta, ssta, modd, chip);
    GlobalLabel lbl(det, ich, true);
    AlignableVolume* parVol = sym2vol[modd < 0 ? geom->composeSymNameStave(lay, hba, sta) : geom->composeSymNameModule(lay, hba, sta, ssta, modd)];
    if (!parVol) {
      LOGP(fatal, "did not find parent for chip {}", ich);
    }
    int nch = modd < 0 ? geom->getNumberOfChipsPerStave(lay) : geom->getNumberOfChipsPerModule(lay);
    int jch = ich % nch;
    auto* chip = parVol->addChild<AlignableSensorITS>(geom->composeSymNameChip(lay, hba, sta, ssta, modd, jch), lbl);
    chip->setSensorId(ich);
    sensorMap[lbl] = chip;
  }
  return root;
}

AlignableVolume::Ptr buildHierarchyIT3(AlignableVolume::SensorMapping& sensorMap)
{
  uint32_t gLbl{0}, det{0};
  auto geom = o2::its::GeometryTGeo::Instance();
  AlignableVolume *volHB{nullptr}, *volSt{nullptr}, *volHSt{nullptr}, *volMod{nullptr};
  std::unordered_map<std::string, AlignableVolume*> sym2vol;

  auto root = std::make_unique<AlignableVolume>(geom->composeSymNameITS(), gLbl++, det, false);
  sym2vol[root->getSymName()] = root.get();
  for (int ilr = 0; ilr < geom->getNumberOfLayers(); ilr++) {
    const bool isLayITS3 = (ilr < 3);
    for (int ihb = 0; ihb < geom->getNumberOfHalfBarrels(); ihb++) {
      volHB = root->addChild(geom->composeSymNameHalfBarrel(ilr, ihb, isLayITS3), gLbl++, det, false);
      sym2vol[volHB->getSymName()] = volHB;
      if (isLayITS3) {
        volHB->setSensorId((2 * ilr) + ihb);
        continue; // no deeper hierarchy for ITS3 layers
      }
      int nstavesHB = geom->getNumberOfStaves(ilr) / 2;
      for (int ist = 0; ist < nstavesHB; ist++) {
        volSt = volHB->addChild(geom->composeSymNameStave(ilr, ihb, ist), gLbl++, det, false);
        sym2vol[volSt->getSymName()] = volSt;
        for (int ihst = 0; ihst < geom->getNumberOfHalfStaves(ilr); ihst++) {
          volHSt = volSt->addChild(geom->composeSymNameHalfStave(ilr, ihb, ist, ihst), gLbl++, det, false);
          sym2vol[volHSt->getSymName()] = volHSt;
          for (int imd = 0; imd < geom->getNumberOfModules(ilr); imd++) {
            volMod = volHSt->addChild(geom->composeSymNameModule(ilr, ihb, ist, ihst, imd), gLbl++, det, false);
            sym2vol[volMod->getSymName()] = volMod;
          }
        }
      }
    }
  }

  int lay = 0, hba = 0, sta = 0, ssta = 0, modd = 0, chip = 0;
  for (int ich = 0; ich < geom->getNumberOfChips(); ich++) {
    geom->getChipId(ich, lay, hba, sta, ssta, modd, chip);
    const bool isLayITS3 = (lay < 3);
    GlobalLabel lbl(det, ich, true);
    if (isLayITS3) {
      // ITS3 chips by construction do not have any DOFs still add them to have the measurment to alignable layer relation
      AlignableVolume* parVol = sym2vol[geom->composeSymNameHalfBarrel(lay, hba, true)];
      if (!parVol) {
        LOGP(fatal, "did not find parent for chip {}", ich);
      }
      auto* tile = parVol->addChild<AlignableSensorIT3>(geom->composeSymNameChip(lay, hba, sta, ssta, modd, chip, true), lbl);
      tile->setPseudo(true);
      tile->setSensorId(ich);
      sensorMap[lbl] = tile;
    } else {
      AlignableVolume* parVol = sym2vol[modd < 0 ? geom->composeSymNameStave(lay, hba, sta) : geom->composeSymNameModule(lay, hba, sta, ssta, modd)];
      if (!parVol) {
        LOGP(fatal, "did not find parent for chip {}", ich);
      }
      int nch = modd < 0 ? geom->getNumberOfChipsPerStave(lay) : geom->getNumberOfChipsPerModule(lay);
      int jch = ich % nch;
      auto* chip = parVol->addChild<AlignableSensorITS>(geom->composeSymNameChip(lay, hba, sta, ssta, modd, jch), lbl);
      chip->setSensorId(ich);
      sensorMap[lbl] = chip;
    }
  }
  return root;
}

void AlignableSensorITS::defineMatrixL2G()
{
  // the chip volume is not the measurment plane, need to correct for the epitaxial layer
  const auto* chipL2G = mPN->GetMatrix();
  mL2G = *chipL2G;
  double delta = itsmft::SegmentationAlpide::SensorLayerThickness - itsmft::SegmentationAlpide::SensorLayerThicknessEff;
  TGeoTranslation tra(0., 0.5 * delta, 0.);
  mL2G *= tra;
}

void AlignableSensorITS::defineMatrixT2L()
{
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  mL2G.LocalToMaster(locA, gloA);
  mL2G.LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  double xp = gloB[0] - (dx * t), yp = gloB[1] - (dy * t);
  double alp = std::atan2(yp, xp);
  o2::math_utils::bringTo02Pid(alp);
  mT2L.RotateZ(alp * TMath::RadToDeg()); // mT2L before is identity and afterwards rotated
  const TGeoHMatrix l2gI = mL2G.Inverse();
  mT2L.MultiplyLeft(l2gI);
}

void AlignableSensorITS::computeJacobianL2T(const double* posLoc, Matrix66& jac) const
{
  jac.setZero();
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotT2L(mT2L.GetRotationMatrix());
  Eigen::Matrix3d skew, rotL2T = rotT2L.transpose();
  skew << 0, -posLoc[2], posLoc[1], posLoc[2], 0, -posLoc[0], -posLoc[1], posLoc[0], 0;
  jac.topLeftCorner<3, 3>() = rotL2T;
  jac.topRightCorner<3, 3>() = -rotL2T * skew;
  jac.bottomRightCorner<3, 3>() = rotL2T;
}

void AlignableSensorIT3::defineMatrixL2G()
{
  mL2G = *mPN->GetMatrix();
}

void AlignableSensorIT3::defineMatrixT2L()
{
  double locA[3] = {-100., 0., 0.}, locB[3] = {100., 0., 0.}, gloA[3], gloB[3];
  mL2G.LocalToMaster(locA, gloA);
  mL2G.LocalToMaster(locB, gloB);
  double dx = gloB[0] - gloA[0], dy = gloB[1] - gloA[1];
  double t = (gloB[0] * dx + gloB[1] * dy) / (dx * dx + dy * dy);
  double xp = gloB[0] - (dx * t), yp = gloB[1] - (dy * t);
  double alp = std::atan2(yp, xp);
  o2::math_utils::bringTo02Pid(alp);
  mT2L.RotateZ(alp * TMath::RadToDeg());
  const TGeoHMatrix l2gI = mL2G.Inverse();
  mT2L.MultiplyLeft(l2gI);
}

void AlignableSensorIT3::computeJacobianL2T(const double* posLoc, Matrix66& jac) const
{
  jac.setZero();
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotT2L(mT2L.GetRotationMatrix());
  Eigen::Matrix3d skew, rotL2T = rotT2L.transpose();
  skew << 0, -posLoc[2], posLoc[1], posLoc[2], 0, -posLoc[0], -posLoc[1], posLoc[0], 0;
  jac.topLeftCorner<3, 3>() = rotL2T;
  jac.topRightCorner<3, 3>() = -rotL2T * skew;
  jac.bottomRightCorner<3, 3>() = rotL2T;
}

} // namespace o2::its3::align
