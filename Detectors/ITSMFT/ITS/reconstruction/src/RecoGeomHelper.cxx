// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoLayer.cxx
/// \brief Implementation of the Aux. container for clusters, optimized for tracking
/// \author iouri.belikov@cern.ch

#include "ITSReconstruction/RecoGeomHelper.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSBase/GeometryTGeo.h"

using namespace o2::its;

//_____________________________________________________________________
void RecoGeomHelper::RecoChip::updateLimits(const o2::math_utils::Point3D<float>& pntTra)
{
  // update limits from the edge point in tracking frame
  yRange.update(pntTra.Y());
  zRange.update(pntTra.Z());
}

//_____________________________________________________________________
void RecoGeomHelper::RecoChip::print() const
{
  printf("Ch#%4d Alp: %+.3f X:%5.2f %+6.3f<y<%+6.3f  %+6.3f<z<%+6.3f | XYEdges: {%+6.3f,%+6.3f}{%+6.3f,%+6.3f}\n",
         id, alp, xRef, yRange.getMin(), yRange.getMax(), zRange.getMin(), zRange.getMax(),
         xyEdges.getX0(), xyEdges.getY0(), xyEdges.getX1(), xyEdges.getY1());
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLadder::updateLimits(const o2::math_utils::Point3D<float>& pntGlo)
{
  // update limits from the point in Global frame
  float phi = pntGlo.phi();    // -pi:pi range
  o2::math_utils::bringTo02Pi(phi); // temporary bring to 0:2pi range
  o2::math_utils::bringTo02Pi(phiRange.getMin());
  o2::math_utils::bringTo02Pi(phiRange.getMax());
  phiRange.update(phi);
  phiMean = phiRange.mean();
  dphiH = 0.5 * phiRange.delta();
  if (phiRange.delta() > o2::constants::math::PI) {     // wrapping, swap
    phiRange.set(phiRange.getMax(), phiRange.getMin()); // swap
    phiMean -= o2::constants::math::PI;
    dphiH = o2::constants::math::PI - dphiH;
  }
  o2::math_utils::bringToPMPi(phiRange.getMin()); // -pi:pi range
  o2::math_utils::bringToPMPi(phiRange.getMax());
  o2::math_utils::bringToPMPi(phiMean);
  //
  zRange.update(pntGlo.Z());
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLadder::init()
{
  auto& chip = chips[0].xyEdges;
  float x0(chip.getX0()), y0(chip.getY0()), x1(chip.getX1()), y1(chip.getY1());
  for (int i = 1; i < (int)chips.size(); i++) {
    chip = chips[i].xyEdges;
    x0 = chip.getDX() > 0 ? std::min(x0, chip.getX0()) : std::max(x0, chip.getX0());
    x1 = chip.getDX() > 0 ? std::max(x1, chip.getX1()) : std::min(x1, chip.getX1());
    y0 = chip.getDY() > 0 ? std::min(y0, chip.getY0()) : std::max(y0, chip.getY0());
    y1 = chip.getDY() > 0 ? std::max(y1, chip.getY1()) : std::min(y1, chip.getY1());
  }
  xyEdges.setEdges(x0, y0, x1, y1);
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLadder::print() const
{
  assert(overlapWithNext != Undefined || chips.size() == 0); // make sure there are no undefined ladders after init is done
  printf("Ladder %3d  %.3f<phi[<%.3f>]<%.3f dPhiH:%.3f | XYEdges: {%+6.3f,%+6.3f}{%+6.3f,%+6.3f} | %3d chips | OvlNext: %s\n",
         id, phiRange.getMin(), phiMean, phiRange.getMax(), dphiH,
         xyEdges.getX0(), xyEdges.getY0(), xyEdges.getX1(), xyEdges.getY1(), (int)chips.size(),
         overlapWithNext == Undefined ? "N/A" : ((overlapWithNext == NoOverlap ? "NO" : (overlapWithNext == Above ? "Above" : "Below"))));
  for (const auto& ch : chips) {
    ch.print();
  }
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLayer::init()
{
  auto gm = o2::its::GeometryTGeo::Instance();
  gm->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2L)); // more matrices ?

  int nHStaves = gm->getNumberOfHalfStaves(id);
  int nStaves = gm->getNumberOfStaves(id);
  float dxH = o2::itsmft::SegmentationAlpide::SensorSizeRows / 2; // half width in rphi
  float dzH = o2::itsmft::SegmentationAlpide::SensorSizeCols / 2; // half width in Z
  int nCh = gm->getNumberOfChipsPerLayer(id), chip0 = gm->getFirstChipIndex(id);
  int nChMod = gm->getNumberOfChipsPerModule(id), nChModH = nChMod / 2;
  nLadders = nHStaves > 1 ? nStaves * nHStaves * 2 : nStaves; // 2 ladders per h-stave for OB
  ladders.resize(nLadders);

  for (auto lad : ladders) {
    lad.chips.reserve(gm->getNumberOfChipsPerHalfStave(id) / (id > 2 ? 2 : 1)); // 2 ladders per h-stave for OB
  }
  for (int ich = 0; ich < nCh; ich++) {
    int chipID = chip0 + ich, lay, sta, ssta, mod, chipInMod;
    gm->getChipId(chipID, lay, sta, ssta, mod, chipInMod);
    int ladID = sta, chipInLadder = nChMod - chipInMod - 1; // count from negative to positive Z, contrary to official chips numbering
    if (nHStaves > 1) {                                     // OB
      int modUpper = chipInMod / nChModH;
      ladID = sta * 4 + ssta * 2 + modUpper; // OB module covers 2 "ladders"
    }
    auto& ladder = ladders[ladID];
    auto& chip = ladder.chips.emplace_back();
    chip.id = chipID;
    gm->getSensorXAlphaRefPlane(chipID, chip.xRef, chip.alp);
    o2::math_utils::sincos(chip.alp, chip.snAlp, chip.csAlp);

    o2::math_utils::Point3D<float> edgeLoc(-dxH, 0.f, -dzH);
    auto edgeTra = gm->getMatrixT2L(chipID) ^ (edgeLoc); // edge in tracking frame
    chip.updateLimits(edgeTra);
    auto edgeGloM = gm->getMatrixT2GRot(chipID)(edgeTra); // edge in global frame
    updateLimits(edgeGloM);
    ladder.updateLimits(edgeGloM);

    edgeLoc.SetXYZ(dxH, 0.f, dzH);
    edgeTra = gm->getMatrixT2L(chipID) ^ (edgeLoc); // edge in tracking frame
    chip.updateLimits(edgeTra);
    auto edgeGloP = gm->getMatrixT2GRot(chipID)(edgeTra); // edge in globalframe
    updateLimits(edgeGloP);
    ladder.updateLimits(edgeGloP);
    chip.xyEdges.setEdges(edgeGloM.X(), edgeGloM.Y(), edgeGloP.X(), edgeGloP.Y());
  }

  // sort according to mean phi (in -pi:pi range!!!)
  std::sort(ladders.begin(), ladders.end(), [](auto& a, auto& b) {
    float pha = a.phiMean, phb = b.phiMean;
    o2::math_utils::bringTo02Pi(pha);
    o2::math_utils::bringTo02Pi(phb);
    return pha < phb;
  });

  // make sure chips within the ladder are ordered in Z, renumber ladders
  for (int i = nLadders; i--;) {
    auto& lad = ladders[i];
    std::sort(lad.chips.begin(), lad.chips.end(), [](auto& a, auto& b) { return a.zRange.getMin() < b.zRange.getMin(); });
    lad.id = i;
    lad.init();
  }
  // relate to its neighbour: is the egde at higher R than the edge of the next ladder and do we check for overlaps ?
  for (int i = nLadders; i--;) {
    auto& lad = ladders[i];
    auto& plad = i > 0 ? ladders[i - 1] : ladders[nLadders - 1];
    if (nHStaves == 2) { // on the OB the ladders of the same halfstave do not overlap
      int tlay, tsta, tssta, tmod, tchipInMod;
      int play, psta, pssta, pmod, pchipInMod;
      gm->getChipId(lad.chips.front().id, tlay, tsta, tssta, tmod, tchipInMod);
      gm->getChipId(plad.chips.front().id, play, psta, pssta, pmod, pchipInMod);
      if (tsta == psta && tssta == pssta) { // these are 2 ladders of the same halfstave, no overlap
        plad.overlapWithNext = RecoLadder::NoOverlap;
        continue;
      }
    }
    // need to compare the radius of the edge at lowest phi for this ladder with radius at highest phi of prev ladder
    // find the radius of the edge at low phi
    float phi0 = std::atan2(lad.xyEdges.getY0(), lad.xyEdges.getX0());
    float phi1 = std::atan2(lad.xyEdges.getY1(), lad.xyEdges.getX1());
    o2::math_utils::bringTo02Pi(phi0); // we don't know a priori if edge0/1 corresponds to low/high phi or vice versa
    o2::math_utils::bringTo02Pi(phi1);
    float r2This = (phi0 < phi1 && phi1 - phi0 < o2::constants::math::PI) ? // pick R of lowest angle
                     lad.xyEdges.getX0() * lad.xyEdges.getX0() + lad.xyEdges.getY0() * lad.xyEdges.getY0()
                                                                          : lad.xyEdges.getX1() * lad.xyEdges.getX1() + lad.xyEdges.getY1() * lad.xyEdges.getY1();
    //
    phi0 = std::atan2(plad.xyEdges.getY0(), plad.xyEdges.getX0());
    phi1 = std::atan2(plad.xyEdges.getY1(), plad.xyEdges.getX1());
    o2::math_utils::bringTo02Pi(phi0); // we don't know a priori if edge0/1 corresponds to low/high phi or vice versa
    o2::math_utils::bringTo02Pi(phi1);
    float r2Prev = (phi0 < phi1 && phi1 - phi0 < o2::constants::math::PI) ? // pick R of highest angle
                     plad.xyEdges.getX1() * plad.xyEdges.getX1() + plad.xyEdges.getY1() * plad.xyEdges.getY1()
                                                                          : plad.xyEdges.getX0() * plad.xyEdges.getX0() + plad.xyEdges.getY0() * plad.xyEdges.getY0();
    //
    plad.overlapWithNext = r2Prev > r2This ? RecoLadder::Above : RecoLadder::Below;
  }

  int ndiv = nLadders * 3; // number of bins for mapping
  phi2ladder.resize(ndiv);
  float dphi = o2::constants::math::TwoPI / ndiv;
  int laddId = 0;
  for (int i = 0; i < ndiv; i++) {
    float phi = (0.5 + i) * dphi;
    o2::math_utils::bringToPMPi(phi);
    while (laddId < nLadders) {
      const auto& lad = ladders[laddId];
      auto rel = lad.isPhiOutside(phi);
      if (rel != RecoGeomHelper::Above) {
        break;
      }
      laddId++; // laddId was below phi, catch up
    }
    phi2ladder[i] = laddId % nLadders;
  }
  lastChipInLadder = ladders[0].chips.size();
  z2chipID = lastChipInLadder / zRange.delta();
  lastChipInLadder--;
  rInv = 1. / rRange.mean();
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLayer::updateLimits(const o2::math_utils::Point3D<float>& pntGlo)
{
  // update limits from the point in global frame
  rRange.update(pntGlo.Rho());
  zRange.update(pntGlo.Z());
}

//_____________________________________________________________________
void RecoGeomHelper::RecoLayer::print() const
{
  printf("\nLayer %d %.2f<r<%.2f %+.2f<z<%+.2f  %d ladders\n",
         id, rRange.getMin(), rRange.getMax(), zRange.getMin(), zRange.getMax(), (int)ladders.size());
  for (const auto& ld : ladders) {
    ld.print();
  }
}

//_____________________________________________________________________
void RecoGeomHelper::init()
{
  for (int il = int(layers.size()); il--;) {
    auto& lr = layers[il];
    lr.id = il;
    lr.init();
  }
}

//_____________________________________________________________________
void RecoGeomHelper::print() const
{
  for (const auto& lr : layers) {
    lr.print();
  }
}
