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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Align/Controller.h"
#include "Align/AlignableVolume.h"
#include "Align/AlignableDetectorITS.h"
#include "Align/AlignableDetectorTPC.h"
#include "Align/AlignableDetectorTRD.h"
#include "Align/AlignableDetectorTOF.h"
#endif
#include "Framework/Logger.h"

using namespace o2::align;

void configITS(Controller* c, int par);
void configTPC(Controller* c, int par);
void configTRD(Controller* c, int par);
void configTOF(Controller* c, int par);

int algconf(Controller* c, int par)
{
  LOG(info) << "calling algconf with " << c << " " << par;

  if (c->getDetector(o2::detectors::DetID::ITS)) {
    configITS(c, par);
  }
  if (c->getDetector(o2::detectors::DetID::TPC)) {
    configTPC(c, par);
  }
  if (c->getDetector(o2::detectors::DetID::TRD)) {
    configTRD(c, par);
  }
  if (c->getDetector(o2::detectors::DetID::TOF)) {
    configTOF(c, par);
  }
  c->Print("long");
  LOG(info) << "user config done";
  return 0;
}

/*
  SymNames for volumes
  ITS
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>,  X=0:6, Y=0,1
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>/ITSUStave<Z>
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>/ITSUStave<Z>/ITSUHalfStave<W>, W=0,1 starting from L3
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>/ITSUStave<Z>/ITSUHalfStave<W>/ITSUModule<T>
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>/ITSUStave<Z>/ITSUChip<C> for L0,1,2
  ITS/ITSULayer<X>/ITSUHalfBarrel<Y>/ITSUStave<Z>/ITSUHalfStave<W>/ITSUModule<T>/ITSUChip<C> for L3-6

  TRD
  TRD/sm<S>, S=00:17
  TRD/sm<S>/st<T>/pl<P>, T=0-6 (stack)

  TOF
  TOF/sm<S>, S=00:17
  TOF/sm<S</strip<T>, T=00:91
*/

void configITS(Controller* c, int par)
{
  const double kCondSig[AlignableVolume::kNDOFGeom] = {0.2, 0.2, 0.3, 1., 1., 1.}; // precondition sigmas
  AlignableDetectorITS* its = (AlignableDetectorITS*)c->getDetector(o2::detectors::DetID::ITS);
  if (!its) {
    LOG(warn) << "No ITS";
    return;
  }

  auto volITS = its->getVolume("ITS"); // envelope volume

  volITS->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ); // no auto constraint
  volITS->setFreeDOFPattern(0);                                                                                              // fix

  its->setFreeDOFPattern(0, -1, ".+ITSUChip[0-9]+$");      // Fix sensors
  its->setFreeDOFPattern(0, -1, ".+ITSUModule[0-9]+$");    // Fix modules on OB
  its->setFreeDOFPattern(0, -1, ".+ITSUHalfStave[0-9]+$"); // Fix halfstaves on OB
  //  its->setFreeDOFPattern(0, -1, ".+ITSUStave[0-9]+$");   // Fix staves
  //  its->setFreeDOFPattern(0, -1, ".+ITSUHalfBarrel[0-9]+$");   // Fix half-barrels

  for (int iv = its->getNVolumes(); iv--;) {
    auto vol = its->getVolume(iv);
    for (int idf = AlignableVolume::kNDOFGeom; idf--;) {
      if (std::abs(vol->getParErr(idf)) < 1e-6) { // there is not yet condition
        vol->setParErr(idf, kCondSig[idf]);       // set condition
      }
    }
    if (!vol->isSensor()) {
      // prevent global shift of children in the containers
      vol->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ);
    }
  }
  /*
  auto nvol = its->getNVolumes();
  for (int i=0;i<nvol;i++) {
    auto vol = its->getVolume(i);
    vol->Print();
  }
  */
}

void configTPC(Controller* c, int par)
{
  const double kCondSig[AlignableVolume::kNDOFGeom] = {1., 1., 1., 1., 1., 1.}; // precondition sigmas
  AlignableDetectorTPC* tpc = (AlignableDetectorTPC*)c->getDetector(o2::detectors::DetID::TPC);
  if (!tpc) {
    LOG(info) << "No TPC";
    return;
  }

  tpc->setFreeDOFPattern(0, -1, "TPC/sec[0-9]+$"); // Fix sectors
  auto tpcvol = tpc->getVolume("TPC_envelope");    // envelope
  if (!tpcvol || !tpcvol->isDummyEnvelope()) {
    LOG(fatal) << "Could not find TPC_envelope volume";
  }
  for (int idf = AlignableVolume::kNDOFGeom; idf--;) {
    tpcvol->setParErr(idf, kCondSig[idf]); // set condition
  }
}

void configTRD(Controller* c, int par)
{
  const double kCondSig[AlignableVolume::kNDOFGeom] = {0.2, 0.2, 0.3, 1., 1., 1.}; // precondition sigmas
  AlignableDetectorTRD* trd = (AlignableDetectorTRD*)c->getDetector(o2::detectors::DetID::TRD);
  if (!trd) {
    LOG(warn) << "No TRD";
    return;
  }

  trd->setFreeDOFPattern(0, -1, ".+/pl[0-9]+$"); // Fix chambers
  // trd->setFreeDOFPattern(0, -1, ".+/sm[0-9]+$");   // Fix sectors

  for (int iv = trd->getNVolumes(); iv--;) {
    auto vol = trd->getVolume(iv);
    for (int idf = AlignableVolume::kNDOFGeom; idf--;) {
      if (std::abs(vol->getParErr(idf)) < 1e-6) { // there is not yet condition
        vol->setParErr(idf, kCondSig[idf]);       // set condition
      }
    }
    if (!vol->isSensor()) {
      // prevent global shift of children in the containers
      vol->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ);
    }
  }
  /*
  auto nvol = trd->getNVolumes();
  for (int i=0;i<nvol;i++) {
    auto vol = trd->getVolume(i);
    vol->Print();
  }
  */
}

void configTOF(Controller* c, int par)
{
  const double kCondSig[AlignableVolume::kNDOFGeom] = {0.2, 0.2, 0.3, 1., 1., 1.}; // precondition sigmas
  AlignableDetectorTOF* tof = (AlignableDetectorTOF*)c->getDetector(o2::detectors::DetID::TOF);
  if (!tof) {
    LOG(warn) << "No TOF";
    return;
  }

  tof->setFreeDOFPattern(0, -1, ".+/strip[0-9]+$"); // Fix strips
  // tof->setFreeDOFPattern(0, -1, ".+/sm[0-9]+$");   // Fix sectors

  for (int iv = tof->getNVolumes(); iv--;) {
    auto vol = tof->getVolume(iv);
    for (int idf = AlignableVolume::kNDOFGeom; idf--;) {
      if (std::abs(vol->getParErr(idf)) < 1e-6) { // there is not yet condition
        vol->setParErr(idf, kCondSig[idf]);       // set condition
      }
    }
    if (!vol->isSensor()) {
      // prevent global shift of children in the containers
      vol->setChildrenConstrainPattern(AlignableVolume::kDOFBitTX | AlignableVolume::kDOFBitTY | AlignableVolume::kDOFBitTZ);
    }
  }
  /*
  auto nvol = tof->getNVolumes();
  for (int i=0;i<nvol;i++) {
    auto vol = tof->getVolume(i);
    vol->Print();
  }
  */
}
