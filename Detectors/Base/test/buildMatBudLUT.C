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

// \file buildMatBudLUT.C
// Demo and test of the Barrel mat.budget LUT

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsBase/MatLayerCylSet.h"
#include "DetectorsBase/MatLayerCyl.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "CommonUtils/NameConf.h"
#include <TFile.h>
#include <TSystem.h>
#include <TStopwatch.h>
#endif

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

o2::base::MatLayerCylSet mbLUT;

bool testMBLUT(const std::string& lutFile = "matbud.root");

bool buildMatBudLUT(int nTst = 30, int maxLr = -1, const std::string& outFile = "matbud.root", const std::string& geomNamePrefix = "o2sim", const std::string& opts = "");

struct LrData {
  float rMin = 0.f;
  float rMax = 0.f;
  float zHalf = 0.f;
  float dZMin = 999.f;    // min Z bin
  float dRPhiMin = 999.f; // min r*phi bin

  LrData(float rMn = 0.f, float rMx = 0.f, float zHlf = 0.f, float dzMn = 9999.f, float drphMn = 9999.) : rMin(rMn), rMax(rMx), zHalf(zHlf), dZMin(dzMn), dRPhiMin(drphMn) {}
};

std::vector<LrData> lrData;
void configLayers();

bool buildMatBudLUT(int nTst, int maxLr, const std::string& outFile, const std::string& geomNamePrefix, const std::string& opts)
{
  auto geomName = o2::base::NameConf::getGeomFileName(geomNamePrefix);
  if (gSystem->AccessPathName(geomName.c_str())) { // if needed, create geometry
    std::cout << geomName << " does not exist. Will create it on the fly\n";
    std::stringstream str;
    // constructing an **unaligned** geom (Geant3 used since faster initialization) --> can be avoided by passing an existing geometry
    str << "${O2_ROOT}/bin/o2-sim-serial -n 0 -e TGeant3 --configKeyValues \"" << opts << "\" --field 0  -o " << geomNamePrefix;
    gSystem->Exec(str.str().c_str());
  }
  o2::base::GeometryManager::loadGeometry(geomNamePrefix);
  configLayers();

  if (maxLr < 1) {
    maxLr = lrData.size();
  } else {
    maxLr = std::min(maxLr, (int)lrData.size());
  }
  for (int i = 0; i < maxLr; i++) {
    auto& l = lrData[i];
    printf("L:%3d %6.2f<R<%6.2f ZH=%5.1f | dz = %6.2f drph = %6.2f\n", i, l.rMin, l.rMax, l.zHalf, l.dZMin, l.dRPhiMin);
    mbLUT.addLayer(l.rMin, l.rMax, l.zHalf, l.dZMin, l.dRPhiMin);
  }

  TStopwatch sw;
  mbLUT.populateFromTGeo(nTst);
  mbLUT.optimizePhiSlices(); // move to populateFromTGeo
  mbLUT.flatten();           // move to populateFromTGeo

  mbLUT.writeToFile(outFile);
  sw.Stop();
  sw.Print();
  sw.Start(false);
  mbLUT.dumpToTree("matbudTree.root");
  sw.Stop();
  sw.Print();
  return true;
}

//_______________________________________________________________________
bool testMBLUT(const std::string& lutFile)
{
  // test reading and creation of copies

  o2::base::MatLayerCylSet* mbr = o2::base::MatLayerCylSet::loadFromFile(lutFile);
  if (!mbr) {
    LOG(error) << "Failed to read LUT from " << lutFile;
    return false;
  }

  gSystem->RedirectOutput("matbudRead.txt", "w");
  mbr->print(true);
  gSystem->RedirectOutput(nullptr);

  if (mbLUT.isConstructed()) {
    gSystem->RedirectOutput("matbudBuilt.txt", "w");
    mbLUT.print(true);
    gSystem->RedirectOutput(nullptr);

    // compare original and built verstions
    auto diff = gSystem->Exec("diff matbudRead.txt matbudBuilt.txt");
    if (diff) {
      LOG(error) << "Difference between originally built and read from the file LUTs";
      return false;
    }
  }

  // object cloning
  o2::base::MatLayerCylSet* mbrC = new o2::base::MatLayerCylSet();
  mbrC->cloneFromObject(*mbr, nullptr);

  // check cloned object
  gSystem->RedirectOutput("matbudCloned.txt", "w");
  mbrC->print(true);
  gSystem->RedirectOutput(nullptr);
  {
    auto diff = gSystem->Exec("diff matbudCloned.txt matbudRead.txt");
    if (diff) {
      LOG(error) << "Difference between cloned and created at ActuallBuffer LUTs";
      return false;
    }
  }

  // copy to "Actual address", the object from which we make a copy remain in clean state
  {
    //>>> start of the lines needed to copy the object
    auto newBuff = new char[mbrC->getFlatBufferSize()];
    auto newObj = new char[sizeof(*mbr)];
    memcpy(newObj, mbrC, sizeof(*mbrC));
    memcpy(newBuff, mbrC->getFlatBufferPtr(), mbrC->getFlatBufferSize());
    o2::base::MatLayerCylSet* mbrA = (o2::base::MatLayerCylSet*)newObj; // !!! this is the object to use
    mbrA->setActualBufferAddress(newBuff);
    //<<< end of the lines needed to copy the object

    // check created object
    gSystem->RedirectOutput("matbudActual.txt", "w");
    mbrA->print(true);
    gSystem->RedirectOutput(nullptr);
    auto diff = gSystem->Exec("diff matbudActual.txt matbudCloned.txt");
    if (diff) {
      LOG(error) << "Difference between Cloned and created at /ActuallBuffer/ LUTs";
      return false;
    }
  }

  // copy to "Future address", the object from which we make a copy becomes dirty, need to be deleted
  {
    //>>> start of the lines needed to copy the object
    auto newBuff = new char[mbrC->getFlatBufferSize()];
    auto newObj = new char[sizeof(*mbrC)];
    auto oldBuff = mbrC->releaseInternalBuffer();
    mbrC->setFutureBufferAddress(newBuff);
    memcpy(newObj, mbrC, sizeof(*mbrC));
    memcpy(newBuff, oldBuff, mbrC->getFlatBufferSize());
    o2::base::MatLayerCylSet* mbrF = (o2::base::MatLayerCylSet*)newObj; // !!! this is the object to use
    //<<< end of the lines needed to copy the object

    delete mbrC;      // delete cloned object and its buffer which are now in a
    delete[] oldBuff; // dirty state
    mbrC = nullptr;

    // check created object
    gSystem->RedirectOutput("matbudFuture.txt", "w");
    mbrF->print(true);
    gSystem->RedirectOutput(nullptr);

    auto diff = gSystem->Exec("diff matbudFuture.txt matbudActual.txt");
    if (diff) {
      LOG(error) << "Difference between cloned at created at /FutureBuffer/ LUTs";
      return false;
    }
  }
  return true;
}

//_______________________________________________________________________
void configLayers()
{
  const int NSect = 18;

  const float kToler = 1e-3;
  float drStep = 0.f, zSpanH = 0.f, zBin = 0.f, rphiBin = 0.f, phiBin = 0.f;

  o2::itsmft::ChipMappingITS mp;
  int nStave = 0;

  //                        rMin    rMax   zHalf
  lrData.emplace_back(LrData(0.0f, 1.8f, 50.f));

  // beam pipe
  lrData.emplace_back(LrData(lrData.back().rMax, 1.92f, 50.f));
  lrData.emplace_back(LrData(lrData.back().rMax, 2.2f, 50.f));

  // ITS Inner Barrel
  drStep = 0.1;
  zSpanH = 20.;
  rphiBin = 0.2; // 0.1;
  zBin = 0.5;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 5 - kToler);

  // air space between Inner and Middle Barrels
  zSpanH = 40.;
  zBin = 5.;
  rphiBin = 2.;
  lrData.emplace_back(LrData(lrData.back().rMax, 19.0, zSpanH, zBin, rphiBin));

  //===================================================================================
  // ITS Middle Barrel
  nStave = mp.getNStavesOnLr(3); // Lr 3
  zSpanH = 55.;
  zBin = 0.5;
  drStep = 0.3;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 21.4 - kToler);

  drStep = 0.5;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 23.4 - kToler);

  nStave = mp.getNStavesOnLr(3); // Lr 4
  drStep = 0.2;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 26.2 - kToler);
  drStep = 0.5;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 29. - kToler);

  //===================================================================================

  // air space between Middle and Outer Barrels
  zSpanH = 80.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 33.5, zSpanH));

  //===================================================================================
  // ITS Outer barrel
  nStave = mp.getNStavesOnLr(5); // Lr 5
  drStep = 0.25;
  zSpanH = 80.;
  zBin = 1.;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 36. - kToler);

  drStep = 1.;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 38.5 - kToler);

  nStave = mp.getNStavesOnLr(6); // Lr 6
  drStep = 0.25;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 41. - kToler);

  drStep = 1.;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (nStave * 10);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 44. - kToler);

  //===================================================================================

  zSpanH = 100.f;
  zBin = 5.;
  lrData.emplace_back(LrData(lrData.back().rMax, 44.8, zSpanH, zBin));
  lrData.emplace_back(LrData(lrData.back().rMax, 46.2, zSpanH, zBin));
  lrData.emplace_back(LrData(lrData.back().rMax, 47.0, zSpanH, zBin));

  drStep = 2.;
  zBin = 5.;
  rphiBin = 2.;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 55. - kToler);

  zSpanH = 120.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 56.5, zSpanH));
  lrData.emplace_back(LrData(lrData.back().rMax, 60.5, zSpanH));
  lrData.emplace_back(LrData(lrData.back().rMax, 61.5, zSpanH));

  zSpanH = 150.f;
  drStep = 3.5;
  zBin = 15.;
  rphiBin = 10;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 68.5 - kToler);

  zSpanH = 250.f;
  zBin = 25.;
  rphiBin = 5;
  {
    auto rmean = (lrData.back().rMax + 76) / 2.;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 2);
    lrData.emplace_back(LrData(lrData.back().rMax, 76, zSpanH, zBin, rphiBin));
  }
  // TPC inner vessel
  // up to r = 78.5
  zSpanH = 250.f;
  zBin = 25.;
  {
    auto rmean = (lrData.back().rMax + 78.5) / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 12);
    lrData.emplace_back(LrData(lrData.back().rMax, 78.8, zSpanH, zBin, rphiBin));
  }
  //
  zSpanH = 250.f;
  zBin = 2;
  {
    auto rmean = (lrData.back().rMax + 78.5) / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 12);
    lrData.emplace_back(LrData(lrData.back().rMax, 84.5, zSpanH, zBin, rphiBin));
  }

  // TPC drum
  zSpanH = 250.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 250.0, zSpanH));

  //===============================

  // TPC outer vessel
  zSpanH = 247.f; // ignore large lumps of material at |z|>247
  rphiBin = 2.;
  zBin = 3.;
  lrData.emplace_back(LrData(lrData.back().rMax, 258., zSpanH, zBin, rphiBin));

  zSpanH = 247.f; // ignore large lumps of material at |z|>247
  rphiBin = 2.;
  zBin = 999.; // no segmentation in Z
  lrData.emplace_back(LrData(lrData.back().rMax, 280., zSpanH, zBin, rphiBin));

  // TRD

  zSpanH = 360.;
  drStep = 1;
  zBin = 10;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 12);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 370);

  // TOF
  zSpanH = 380.;
  drStep = 1;
  zBin = 10;
  rphiBin = 5.;
  do {
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 12);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 400);

  // rest
  drStep = 1;
  zBin = 10;
  rphiBin = 5.;
  do {
    zSpanH = lrData.back().rMax;
    auto rmean = lrData.back().rMax + drStep / 2;
    rphiBin = rmean * TMath::Pi() * 2 / (NSect * 12);
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 500);
}

#endif //!_COMPILED_ON_GPU_
