// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include <TFile.h>
#include <TSystem.h>
#endif

#ifndef GPUCA_ALIGPUCODE // this part is unvisible on GPU version

o2::base::MatLayerCylSet mbLUT;

bool testMBLUT(std::string lutName = "MatBud", std::string lutFile = "matbud.root");

bool buildMatBudLUT(int nTst = 30, int maxLr = -1,
                    std::string outName = "MatBud", std::string outFile = "matbud.root",
                    std::string geomName = "O2geometry.root");

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

bool buildMatBudLUT(int nTst, int maxLr, std::string outName, std::string outFile, std::string geomName)
{

  if (gSystem->AccessPathName(geomName.c_str())) { // if needed, create geometry
    gSystem->Exec("$O2_ROOT/bin/o2-sim -n 0");
    geomName = "./O2geometry.root";
  }
  o2::base::GeometryManager::loadGeometry(geomName);
  configLayers();

  if (maxLr < 1) {
    maxLr = lrData.size();
  } else {
    maxLr = std::min(maxLr, (int)lrData.size());
  }
  for (int i = 0; i < maxLr; i++) {
    auto& l = lrData[i];
    mbLUT.addLayer(l.rMin, l.rMax, l.zHalf, l.dZMin, l.dRPhiMin);
  }

  mbLUT.populateFromTGeo(nTst);
  mbLUT.optimizePhiSlices(); // move to populateFromTGeo
  mbLUT.flatten();           // move to populateFromTGeo

  mbLUT.writeToFile(outFile, outName);

  mbLUT.dumpToTree("matbudTree.root");
  return true;
}

//_______________________________________________________________________
bool testMBLUT(std::string lutName, std::string lutFile)
{
  // test reading and creation of copies

  o2::base::MatLayerCylSet* mbr = o2::base::MatLayerCylSet::loadFromFile(lutFile, lutName);
  if (!mbr) {
    LOG(ERROR) << "Failed to read LUT " << lutName << " from " << lutFile;
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
      LOG(ERROR) << "Difference between originally built and read from the file LUTs";
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
      LOG(ERROR) << "Difference between cloned and created at ActuallBuffer LUTs";
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
      LOG(ERROR) << "Difference between Cloned and created at /ActuallBuffer/ LUTs";
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
      LOG(ERROR) << "Difference between cloned at created at /FutureBuffer/ LUTs";
      return false;
    }
  }
  return true;
}

//_______________________________________________________________________
void configLayers()
{
  const float kToler = 1e-3;
  float drStep = 0.f, zSpanH = 0.f, zBin = 0.f, rphiBin = 0.f, phiBin = 0.f;

  //                           rMin    rMax   zHalf
  lrData.emplace_back(LrData(0.0f, 1.8f, 30.f));

  // beam pipe
  lrData.emplace_back(LrData(lrData.back().rMax, 1.9f, 30.f));

  // ITS Inner Barrel
  drStep = 0.1;
  zSpanH = 17.;
  rphiBin = 0.2; // 0.1;
  zBin = 0.5;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 5.2 + kToler);

  // air space between Inner and Middle Barrels
  lrData.emplace_back(LrData(lrData.back().rMax, 18.0, zSpanH));

  // ITS Middle Barrel
  drStep = 0.2;
  zSpanH = 50.;
  rphiBin = 0.5;
  zBin = 0.5;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 29. + kToler);

  // air space between Middle and Outer Barrels
  zSpanH = 80.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 33.5, zSpanH));

  // ITS Outer barrel
  drStep = 0.5;
  zSpanH = 80.;
  rphiBin = 1.;
  zBin = 1.;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 45.5 + kToler);

  // air space between Outer Barrel and shell
  zSpanH = 100.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 59.5, zSpanH));

  // Shell
  drStep = 0.5;
  zSpanH = 100.;
  rphiBin = 1.;
  zBin = 1.;
  do {
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 63. + kToler);

  // air space between Shell and TPC
  zSpanH = 250.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 76, zSpanH));

  // TPC inner vessel
  // up to r = 78.5
  zSpanH = 250.f;
  rphiBin = 1.;
  zBin = 25.;
  lrData.emplace_back(LrData(lrData.back().rMax, 78.5, zSpanH, zBin, rphiBin));

  //
  zSpanH = 250.f;
  rphiBin = 2.;
  zBin = 2;
  lrData.emplace_back(LrData(lrData.back().rMax, 84.5, zSpanH, zBin, rphiBin));

  // TPC drum
  zSpanH = 250.f;
  lrData.emplace_back(LrData(lrData.back().rMax, 250.0, zSpanH));

  // TPC outer vessel
  zSpanH = 247.f; // ignore large lumps of material at |z|>247
  rphiBin = 2.;
  zBin = 3.;
  lrData.emplace_back(LrData(lrData.back().rMax, 258., zSpanH, zBin, rphiBin));

  zSpanH = 247.f; // ignore large lumps of material at |z|>247
  rphiBin = 2.;
  zBin = 999.; // no segmentation in Z
  lrData.emplace_back(LrData(lrData.back().rMax, 280., zSpanH, zBin, rphiBin));

  drStep = 1;
  zBin = 5;
  rphiBin = 5.;
  do {
    zSpanH = lrData.back().rMax;
    lrData.emplace_back(LrData(lrData.back().rMax, lrData.back().rMax + drStep, zSpanH, zBin, rphiBin));
  } while (lrData.back().rMax < 400);
}

#endif //!_COMPILED_ON_GPU_
