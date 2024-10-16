#include "AliCDBManager.h"
#include "AliGeomManager.h"
#include "AliTRDgeometry.h"

#include "TGeoMatrix.h"

#include <iostream>
#include <fstream>

/*
  This macro loads the TRD geometry taking into account the alignment from Run 2
  and dumps the inverse of all cluster matrices to a file.
  To load the matrices add the following include and method to
  GeometryFlat.h:

#include <fstream>

  GPUd() bool readMatricesFromFile()
  {
    std::ifstream fIn("matrices.dat", std::ios::in | std::ios::binary);
    if (!fIn) {
      printf("Cannot read file\n");
      return false;
    }
    for (int32_t iDet = 0; iDet < constants::NCHAMBER; ++iDet) {
      float m[12];
      for (int32_t j=0; j<12; ++j) {
        fIn.read((char*) &m[j], sizeof(float));
      }
      mMatrixCache[iDet] = o2::gpu::Transform3D(m);
    }
    return true;
  }

  Don't forget to uncomment the line in createGeo.C which calls this method.

  Note that only 521 chambers are installed.

*/

void dumpTRDClusterMatrices()
{
  auto man = AliCDBManager::Instance();
  if (!man->IsDefaultStorageSet()) {
    man->SetDefaultStorage("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/2015/OCDB");
    man->SetRun(244340);
  }
  AliGeomManager::LoadGeometry();
  AliGeomManager::ApplyAlignObjsFromCDB("ITS TPC TRD TOF");

  auto geo = new AliTRDgeometry();

  std::ofstream fOut("matrices.dat", std::ios::out | std::ios::binary);
  if (!fOut) {
    printf("Cannot open file\n");
    return;
  }

  int32_t nMatrices = 0;

  for (int32_t i = 0; i < 540; ++i) {
    // dump all available matrices to a file
    auto matrix = geo->GetClusterMatrix(i);
    if (!matrix) {
      printf("No matrix for chamber %i\n", i);
      continue;
    }
    ++nMatrices;
    auto matrixO2 = matrix->Inverse();

    auto tr = matrixO2.GetTranslation();
    auto rot = matrixO2.GetRotationMatrix();

    float m[12] = {static_cast<float>(rot[0]), static_cast<float>(rot[1]), static_cast<float>(rot[2]), static_cast<float>(tr[0]), static_cast<float>(rot[3]), static_cast<float>(rot[4]), static_cast<float>(rot[5]), static_cast<float>(tr[1]), static_cast<float>(rot[6]), static_cast<float>(rot[7]), static_cast<float>(rot[8]), static_cast<float>(tr[2])};

    for (int32_t j = 0; j < 12; ++j) {
      fOut.write((char*)&m[j], sizeof(float));
    }
  }

  fOut.close();
  printf("Dumped %i matrices\n", nMatrices);
}
