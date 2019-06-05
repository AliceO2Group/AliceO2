// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
  Load the macro:
  gSystem->Load("libO2TPCFastTransformation");
  gSystem->Load("libTPCReconstruction");
  .x getTPCTransformationExample.C++
*/
#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "TPCReconstruction/TPCFastTransformHelperO2.h"
#include "TH1F.h"
#include "TStyle.h"

#endif

using namespace o2::tpc;
using namespace o2::gpu;

void spaceChargeCorrection(const double XYZ[3], double dXdYdZ[3])
{
  double a = XYZ[2] / 100;
  dXdYdZ[0] = a + 0.1 * a * a;
  dXdYdZ[1] = a;
  dXdYdZ[2] = a;
}

void getTPCTransformationExample()
{

  o2::tpc::TPCFastTransformHelperO2::instance()->setSpaceChargeCorrection(spaceChargeCorrection);

  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));

  TH1F* hist = new TH1F("h", "h", 100, -1.e-4, 1.e-4);

  double statDiff = 0., statN = 0.;

  for (int slice = 0; slice < fastTransform->getNumberOfSlices(); slice += 1) {
    std::cout << "slice " << slice << " ... " << std::endl;

    const TPCFastTransform::SliceInfo& sliceInfo = fastTransform->getSliceInfo(slice);

    for (int row = 0; row < fastTransform->getNumberOfRows(); row++) {

      int nPads = fastTransform->getRowInfo(row).maxPad + 1;

      for (int pad = 0; pad < nPads; pad += 10) {

        for (float time = 0; time < 1000; time += 30) {

          fastTransform->setApplyDistortionFlag(0);
          float x0 = 0., y0 = 0., z0 = 0.;
          int err0 = fastTransform->Transform(slice, row, pad, time, x0, y0, z0);

          fastTransform->setApplyDistortionFlag(1);
          float x1 = 0., y1 = 0., z1 = 0.;
          int err1 = fastTransform->Transform(slice, row, pad, time, x1, y1, z1);

          if (err0 != 0 || err1 != 0) {
            std::cout << "can not transform!!" << std::endl;
            continue;
          }

          // local 2 global
          float gx0, gy0, gz0;
          fastTransform->convLocalToGlobal(slice, x0, y0, z0, gx0, gy0, gz0);
          float gx1, gy1, gz1;
          fastTransform->convLocalToGlobal(slice, x1, y1, z1, gx1, gy1, gz1);

          // compare the original correction to the difference ( transformation with correction - transformation without correction )

          double xyz[3] = { gx0, gy0, gz0 };
          double d[3] = { 0, 0, 0 };
          spaceChargeCorrection(xyz, d);

          hist->Fill((gx1 - gx0) - d[0]);
          hist->Fill((gy1 - gy0) - d[1]);
          hist->Fill((gz1 - gz0) - d[2]);

          //std::cout << (x1g-x0g) - d[0]<<" "<< (y1g-y0g) - d[1]<<" "<< (z1g-z0g) - d[2]<<std::endl;
        }
      }
    }
  }
  std::cout << "draw.." << std::endl;
  gStyle->SetOptStat("emruo");
  hist->Draw();
}
