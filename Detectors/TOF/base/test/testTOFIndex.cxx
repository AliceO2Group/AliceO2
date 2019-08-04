// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TOFIndex
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "FairLogger.h" // for FairLogger
#include "TOFBase/Geo.h"
#include <TRandom.h>
#include <TStopwatch.h>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace o2::tof;

BOOST_AUTO_TEST_CASE(testTOFIndex)
{
  BOOST_TEST_CHECKPOINT("Starting");
  Int_t indextof[5] = {0};

  Bool_t ErrorSe = 0;
  Bool_t ErrorPl = 0;
  Bool_t ErrorSt = 0;
  Bool_t ErrorPx = 0;
  Bool_t ErrorPz = 0;

  for (Int_t i = 0; i < Geo::NSECTORS; i++) { // Loop on all Sectors
    indextof[0] = i;
    for (Int_t j = 0; j < Geo::NPLATES; j++) { // Loop on all Plates
      indextof[1] = j;
      //
      const Int_t nStrips =
        j < 2
          ? Geo::NSTRIPB
          : j > 2 ? Geo::NSTRIPC
                  : Geo::NSTRIPA; // Define the numer of strips of the plate
      if (j == 2 &&
          (i == 15 || i == 14 || i == 13)) // Skip sectors without A plate
        continue;
      //
      for (Int_t k = 0; k < nStrips; k++) { // Loop on all Strips
        indextof[2] = k;

        for (Int_t l = 0; l < Geo::NPADZ; l++) { // Loop on all Pads Z
          indextof[3] = l;
          for (Int_t m = 0; m < Geo::NPADX; m++) { // Loop on all Pads X
            indextof[4] = m;
            Int_t chan = o2::tof::Geo::getIndex(indextof);
            Int_t indextofTest[5] = {0};

            o2::tof::Geo::getVolumeIndices(chan, indextofTest);
            BOOST_CHECK_MESSAGE(indextof[0] == indextofTest[0],
                                "Different Sector ==> in:" << indextof[0]
                                                           << " --> out:"
                                                           << indextofTest[0]);
            BOOST_CHECK_MESSAGE(indextof[1] == indextofTest[1],
                                "Different Plate ==> in:" << indextof[1]
                                                          << " --> out:"
                                                          << indextofTest[1]);
            BOOST_CHECK_MESSAGE(indextof[2] == indextofTest[2],
                                "Different Strip ==> in:" << indextof[2]
                                                          << " --> out:"
                                                          << indextofTest[2]);
            BOOST_CHECK_MESSAGE(indextof[3] == indextofTest[3],
                                "Different PadZ ==> in:" << indextof[3]
                                                         << " --> out:"
                                                         << indextofTest[3]);
            BOOST_CHECK_MESSAGE(indextof[4] == indextofTest[4],
                                "Different PadX ==> in:" << indextof[4]
                                                         << " --> out:"
                                                         << indextofTest[4]);
            // if (locError && j == 1 && k == 3) {
            LOG(INFO) << "in:" << indextof[0] << ", " << indextof[1] << ", "
                      << indextof[2] << ", " << indextof[3] << ", "
                      << indextof[4] << " --> out:" << indextofTest[0] << ", "
                      << indextofTest[1] << ", " << indextofTest[2] << ", "
                      << indextofTest[3] << ", " << indextofTest[4]
                      << " (ch=" << chan << ")" << FairLogger::endl;
            // }
          }
        }
      }
    }
  }
  BOOST_TEST_CHECKPOINT("Ending");
}
