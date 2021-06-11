// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTRDGeometry.cxx
/// \brief This task tests the Geometry
/// \author Sean Murray, murrays@cern.ch

#define BOOST_TEST_MODULE Test TRD_Digit
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <numeric>

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{

using namespace o2::trd::constants;

void testDigitDetRowPad(Digit& test, int det, int row, int pad)
{
  BOOST_CHECK(test.getPad() == pad);
  BOOST_CHECK(test.getRow() == row);
  BOOST_CHECK(test.getDetector() == det);
}

void testDigitDetROBMCM(Digit& test, int det, int rob, int mcm, int channel)
{
  BOOST_CHECK(test.getMCM() == mcm);
  BOOST_CHECK(test.getROB() == rob);
  BOOST_CHECK(test.getChannel() == channel);
  BOOST_CHECK(test.getDetector() == det);
}

BOOST_AUTO_TEST_CASE(TRDDigit_test)
{
  //TPD
  //pg 14 for rob to chamber
  //
  // 540 read out chambers (detector)
  // each one is made up of 16 row of 144 pads.
  // each one is also made up of 8 or 6 read out boards comprising 16 mcm and 21 adc each.
  // we need to check the pad,row to rob,mcm and back and the inverse holds true.
  // also the boundaries hold true.ends, of readout boards, ends of mcm.
  //
  //check digit at bottom of row is correctly assigned.

  // a pad row spans 2 read out boards. with 4 mcm in each.
  // i.e. pad row 0 will have read out board 0 and 1 and mcm 0-4 and 0-4 in each making up the 8 mcm in the pad row.
  // channel 0 and 1 of MCM n are shared with 18 and 19 (respectively) of MCM n+1 and channel 20 of MCM n is shared with MCM n-1 channel 2
  // channel 20 of MCM n is connected to the preceding MCM's highest number pad. i.e. MCM01 channel 20 is connected to MCM00 pad 17 (18th pad) of row.

  // so check various combinations of that, mostly just the boundaries.
  Digit first(0, 0, 0); //det row pad
  BOOST_CHECK(first.getMCM() == 0);

  Digit last(MAXCHAMBER - 1, NROWC1 - 1, NCOLUMN - 1); // last det row and pad
  BOOST_CHECK(last.getMCM() == NMCMROB - 1);
  // end of first mcm
  Digit a(0, 0, NCOLMCM - 1);
  BOOST_CHECK(a.getMCM() == 0);
  // start of new mcm?
  Digit b(0, 0, NCOLMCM);
  BOOST_CHECK(b.getMCM() == 1);
  // last pad connected to start of new mcm?
  Digit c(0, 0, 89);
  BOOST_CHECK(c.getMCM() == 0);
  // last pad connected to start of new mcm?
  Digit d(0, 0, 90);
  BOOST_CHECK(d.getMCM() == 1);
  // now to test if we set the rob and mcm do we get the correct pad and row.
  // using the reciprical of the values above for simplicity.
  //
  //test block 1.
  Digit e(0, 0, 0, 0);
  //first channel of the first mcm, this is in fact the 19 pad of the first row, and connected to the 18th adc of the second trap ...
  Digit f(0, e.getRow(), e.getPad()); // createa digit based on the above digits pad and row.
  // we *shoulud* end up with a rob:mcm of 0:1 and channel 18
  testDigitDetROBMCM(f, 0, 0, 1, NCOLMCM);

  Digit g(0, 0, NCOLMCM - 1); // row 0 pad 17 --- should be mcm 0 and channel 2
  testDigitDetROBMCM(g, 0, 0, 0, 2);

  Digit h(0, 0, 0, 2);
  testDigitDetRowPad(h, 0, 0, NCOLMCM - 1);

  //test block2 repeat block1 but at the edge of a rob boundary i.e. going from row0 the 72nd pad to 73rd. Spanning the half of 144(NCOLUMN)
  Digit i(0, 0, (NCOLUMN / 2) - 1);
  testDigitDetROBMCM(i, 0, 0, 3, 2);
  //check the reverse creation
  Digit k(0, 0, 3, 2);
  testDigitDetRowPad(k, 0, 0, (NCOLUMN / 2) - 1);

  Digit j(0, 0, NCOLUMN / 2);
  testDigitDetROBMCM(j, 0, 1, 0, 19);
  //check the reverse creation
  Digit l(0, 1, 0, 19);
  testDigitDetRowPad(l, 0, 0, NCOLUMN / 2);

  // now repeat the same for another part of the first chamber, middle rows
  //
  Digit m(0, 12, (NCOLUMN / 2) - 1);
  testDigitDetROBMCM(m, 0, 6, 3, 2);
  //check the reverse creation
  Digit n(0, 6, 3, 2);
  testDigitDetRowPad(n, 0, 12, (NCOLUMN / 2) - 1);

  Digit o(0, 12, NCOLMCM - 1);
  testDigitDetROBMCM(o, 0, 6, 0, 2);
  Digit p(0, 6, 0, 2);
  testDigitDetRowPad(p, 0, 12, NCOLMCM - 1);

  //and now for the last row.
  Digit q(0, 15, (NCOLUMN / 2) - 1);
  testDigitDetROBMCM(q, 0, 6, 15, 2);
  //check the reverse creation
  Digit r(0, 6, 3, 2);
  testDigitDetRowPad(r, 0, 12, (NCOLUMN / 2) - 1);

  Digit s(0, 15, NCOLMCM - 1);
  testDigitDetROBMCM(s, 0, 6, 12, 2);
  Digit t(0, 6, 0, 2);
  testDigitDetRowPad(t, 0, 12, NCOLMCM - 1);

  // as a last check that for detector changes.
  //
  Digit u(1, 15, (NCOLUMN / 2) - 1);
  testDigitDetROBMCM(u, 1, 6, 15, 2);
  //check the reverse creation
  Digit v(1, 6, 3, 2);
  testDigitDetRowPad(v, 1, 12, (NCOLUMN / 2) - 1);

  Digit w(10, 15, NCOLMCM - 1);
  testDigitDetROBMCM(w, 10, 6, 12, 2);
  Digit x(10, 6, 0, 2);
  testDigitDetRowPad(x, 10, 12, NCOLMCM - 1);

  /*
  * The below is left in, it helps to remove confusion when debugging
 for(int rob=0;rob<8;rob++)for(int mcm=0;mcm<16;mcm++)for(int channel=0;channel<21;channel++){
  std::cout << "Digit e(0,"<<rob<<"," << mcm <<","<< channel<<");" << std::endl;
  Digit e(0,rob,mcm,channel);
    std::cout << " e is " << e.getRow() << " " << e.getPad();
    std::cout << " for an rob:mcm combo of " << e.getROB() << ":"<< e.getMCM() << " and adc channel:" << e.getChannel() <<std::endl;
  std::cout << "Digit f(0,e.getRow(),e.getPad())" << std::endl;;
  Digit f(0,e.getRow(),e.getPad());
    std::cout << " f is " << f.getRow() << " " << f.getPad();
    std::cout << " for an rob:mcm combo of " << f.getROB() << ":"<< f.getMCM() << " and adc channel:" << f.getChannel() <<std::endl;
    std::cout << "*********************************************************************" << std::endl;
  }
  */

  //now check that the timebins get correctly assigned on instantiation
  ArrayADC data;
  std::iota(data.begin(), data.end(), 42); // 42 for my personal amusement.
  Digit z(10, 15, NCOLMCM - 1, data);
  testDigitDetROBMCM(z, 10, 6, 12, 2);
  //test adc values are true.
  BOOST_CHECK(z.getADC()[4] == 46); // 4th time bin should be 46;
  BOOST_CHECK(z.getADC()[6] == 48); // 6th time bin should be 48;

  Digit za(10, 6, 0, 2, data);
  testDigitDetRowPad(za, 10, 12, NCOLMCM - 1);
  BOOST_CHECK(za.getADC()[14] == 56); // 14th time bin should be 56;
  BOOST_CHECK(za.getADC()[16] == 58); // 16th time bin should be 58;
}

} // namespace trd
} // namespace o2
