// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCBase/Geometry.h"
#include "FairLogger.h"

ClassImp(o2::zdc::Geometry);

using namespace o2::zdc;

 constexpr double Geometry::ZNDIMENSION[3]; //;
 constexpr double Geometry::ZNFIBRE[3]; //;
 constexpr double Geometry::ZNSECTORS[2]; //;
 constexpr double Geometry::ZNDIVISION[2]; //;
 constexpr double Geometry::ZNGROOVES[3]; // = {0.03, 0.03, 50.};
 constexpr double Geometry::ZNCPOSITION[3]; //; // = {0., 0., -11261.3};
 constexpr double Geometry::ZNAPOSITION[3]; //; // = {0., 0., 11259.8};
 constexpr double Geometry::ZNSUPPORTBASE[3]; //; // = {6.3, 4.57, 71.2}; //Basement of ZN table (thick one)
 constexpr double Geometry::ZNSUPPORTBASEPOS[3]; //; // = {0., -14., 21.2};
 constexpr double Geometry::ZNSUPPORTSCINTILH[3]; //; // = {4.32-0.8, 0.8, 50.}; //Scintillator container: top&bottom
 constexpr double Geometry::ZNSUPPORTSCINTILV[3]; //; // = {0.8, 1.955, 50.}; //Scintillator container: sides
 constexpr double Geometry::ZNSUPPORTWALLSUD[3]; //; // = {3.52,1.,50.}; //Top and bottom walls
 constexpr double Geometry::ZNSUPPORTWALLSLAT[3]; //; // = {0.4, 5.52, 50.}; //Side walls
 constexpr double Geometry::ZNFIBREDIAMETER;// 18;
//
 constexpr double Geometry::ZPDIMENSION[3]; // = {11.2, 6., 75.};
 constexpr double Geometry::ZPFIBRE[3]; // = {0., 0.0275, 75.};
 constexpr double Geometry::ZPSECTORS[2]; // = {4, 1};
 constexpr double Geometry::ZPDIVISION[2]; // = {7, 15};
 constexpr double Geometry::ZPGROOVES[3]; // = {0.04, 0.04, 75.};
 constexpr double Geometry::ZPCPOSITION[3]; // = {24.35, 0., -11253.3};
 constexpr double Geometry::ZPAPOSITION[3]; // = {24.35, 0., 11251.8};
 constexpr double Geometry::ZPSUPPORTBASE1[3]; // = {12.5, 1.4, 75.};//Bottom basement of ZP table (thinner one)
 constexpr double Geometry::ZPSUPPORTBASE1POS[3]; // = {0., -17., 0.};
 constexpr double Geometry::ZPSUPPORTBASE2[3]; // = {12.5, 2.5, 75.};//Upper basement of ZP table (thicker one)
 constexpr double Geometry::ZPSUPPORTBASE2POS[3]; // = {0., -9., 0.};
 constexpr double Geometry::ZPSUPPORTBASE3[3]; // = {1.5, 2.05, 75.};//support table heels (piedini)
 constexpr double Geometry::ZPSUPPORTWALLBOTTOM[3]; // = {11.2, 0.25, 75.}; //Bottom wall
 constexpr double Geometry::ZPSUPPORTWALLUP[3]; // = {11.2, 1., 75.}; //Top wall
// constexpr double Geometry::ZPSUPPORTWALLSLAT[3]; // = {0.5, 7.25, 75.}; //Side walls (original)
 constexpr double Geometry::ZPSUPPORTWALLSLAT[3]; // = {0.5, 6., 75.}; //Side walls (modified)
 constexpr double Geometry::ZPFIBREDIAMETER;// = 28;

//
 constexpr double Geometry::ZEMDIMENSION[6]; // = {20.62/2., 3.5, 3.5, 45., 0., 0.};
 constexpr double Geometry::ZEMFIBRE[3]; // = {0., 0.0275, 7.};
 constexpr double Geometry::ZEMDIVISION[3]; // = {92, 0, 20};

//
// using 1.4142135 instead of sqrt(2.): check!
 constexpr double Geometry::ZEMPBSLICE[6]; // = {0.15*1.4142135, 3.5, 3.5, 45., 0., 0.};
 constexpr double Geometry::ZEMVOIDLAYER[6]; // = {(20.62/20.)/2., 3.5, 3.5, 45., 0., 0.};
 constexpr double Geometry::ZEMPOSITION[3]; // = {9.69, 0., 760.};
 constexpr double Geometry::ZEMSUPPORTTABLE[3]; // =  {55./2., 1.5/2., 110./2.};
 constexpr double Geometry::ZEMSUPPORTBOX[6]; // = {10.5/2., 100./2., 95./2., 0.25/2., 2./2., 2./2.};
 constexpr double Geometry::ZEMSUPPORT1[3]; // = {15./2, 3./2., 95./2.}; //support table
 constexpr double Geometry::ZEMSUPPORT2[3]; // = {2./2, 5./2., 95./2.};  //support table heels (piedini)
 constexpr double Geometry::ZEMSUPPORT3[3]; // = {3.5, 2./2., 20./2.};   //screens around ZEM
 constexpr double Geometry::ZEMSUPPORT4[6]; // = {20./2., 3.5, 1.5/2., 45., 0., 0.}; //detector box walls (side)
 constexpr double Geometry::ZEMWALLH[3]; // = {10.5/2., /*bthickness[1]; //*/ 1., 95./2.}; //box walls
 constexpr double Geometry::ZEMWALLVFWD[3]; // = {10.5/2., (100.-2.)/2., 0.2};
 constexpr double Geometry::ZEMWALLVBKW[3]; // = {10.5/2., (100.-2.)/2., 2./2.};
 constexpr double Geometry::ZEMWALLVSIDE[3]; // = {0.25/2., (100.-2.)/2., (95.-2.)/2.};
//
 constexpr double Geometry::PIPEROTANGLE1[6]; // = {90.-1.0027, 0., 90., 90., 1.0027, 180.};
 constexpr double Geometry::PIPEROTANGLE2[6]; // = {90.+1.0027, 0., 90., 90., 1.0027, 0.};
 constexpr double Geometry::PIPEROTANGLE3[6]; // = {90.-1.8934,0.,90.,90.,1.8934,180.};
 constexpr double Geometry::PIPEROTANGLE4[6]; // = {90.-3.8,0.,90.,90.,3.8,180.};
 constexpr double Geometry::PIPEROTANGLE5[6]; // = {90.+9.8,0.,90.,90.,9.8,0.};
 constexpr double Geometry::ZNCROTANGLE[6]; // = {90., 180., 90., 90., 180., 0.};
 constexpr double Geometry::ZEMROTANGLE1[6]; // = {0.,0.,90.,90.,-90.,0.};
 constexpr double Geometry::ZEMROTANGLE2[6]; // = {180.,0.,90.,45.+90.,90.,45.};
//
 constexpr double Geometry::ZBEGINCSIDE;// = -1947.2;
 constexpr double Geometry::ZBEGINASIDE;// = 1910.2;
 constexpr double Geometry::ZENDMOTHERVOLUME;// = 13500.;
 constexpr double Geometry::ZCOMPENSATOR;// = -1974.;
 constexpr double Geometry::ZCOMPMAGFIELD;// = 1972.5;
 constexpr double Geometry::ZITMAGFIELD;// = 2296.5;
 constexpr double Geometry::ZD1MAGFIELD;// = 5838.3;
 constexpr double Geometry::ZD2MAGFIELD;// = 12167.8;
