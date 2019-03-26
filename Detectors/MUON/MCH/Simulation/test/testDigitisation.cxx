// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \file testDigitisation.cxx
/// \brief This task tests the Digitizer and the Response of the MCH digitization
/// \author Michael Winn, DPhN/IRFU/CEA, michael.winn@cern.ch

#define BOOST_TEST_MODULE Test MCH Digitization
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
//#include <boost/timer/timer.hpp>
#include <memory>
#include <vector>
#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/Hit.h"

#include "TGeoManager.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/GeometryTest.h"

#include "MCHMappingInterface/Segmentation.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

 struct GEOMETRY {
  GEOMETRY()
  {
    if (!gGeoManager) {
      o2::mch::test::createStandaloneGeometry();
    }
  }
 };



namespace o2
{
namespace mch
{

/// \brief Test of the Digitization
/// A couple of values are filled into a Hits and we check whether we get reproducible output in terms of digits
/// and MClabels

void Timing_test(std::vector<Digit>& digits, Digitizer digitizer);
  

  
  
BOOST_AUTO_TEST_CASE(Digitizer_test1)
{
  GEOMETRY();
  Digitizer digitizer;
   
  //could generate a lot of hits like that station by station
  int trackId1 = 0;
  int trackId2 = 1;
  short detElemId1 = 101;
  short detElemId2 = 1012;
  Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); //x,y,z coordinates in cm
  Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
  Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
  Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);
  float eloss1 = 1e-6;
  float eloss2 =1e-6;
  float length = 0.f;
  float tof = 0.0;

    std::vector<Hit> hits(2);
  hits.at(0) = Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss1, length, tof);
  hits.at(1) = Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss2, length, tof);

  
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mctruthcontainer;  
  std::vector<Digit> digits;
  mapping::Segmentation seg1{ detElemId1 };
  mapping:: Segmentation seg2{ detElemId2 };
  digitizer.process(hits, digits);
  digitizer.provideMC(mctruthcontainer);

  
  int digitcounter1 = 0;
  int digitcounter2 = 1;
  int count = 0;
  
  for (auto& digit : digits) {
    
    int padid = digit.getPadID();
    int adc = digit.getADC();
    auto label =   (mctruthcontainer.getLabels(count))[0];
    int trackID = label.getTrackID();
    ++count;
    
    if(trackID == trackId1)
      {
	bool check = seg1.isValid(digit.getPadID());
	if (!check)   BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
	double padposX = seg1.padPositionX(padid);
	double padsizeX = seg1.padSizeX(padid);
	double padposY = seg1.padPositionY(padid);
	double padsizeY = seg1.padSizeY(padid);
	auto t = o2::mch::getTransformation(detElemId1, *gGeoManager);

	Point3D<float> pos(hits.at(0).GetX(), hits.at(0).GetY(), hits.at(0).GetZ());
	Point3D<float> lpos;
	t.MasterToLocal(pos, lpos);
	
	BOOST_CHECK_CLOSE(lpos.x(), padposX, padsizeX*4.0 );
	BOOST_CHECK_CLOSE(lpos.y(), padposY, padsizeY*10.0 );
	//non uniform pad sizes?
	
	digitcounter1++;
      } else if (trackID == trackId2)
      {
	bool check = seg2.isValid(digit.getPadID());
	if (!check)   BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
        double padposX = seg2.padPositionX(padid);
        double padsizeX = seg2.padSizeX(padid);
        double padposY = seg2.padPositionY(padid);
        double padsizeY = seg2.padSizeY(padid);
	auto t = o2::mch::getTransformation(detElemId2, *gGeoManager);

        Point3D<float> pos(hits.at(1).GetX(), hits.at(1).GetY(), hits.at(1).GetZ());
        Point3D<float> lpos;
        t.MasterToLocal(pos, lpos);


        BOOST_CHECK_CLOSE(lpos.x(), padposX, padsizeX*4.0 );
        BOOST_CHECK_CLOSE(lpos.y(), padposY, padsizeY*10.0 );
	digitcounter2++;
	
      } else
      {

	BOOST_FAIL(" MC-labels not matching between hit and digit ");
      };
    
  }

  if ( digitcounter1==0 ) BOOST_FAIL(" no digit at all from hit in station 1 ");
  if ( digitcounter1>9 ) BOOST_FAIL("more than 10 digits for one hit in station 1 ");
  if ( digitcounter2==0 ) BOOST_FAIL(" no digit at all from hit in station 2 ");
  if (digitcounter2>9 ) BOOST_FAIL(" more than 10 digits for one hit in station 2 ");
  
  //what to test:
  //1) compare label of hit and of MCtruthContainer, certainly makes sense: to be done
  /*
void Digitizer::provideMC(o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer)
   */
  
  //1) check condition that equal or more than 1 digit per hit: ok, but not very stringent
  //2) compare position of digit with hit position within a certain precision?
  // (would rely on segmentation to map back from pad number to position, not really this unit)
  //3) something with charge size? Not really useful.
  //could also check charge summation for different hits acting on same pad...
  //should one introduce member constants and getters to display intermediate steps?
  //https://www.boost.org/doc/libs/1_54_0/libs/timer/doc/cpu_timers.html
  
  //Control and timing of merging 
  //no MC truth for the moment  
  double adcbefmerge = digits.at(0).getADC();
  double adcbefmerge2 = digits.at(1).getADC();
  int digitssize = digits.size();
  Timing_test(digits, digitizer);
  BOOST_CHECK_CLOSE(digits.at(0).getADC(), adcbefmerge*10.0, adcbefmerge/10000.);
  BOOST_CHECK_CLOSE(digits.at(1).getADC(), adcbefmerge2*10.0, adcbefmerge2/10000.);

  BOOST_CHECK_CLOSE((float)digitssize, (float)digits.size(), 0.1);
}//testing 

  void Timing_test(std::vector<Digit>& digits, Digitizer digitizer){

    //boost::timer::auto_cpu_timer t;
    auto timestamp0 = digits.at(0).getTimeStamp();
    auto padid0 = digits.at(0).getPadID();
    auto adc0 = digits.at(0).getADC();
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);
    digits.emplace_back(timestamp0, padid0, adc0);

    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());
    digits.emplace_back(digits.at(1).getTimeStamp(), digits.at(1).getPadID(), digits.at(1).getADC());

    
    digitizer.mergeDigits(digits);
    return;
  }
  
}//namespace mch
}//namespace o2
