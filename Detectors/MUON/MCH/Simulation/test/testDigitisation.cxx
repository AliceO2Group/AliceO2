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
#include <boost/timer/timer.hpp>
#include <memory>
#include <vector>
#include "MCHSimulation/Digit.h"
#include "MCHSimulation/Digitizer.h"
#include "MCHSimulation/Hit.h"


#include "MCHMappingInterface/Segmentation.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"


namespace o2
{
namespace mch
{

/// \brief Test of the Digitization
/// A couple of values are filled into a Hits and we check whether we get reproducible output in terms of digits
/// and MClabels

  
BOOST_AUTO_TEST_CASE(Digitizer_test1)
{
  
  Digitizer digitizer;
  //test amplification
  
 
  //need to produce hits with proper MC labels
  int trackId1 = 0;
  int trackId2 = 1;
  short detElemId1 = 101;//check what to put
  short detElemId2 = 1012;
  Point3D<float> entrancePoint1(-17.7993, 8.929883, -522.201); //x,y,z coordinates in cm
  Point3D<float> exitPoint1(-17.8136, 8.93606, -522.62);
  Point3D<float> entrancePoint2(-49.2793, 28.8673, -1441.25);
  Point3D<float> exitPoint2(-49.2965, 28.8806, -1441.75);
  float eloss1 = 1e-6;
  float eloss2 =1e-6;
  float length = 0.f;//no ida what it is good for
  float tof = 0.0;//not used

  //could also check to give the same input and see whether I get the same output as well
  std::vector<Hit> hits(2);
  vector.at(0) = Hit(trackId1, detElemId1, entrancePoint1, exitPoint1, eloss1, length, tof);//put some values
  vector.at(1) = Hit(trackId2, detElemId2, entrancePoint2, exitPoint2, eloss2, length, tof);//put some values
  // one hit per station, if feasible and energy deposition such that from 1 to 4 pad digits all included
  //
  //test first only single processHit
  MCTruthContainer<o2::MCCompLabel> mctruthcontainer;  
  std::vector<Digit> digits;
  mapping::Segmentation seg1{ detElemId1 };
  mapping:: Segmentation seg2{ detElemId2 };
  digitizer.process(hits, &digits);
  digitizer.provideMC(&mctruthcontainer);
  //todo do something with mctruth
  //digit members: 
  //retrieve information from digits: getPad(), getADC(), getLabel()
  //compare Hit
  int digitcounter1 = 0;
  int digitcounter2 = 1;
  
  for (auto& digit : digits) {
    
    int padid = digit.getPad();
    int adc = digit.getADC();
    int label = digit.getLabel();
    
    if(label == trackId1)
      {
	bool check = seg1.isValid(digit.getPad());// is pad ID unique across full detector?
	if (!check)   BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
	double padposX = seg1.padPositionX(padid);
	double padsizeX = seg1.padSizeX(padid);
	double padposY = seg1.padPositionY(padid);
	double padsizeY = seg1.padSizeY(padid);
	
	BOOST_CHECK_CLOSE(entrancePoint1.x(), padposX, padsizeX*4.0 );
	BOOST_CHECK_CLOSE(entrancePoint1.y(), padposY, padsizeY*4.0 );
	
	digitcounter1++;
      } else if (label == trackId2)
      {
	bool check = seg2.isValid(digit.getPad());
	if (!check)   BOOST_FAIL(" digit-pad not belonging to hit det-element-ID ");
        double padposX = seg2.padPositionX(padid);
        double padsizeX = seg2.padSizeX(padid);
        double padposY = seg2.padPositionY(padid);
        double padsizeY = seg2.padSizeY(padid);

        BOOST_CHECK_CLOSE(entrancePoint2.x(), padposX, padsizeX*4.0 );
        BOOST_CHECK_CLOSE(entrancePoint2.y(), padposY, padsizeY*4.0 );
	digitcounter2++;
	
      } else
      {
	//some boost functionality, if not one of two values?, not found at https://www.boost.org/doc/libs/1_66_0/libs/test/doc/html/index.html
	BOOST_FAIL(" MC-labels not matching between hit and digit ");
      };
    
  }

  if ( digitcounter1==0 ) BOOST_FAIL(" no digit at all from hit in station 1 ");
  //how dead maps are considered? Are there already put to 0 in the beginning or later?
  // upper bound as well? 5-10 digits too much for one hit?
  if ( digitcounter1>9 ) BOOST_FAIL("more than 10 digits for one hit in station 1 ");
  if ( digitcounter2==0 ) BOOST_FAIL(" no digit at all from hit in station 2 ");
  if (digitcounter2>9 ) BOOST_FAIL(" more than 10 digits for one hit in station 2 ");
  //how dead maps are considered? Are there already put to 0 in the beginning or later?
  
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

  
  //Control and timing of digits
  //no MC truth for the moment
  
  double adcbefmerge = digits.at(0).getADC();
  Timing_test(&digits);
  BOOST_CHECK_CLOSE(digits.at(0).getADC(), adcbefmerge*2.2);
  
}//testing 

  void Timing_test(std::vector<Digit>& digits){

    boost::timer::auto_cpu_timer t;
    
    std::cout << "digits.size() before adding " <<   digits.size() << std::endl;
    digits.emplace_back(digits.at(0).getTimeStamp(), digits.at(0).getPad(), digits.at(0).getADC() * 1.2);
    std::cout << "digits.size() after adding " <<   digits.size() << std::endl;
    digitizer.mergeDigits(digits);
    std::cout << "digits.size() after merging " <<   digits.size() << std::endl;
    return;
  }
  
}//namespace mch
}//namespace o2
