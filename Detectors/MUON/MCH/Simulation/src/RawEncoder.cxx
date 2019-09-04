// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
// In applying this license CERN does not waive the privileges and immunities                                                                                         // granted to it by virtue of its status as an Intergovernmental Organization                                                                                         // or submit itself to any jurisdiction.

#include "MCHSimulation/RawEncoder.h"


#include "MCHMappingInterface/Segmentation.h"

using namespace o2::mch:


void RawEncoder::RawEncoder(int) {}

void RawEncoder::init()
{
}
//______________________________________________________________________
void RawEncoder::process(const std::vector<Digit> digits, std::vector<char>& raw){

  raw.clear();


  for (auto& digit : digits ){
    processDigit(digit, raw);
  }
  

}
//______________________________________________________________________
int RawEncoder::processDigit(const Digit& digit, std::vector<char>& raw){

  //time: check TimStamp class
  int detID = digit.getDetID(); //
  int padid = digit.getPadID();
  double adc = digit.getADC();

  //todo
  //fill header
  
  char header = 0;

  raw.emplace_back(header);
  
  //fill
  //payload:
  int timebins = timeBins();//todo generate from a distribution
  int adcsum = (int) adc;//conversion according to total number
  for (int i=0; i< timbins; ++i)
    {
      adcbin = intSignal(adcsum, timebins, i); //to be done with function
      //dummy stuff
      raw.emplace_back(adcbin);
    }
  
  
  //header
  //
  //to do: functional shape for a given sum of adc

  // need to sample first from a function the distribution of
  //time duration of a signal, for the first trial, just one signal

  //second step: for a given time duration (e.g. bin number) and a given ADC
  //sum, generate counts from a distribution.
  //easiest: have a function with these two parameters, that is integrated
  //for each bin,
  // these two parameters
  
  
  return raw.size();

}
//______________________________________________________________________
int RawEncoder::intSignal(int adcsum, int timebins, int timebin){

  //uniform distribution as dummy, time independent
  return adcsum/timebins;

}
//______________________________________________________________________
int RawEncoder::timeBins(){

  return 20;
}



