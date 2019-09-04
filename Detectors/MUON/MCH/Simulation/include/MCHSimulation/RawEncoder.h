// Copyright CERN and copyright holders of ALICE O2. This software is 
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//                                                                                                                                                                    // See http://alice-o2.web.cern.ch/license for full licensing information.                                                                                            //                                                                                                                                                                    // In applying this license CERN does not waive the privileges and immunities                                                                                         // granted to it by virtue of its status as an Intergovernmental Organization                                                                                         // or submit itself to any jurisdiction.                                                                                                                              
/** @file RawEncoder.h                                                                                                                                                  * C++  MCH RawEncoder.
 * @author Michael Winn
 */

#ifndef O2_MCH_SIMULATION_RAWENCODER_H_
#define O2_MCH_SIMULATION_RAWENCODER_H_

#include "MCHSimulation/Digit.h"

namespace o2
{
namespace mch
{
 public:
  RawEncoder(int mode = 3);

  ~RawEncoder() = default;

  void init();

  //process digits: fill raw "vector" with raw format
  void process(const std::vector<Digit> digits, std::vector<char>& raw);
  //for the moment 1 char 5 bit

  
  //to be clarify integration in DPL: outputcontainers
  //
  //
 private:
  const static int mNdE = 156;//tbc, if needed


  int processDigit(const Digit& digit, std::vector<char>& raw); //tbc if anything else needed
  int intSignal(int adcsum, int timebins, int timebin);

  int timeBins();

  

}// namespace mch
}// namespace o2
 
#endif
