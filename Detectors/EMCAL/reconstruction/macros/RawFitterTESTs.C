// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include "RStringView.h"
#include <Rtypes.h>
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/AltroDecoder.h"
//#include "EMCALReconstruction/RawHeaderStream.h"
#endif

using namespace o2::emcal;

/// \brief Testing the standard raw fitter on run2 to run3 converted data
void RawFitterTESTs()
{

  const Int_t NoiseThreshold = 3;

  // Use the RawReaderFile to read the raw data file
  const char* aliceO2env = std::getenv("O2_ROOT");
  std::string inputDir = " ";
  if (aliceO2env)
    inputDir = aliceO2env;
  inputDir += "/share/Detectors/EMC/files/";

  o2::emcal::RawReaderFile<o2::header::RAWDataHeaderV4> rawreader(inputDir + "emcal.raw");

  // define the standard raw fitter
  o2::emcal::CaloRawFitterStandard RawFitter;
  RawFitter.setAmpCut(NoiseThreshold);
  RawFitter.setL1Phase(0.);

  // loop over all the DMA pages
  while (rawreader.hasNext()) {

    rawreader.next();
    cout << "next page \n";

    //std::cout<<rawreader.getRawHeader()<<std::endl;

    // use the altro decoder to decode the raw data, and extract the RCU trailer
    o2::emcal::AltroDecoder<decltype(rawreader)> decoder(rawreader);
    decoder.decode();

    std::cout << decoder.getRCUTrailer() << std::endl;

    // Loop over all the channels
    for (auto& chan : decoder.getChannels()) {

      // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
      o2::emcal::CaloFitResults fitResults = RawFitter.evaluate(chan.getBunches(), 0, 0);

      // print the fit output
      std::cout << "The Time is : " << fitResults.getTime() << " And the Amplitude is : " << fitResults.getAmp() << std::endl;
    }
  }
}
