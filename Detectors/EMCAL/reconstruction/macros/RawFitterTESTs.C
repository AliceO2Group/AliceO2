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
#include "DetectorsRaw/RawFileReader.h"
#include "DetectorsRaw/RDHUtils.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/AltroDecoder.h"
#include "EMCALReconstruction/CaloRawFitterGamma2.h"
//#include "EMCALReconstruction/RawHeaderStream.h"
#endif

using namespace o2::emcal;

/// \brief Testing the standard raw fitter on run2 to run3 converted data
void RawFitterTESTs(const char* filename = "")
{

  const Int_t NoiseThreshold = 3;

  std::string inputfile = filename;
  if (!inputfile.length()) {
    // Use the RawReaderFile to read the raw data file
    const char* aliceO2env = std::getenv("O2_ROOT");
    std::string inputDir = " ";
    if (aliceO2env)
      inputDir = aliceO2env;
    inputDir += "/share/Detectors/EMC/files/";
    inputfile = inputDir + "emcal.raw";
  }
  std::cout << "Using input file " << inputfile << std::endl;

  o2::raw::RawFileReader reader;
  reader.setDefaultDataOrigin(o2::header::gDataOriginEMC);
  reader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  reader.setDefaultReadoutCardType(o2::raw::RawFileReader::RORC);
  reader.addFile(inputfile.data());
  reader.init();

  // define the standard raw fitter
  //o2::emcal::CaloRawFitterStandard RawFitter;
  o2::emcal::CaloRawFitterGamma2 RawFitter;
  RawFitter.setAmpCut(NoiseThreshold);
  RawFitter.setL1Phase(0.);

  while (1) {
    int tfID = reader.getNextTFToRead();
    if (tfID >= reader.getNTimeFrames()) {
      std::cerr << "nothing left to read after " << tfID << " TFs read" << std::endl;
      break;
    }
    std::vector<char> dataBuffer; // where to put extracted data
    std::cout << "Next iteration: Number of links: " << reader.getNLinks() << std::endl;
    for (int il = 0; il < reader.getNLinks(); il++) {
      auto& link = reader.getLink(il);
      std::cout << "Decoding link " << il << std::endl;

      auto sz = link.getNextTFSize(); // size in bytes needed for the next TF of this link
      dataBuffer.resize(sz);
      link.readNextTF(dataBuffer.data());

      // Parse
      o2::emcal::RawReaderMemory parser(dataBuffer);
      while (parser.hasNext()) {
        parser.next();
        std::cout << "next page \n";
        if (o2::raw::RDHUtils::getFEEID(parser.getRawHeader()) >= 40)
          continue;

        //std::cout<<rawreader.getRawHeader()<<std::endl;

        // use the altro decoder to decode the raw data, and extract the RCU trailer
        o2::emcal::AltroDecoder decoder(parser);
        std::cout << "Decoding" << std::endl;
        decoder.decode();

        std::cout << decoder.getRCUTrailer() << std::endl;
        std::cout << "Found number of channels: " << decoder.getChannels().size() << std::endl;

        // Loop over all the channels
        for (auto& chan : decoder.getChannels()) {
          std::cout << "processing next channel idx " << chan.getChannelIndex() << ", " << chan.getHardwareAddress() << std::endl;
          // define the conatiner for the fit results, and perform the raw fitting using the stadnard raw fitter
          continue;
          o2::emcal::CaloFitResults fitResults = RawFitter.evaluate(chan.getBunches(), 0, 0);

          // print the fit output
          std::cout << "The Time is : " << fitResults.getTime() << " And the Amplitude is : " << fitResults.getAmp() << std::endl;
        }
      }
    }
    reader.setNextTFToRead(++tfID);
  }
}
