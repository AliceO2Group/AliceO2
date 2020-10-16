// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file rawReaderFile.cxx 
/// \author Boris Polishchuk (Boris.Polishchuk@cern.ch)

#include <array>
#include <iostream>
#include <fstream>
#include <vector>
#include "RStringView.h"
#include <Rtypes.h>
#include "DetectorsRaw/RawFileReader.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloRawFitterStandard.h"
#include "EMCALReconstruction/AltroDecoder.h"

using namespace o2::emcal;

int main(int argc, char** argv)
{

  char *rawFileName    = argv[1];
  std::string inputDir = " ";

  o2::raw::RawFileReader reader;
  reader.setDefaultDataOrigin(o2::header::gDataOriginEMC);
  reader.setDefaultDataDescription(o2::header::gDataDescriptionRawData);
  reader.setDefaultReadoutCardType(o2::raw::RawFileReader::RORC);
  reader.addFile(rawFileName);
  reader.init();

  while (1) {
    
    int tfID = reader.getNextTFToRead();
    
    if (tfID >= reader.getNTimeFrames()) {
      std::cerr << "nothing left to read after " << tfID << " TFs read";
      break;
    }

    std::vector<char> dataBuffer; // where to put extracted data
    
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
	
        //std::cout<<rawreader.getRawHeader()<<std::endl;

        // use the altro decoder to decode the raw data, and extract the RCU trailer
        o2::emcal::AltroDecoder decoder(parser);
        decoder.decode();
	
        std::cout << decoder.getRCUTrailer() << std::endl;
	
        // Loop over all the channels
        for (auto& chan : decoder.getChannels()) {
	  
	  std::cout << "HW=" << chan.getHardwareAddress() <<  " (FEC"<< chan.getFECIndex() << "): ";
	  for(auto &bunch : chan.getBunches()) 
	    {
	      std::cout << "BunchLength=" << (int) bunch.getBunchLength() << "  StartTiime=" <<  (int) bunch.getStartTime() << "  :";
	      for(auto const e : bunch.getADC()) std::cout << e << " ";
	      std::cout << std::endl;
	    }
        }
      }
    }
    
    reader.setNextTFToRead(++tfID);
  }
}
