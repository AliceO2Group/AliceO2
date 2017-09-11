// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \brief Implementation of the conversion from hits to digits (ITS)
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZER_H_
#define ALICEO2_MFT_DIGITIZER_H_

#include "TClonesArray.h"

#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/DigitContainer.h"
#include "ITSMFTSimulation/Hit.h"

#include "MFTBase/GeometryTGeo.h"

namespace o2 
{
  namespace MFT 
  {

    class Digitizer : public TObject
    {
      
    public:
      
      Digitizer();
      ~Digitizer() override;
      Digitizer(const Digitizer&) = delete;
      Digitizer& operator=(const Digitizer&) = delete;
      
      void init(Bool_t build = kTRUE);
      
      /// Steer conversion of hits to digits
      /// @param points Container with MFT hits
      /// @return digits container
      o2::ITSMFT::DigitContainer& process(TClonesArray* hits);
      void process(TClonesArray* hits, TClonesArray* digits);

    private:
      
      GeometryTGeo mGeometry;                     ///< MFT geometry helper
      std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      o2::ITSMFT::DigitContainer mDigitContainer;             ///< Internal digit storage
      
      ClassDefOverride(Digitizer, 1)
	
    };
  }
}

#endif

