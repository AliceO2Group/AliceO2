// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.h
/// \brief Implementation of the conversion from points to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZER_H_
#define ALICEO2_MFT_DIGITIZER_H_

#include "TClonesArray.h"

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/SimulationAlpide.h"

#include "MFTBase/GeometryTGeo.h"
#include "MFTSimulation/DigitContainer.h"

namespace o2 
{
  namespace MFT 
  {
    class Digitizer : public TObject
    {
      
    public:
      
      Digitizer();
      ~Digitizer() override;
      
      void init(Bool_t build = kTRUE);
      
      /// Steer conversion of points to digits
      /// @param points Container with ITS points
      /// @return digits container
      DigitContainer& process(TClonesArray* points);
      void process(TClonesArray* points, TClonesArray* digits);
      
    private:
      
      Digitizer(const Digitizer&);
      Digitizer& operator=(const Digitizer&);
      
      GeometryTGeo mGeometry;                     ///< ITS upgrade geometry
      Int_t mNumOfChips;                          ///< Number of chips
      std::vector<o2::ITSMFT::Chip> mChips;  ///< Array of chips
      std::vector<o2::ITSMFT::SimulationAlpide> mSimulations; ///< Array of chips response simulations
      DigitContainer mDigitContainer;             ///< Internal digit storage
      
      ClassDefOverride(Digitizer, 1)
	
    };
  }
}

#endif

