// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.h
/// \brief Container of digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITCONTAINER
#define ALICEO2_MFT_DIGITCONTAINER

namespace o2
{
  namespace ITSMFT
  {
    class Digit;
  }
}

namespace o2
{
  namespace MFT
  {
    class DigitContainer
    {
      
    public:

      DigitContainer() {}
      ~DigitContainer() = default;

      void reset();
      o2::ITSMFT::Digit* addDigit();
      o2::ITSMFT::Digit* getDigit();
      
      void fillOutputContainer(TClonesArray* output);

    private:

      ClassDef(DigitContainer,1);

    };
  }
}

#endif
      
