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
/// \brief Definition of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitContainer_H_
#define ALICEO2_TPC_DigitContainer_H_

#include "TPCSimulation/DigitMC.h"
#include "TPCBase/CRU.h"
#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCSimulation/Constants.h"

class TClonesArray;

namespace o2 {
namespace TPC {

/// \class DigitContainer
/// This is the base class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the CRU containers.

class DigitContainer{
  public:
    
    /// Default constructor
    DigitContainer();

    /// Destructor
    ~DigitContainer() = default;

    void reset();

    /// Get the size of the container
    /// \return Size of the CRU container
    size_t getSize() const {return mCRU.size();}

    /// Get the number of entries in the container
    /// \return Number of entries in the CRU container
    int getNentries() const;

    /// Add digit to the container
    /// \param hitID MC Hit ID
    /// \param cru CRU of the digit
    /// \param row Pad row of digit
    /// \param pad Pad of digit
    /// \param timeBin Time bin of the digit
    /// \param charge Charge of the digit
    void addDigit(size_t hitID, int cru, int timeBin, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    void fillOutputContainer(TClonesArray *output, int eventTime=0, bool isContinuous=true);

    /// Fill output TClonesArray
    /// \param output Output container
    void fillOutputContainer(TClonesArray *output, std::vector<CommonMode> &commonModeContainer);

    /// Process Common Mode Information and sum up all charges in a given CRU and time bin
    /// \param output Output container
    void processCommonMode(std::vector<CommonMode> &);

  private:
    std::array<std::unique_ptr<DigitCRU> , CRU::MaxCRU> mCRU;   ///< CRU Container for the ADC value
};

inline
DigitContainer::DigitContainer()
  : mCRU()
{}


inline
void DigitContainer::reset() 
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->reset();
  }
  std::fill(mCRU.begin(),mCRU.end(), nullptr);
}

inline
int DigitContainer::getNentries() const
{
  int counter = 0;
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitContainer_H_
