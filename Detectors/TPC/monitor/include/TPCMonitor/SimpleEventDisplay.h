// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_SIMPLEEVENTDISPLAY_H_
#define ALICEO2_TPC_SIMPLEEVENTDISPLAY_H_

/// \file   SimpleEventDisplay.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de


#include "THnSparse.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"

class TH2D;

namespace o2
{
namespace TPC
{

class Mapper;
/// \class SimpleEventDisplay
/// \brief Base of a simple event display for digits
///
/// This class is a base for a simple event display
/// It processes raw data and saves digit information for pad and row.
///
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
class SimpleEventDisplay : public CalibRawBase
{
  public:
    SimpleEventDisplay();
    
    ~SimpleEventDisplay() override = default;

    Int_t updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final;
  
    /// not used
    Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final { return 0;}

    CalPad* getCalPadMax() {return &mPadMax;}

    /// Set currently selected sector
    void setSelectedSector(Int_t selectedSector) { mSelectedSector=selectedSector; }

    /// Set last processed sector
    void setLastSector(Int_t lastSector) { mLastSector=lastSector; }

    void setPedstals(CalPad* pedestals) { mPedestals = pedestals; }
  //   TH1D* MakePadSignals(Int_t roc, Int_t channel);
    TH1D* MakePadSignals(Int_t roc, Int_t row, Int_t pad);

    /// Dummy end event
    void endEvent() final {};

  private:
    THnSparseS  *mHnDataIROC;      //!< Event Data IROCs
    THnSparseS  *mHnDataOROC;      //!< Event Data OROCs
    CalPad       mPadMax;          //!< Cal Pad with max Entry per channel
    TH2D        *mHSigIROC;        //!< iroc signals
    TH2D        *mHSigOROC;        //!< oroc signals
    CalPad*   mPedestals;          //!< Pedestal calibratino object  
    
    Int_t     mCurrentChannel;         //!< current channel processed
    Int_t     mCurrentROC;             //!< current ROC processed
    Int_t     mLastSector;             //!< Last sector processed
    Int_t     mSelectedSector;         //!< Sector selected for processing
    Int_t     mLastSelSector;          //!< Last sector selected for processing
    Int_t     mCurrentRow;             //!< current row processed
    Int_t     mCurrentPad;             //!< current pad processed
    Float_t   mMaxPadSignal;           //!< maximum bin of current pad
    Int_t     mMaxTimeBin;             //!< time bin with maximum value
    Bool_t    mSectorLoop;             //!< only process one sector
    Int_t     mFirstTimeBin;           //!< first time bin to accept
    Int_t     mLastTimeBin;            //!< last time bin to accept
  
    const Mapper&  mTPCmapper;          //! mapper
  
    void resetEvent() final;
};


} // namespace TPC

} // namespace o2
#endif
