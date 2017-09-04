// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_CALIBPEDESTAL_H_
#define ALICEO2_TPC_CALIBPEDESTAL_H_

/// \file   CalibPedestal.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <vector>
#include <memory>

#include "Rtypes.h"

#include "TPCBase/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCCalibration/CalibRawBase.h"

namespace o2
{
namespace TPC
{


/// \brief Pedestal calibration class
///
/// This class is used to produce pad wise pedestal and noise calibration data
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class CalibPedestal : public CalibRawBase
{
  public:
    using vectorType = std::vector<float>;

    /// default constructor
    CalibPedestal(PadSubset padSubset = PadSubset::ROC);

    /// default destructor
    virtual ~CalibPedestal() = default;

    /// update function called once per digit
    ///
    /// \param sector
    Int_t updateROC(const Int_t sector, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final;

    /// not used
    Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                    const Int_t timeBin, const Float_t signal) final { return 0;}

    /// Reset pedestal data
    void resetData();

    /// set the adc range
    void setADCRange(int minADC, int maxADC) { mADCMin = minADC; mADCMax = maxADC; mNumberOfADCs = mADCMax-mADCMin+1;}

    /// Analyse the buffered adc values and calculate noise and pedestal
    void analyse();

    /// Get the pedestal calibration object
    ///
    /// \return pedestal calibration object
    const CalPad& getPedestal() const { return mPedestal; }

    /// Get the noise calibration object
    ///
    /// \return noise calibration object
    const CalPad& getNoise() const { return mNoise; }

    /// Dump the relevant data to file
    void dumpToFile(TString filename) final;

    /// Dummy end event
    void endEvent() final {};

  //private:
    Int_t      mADCMin;    ///< minimum adc value
    Int_t      mADCMax;    ///< maximum adc value
    Int_t      mNumberOfADCs; ///< number of adc values (mADCMax-mADCMin+1)
    CalPad     mPedestal;  ///< CalDet object with pedestal information
    CalPad     mNoise;     ///< CalDet object with noise

    std::vector<std::unique_ptr<vectorType>> mADCdata; //!< ADC data to calculate noise and pedestal

    /// return the value vector for a readout chamber
    ///
    /// \param roc readout chamber
    /// \param create if to create the vector if it does not exist
    vectorType* getVector(ROC roc, bool create=kFALSE);

    /// dummy reset
    void resetEvent() final {}
};

} // namespace TPC

} // namespace o2
#endif
