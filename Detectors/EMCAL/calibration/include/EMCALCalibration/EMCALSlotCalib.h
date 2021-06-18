// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \class EMCALSlotCalib
/// \brief  Mark cells in a time slot according to their status and create a bad channel map.
/// \author Hannah Bossi, Yale University
/// \ingroup EMCALCalib
/// \since Feb 20, 2021

#ifndef EMCAL_SLOT_CALIB_H_
#define EMCAL_SLOT_CALIB_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALBase/Geometry.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include <boost/format.hpp>


namespace o2
{
namespace emcal
{
    class EMCALChannelCalibrator;
    class EMCALSlotCalib
    {
        //using Slot = o2::calibration::TimeSlot<o2::emcal::EMCALChannelData>;
        using Cell = o2::emcal::Cell;
        using TFType = uint64_t;
        using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;
        public:

            EMCALSlotCalib(int ns) : mSigma(ns){
                // NCELLS includes DCal, treat as one calibration
                o2::emcal::Geometry* mGeometry = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);
                int NCELLS = mGeometry->GetNCells();
            }

            ~EMCALSlotCalib() = default;


            /// \brief Average energy per hit is caluclated for each cell.
            /// \param emin -- min. energy for cell amplitudes
            /// \param emax -- max. energy for cell amplitudes
            void buildHitAndEnergyMean(double emin, double emax);
            /// \brief Peform the calibration and flag the bad channel map
            /// Average energy per hit histogram is fitted with a gaussian
            /// good area is +-mSigma
            /// cells beyond that value are flagged as bad.
            void analyzeSlot();

            int getNsigma() const { return mSigma; }
            void setNsigma(int ns) { mSigma = ns; }
            
        private:
            boostHisto mEsumHisto;     ///< contains the average energy per hit for each cell
            boostHisto mCellAmplitude; ///< is the input for the calibration, hist of cell E vs. ID
            int mSigma = 4;            ///< number of sigma used in the calibration to define outliers
            BadChannelMap mOutputBCM;  ///< bad channel map we will write the results to
            int NCELLS = 0;            ///< number of cells in EMCAL + DCAL
        ClassDefNV(EMCALSlotCalib, 1);
    };

} // end namespace emcal
} // end namespace o2

#endif /*EMCAL_SLOT_CALIB_H_ */