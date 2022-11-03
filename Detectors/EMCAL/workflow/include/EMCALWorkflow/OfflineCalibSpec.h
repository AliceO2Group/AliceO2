// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <string>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "THnSparse.h"

namespace o2
{

namespace emcal
{

/// \class OfflineCalibSpec
/// \brief Task for producing offline calibration objects
/// \ingroup EMCALworkflow
/// \author Hannah Bossi <hannah.bossi@cern.ch>, Yale University
/// \since August 16th, 2022
///
/// This task fills offline calibration objects for the EMCAL.
class OfflineCalibSpec : public framework::Task
{
 public:
  /// \brief Constructor
  /// \param makeCellIDTimeEnergy If true the THnSparseF of cell ID, time, and energy is made
  /// \param rejectCalibTriggers if true, only events which have the o2::trigger::PhT flag will be taken into account
  OfflineCalibSpec(bool makeCellIDTimeEnergy, bool rejectCalibTriggers) : mMakeCellIDTimeEnergy(makeCellIDTimeEnergy), mRejectCalibTriggers(rejectCalibTriggers){};

  /// \brief Destructor
  ~OfflineCalibSpec() override = default;

  /// \brief Initializing the offline calib task
  /// \param ctx Init context
  void init(framework::InitContext& ctx) final;

  /// \brief Fill histograms needed for the offline calibration
  /// \param ctx Processing context
  ///
  void run(framework::ProcessingContext& ctx) final;

  /// \brief Write histograms to an output root file
  /// \param ec end of stream context
  ///
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

 private:
  std::unique_ptr<TH2> mCellAmplitude;         ///< Cell energy vs. cell ID
  std::unique_ptr<TH2> mCellTime;              ///< Cell time vs. cell ID
  std::unique_ptr<TH2> mCellTimeLG;            ///< Cell time vs. cell ID for low gain cells
  std::unique_ptr<TH2> mCellTimeHG;            ///< Cell time vs. cell ID for high gain cells
  std::unique_ptr<TH1> mNevents;               ///< Number of events
  std::unique_ptr<THnSparseF> mCellTimeEnergy; ///< ID, time, energy
  bool mMakeCellIDTimeEnergy = true;           ///< Switch whether or not to make a THnSparseF of cell ID, time, and energy
  bool mRejectCalibTriggers = true;            ///< Switch to select if calib triggerred events should be rejected
};

/// \brief Creating offline calib spec
/// \ingroup EMCALworkflow
///
o2::framework::DataProcessorSpec getEmcalOfflineCalibSpec(bool makeCellIDTimeEnergy, bool rejectCalibTriggers);

} // namespace emcal

} // namespace o2
