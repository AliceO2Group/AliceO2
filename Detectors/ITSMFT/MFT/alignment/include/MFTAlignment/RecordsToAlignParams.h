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

/// \file RecordsToAlignParams.h
/// \author arakotoz@cern.ch
/// \brief Class using records to run MillePede global fit and extract align params

#ifndef ALICEO2_MFT_RECORDS_TO_ALIGN_PARAMS_H
#define ALICEO2_MFT_RECORDS_TO_ALIGN_PARAMS_H

#include <vector>
#include <TChain.h>

#include "MFTAlignment/MillePede2.h"
#include "MFTAlignment/MilleRecordReader.h"
#include "DetectorsCommonDataFormats/AlignParam.h"

#include "MFTAlignment/Aligner.h"

namespace o2
{
namespace mft
{

class RecordsToAlignParams : public Aligner
{
 public:
  /// \brief construtor
  RecordsToAlignParams();

  /// \brief destructor
  ~RecordsToAlignParams() override;

  /// \brief init MilliPede
  void init() override;

  // simple setters

  void setWithControl(const bool choice) { mWithControl = choice; }
  void setNEntriesAutoSave(const int value) { mNEntriesAutoSave = value; }
  void setWithConstraintsRecReader(const bool choice) { mWithConstraintsRecReader = choice; }

  /// \brief perform the simultaneous fit of track (local) and alignement (global) parameters
  void globalFit();

  /// \brief provide access to the AlignParam vector
  void getAlignParams(std::vector<o2::detectors::AlignParam>& alignParams) { alignParams = mAlignParams; }

  /// \brief provide access to the vector of alignment corrections
  void getPedeOutParams(std::vector<double>& output) { output = mPedeOutParams; }

  /// \brief provide access to the vector of errors on the alignement corrections
  void getPedeOutParamsErrors(std::vector<double>& output) { output = mPedeOutParamsErrors; }

  /// \brief provide access to the vector of pulls on the alignement corrections
  void getPedeOutParamsPulls(std::vector<double>& output) { output = mPedeOutParamsPulls; }

  /// \brief connect data record reader to input TChain of records
  void connectRecordReaderToChain(TChain* ch);

  /// \brief conect constraints record reader to input TChain of constraints record
  void connectConstraintsRecReaderToChain(TChain* ch);

 protected:
  bool mWithControl;                                   ///< boolean to set the use of the control tree = chi2 per track filled by MillePede LocalFit()
  long mNEntriesAutoSave = 10000;                      ///< number of entries needed to cyclically call AutoSave for the output control tree
  std::vector<o2::detectors::AlignParam> mAlignParams; ///< vector of alignment parameters computed by MillePede simultaneous fit
  o2::mft::MilleRecordReader* mRecordReader;           ///< utility that handles the reading of the data records used to feed MillePede solver
  bool mWithConstraintsRecReader;                      ///< boolean to set to true if one wants to also read constraints records
  o2::mft::MilleRecordReader* mConstraintsRecReader;   ///< utility that handles the reading of the constraints records
  o2::mft::MillePede2* mMillepede;                     ///< Millepede2 implementation copied from AliROOT
  std::vector<double> mPedeOutParams;                  ///< Vector to store the outputs (alignment corrections) of the MillePede simulatenous fit
  std::vector<double> mPedeOutParamsErrors;            ///< Vector to store the outputs (errors on the alignement corrections) of the MillePede simulatenous fit
  std::vector<double> mPedeOutParamsPulls;             ///< Vector to store the outputs (pulls on the alignement corrections) of the MillePede simulatenous fit

  ClassDefOverride(RecordsToAlignParams, 0);
};

} // namespace mft
} // namespace o2

#endif
