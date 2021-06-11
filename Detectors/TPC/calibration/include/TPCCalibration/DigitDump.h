// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_DIGITDUMP_H_
#define ALICEO2_TPC_DIGITDUMP_H_

/// \file   DigitDump.h
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "Rtypes.h"
#include "TFile.h"

#include "DataFormatsTPC/Defs.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCCalibration/CalibRawBase.h"

class TTree;

namespace o2
{
namespace tpc
{

/// \brief Pedestal calibration class
///
/// This class is used to produce pad wise pedestal and noise calibration data
///
/// origin: TPC
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

class DigitDump : public CalibRawBase
{
 public:
  /// default constructor
  DigitDump(PadSubset padSubset = PadSubset::ROC) : CalibRawBase(padSubset) {}

  /// output file name
  void setDigitFileName(std::string_view fileName) { mDigitFile = fileName; }

  /// pedestal file name
  void setPedestalAndNoiseFile(std::string_view fileName) { mPedestalAndNoiseFile = fileName; }

  /// default destructor
  ~DigitDump() override;

  /// initialize DigitDump from DigitDumpParam
  void init();

  /// update function called once per digit
  ///
  /// \param roc readout chamber
  /// \param row row in roc
  /// \param pad pad in row
  /// \param timeBin time bin
  /// \param signal ADC signal
  Int_t updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final { return 0; }

  /// not used
  Int_t updateCRU(const CRU& cru, const Int_t row, const Int_t pad,
                  const Int_t timeBin, const Float_t signal) final;

  /// Get the pedestal calibration object
  ///
  /// \return pedestal calibration object
  const CalPad& getPedestal() const { return *mPedestal.get(); }

  /// Get the noise calibration object
  ///
  /// \return noise calibration object
  const CalPad& getNoise() const { return *mNoise.get(); }

  /// add a masked pad
  void addPadMask(int roc, int row, int pad) { mPadMask.push_back({roc, row, pad}); }

  /// set noise threshold
  void setNoiseThreshold(float noiseThreshold) { mNoiseThreshold = noiseThreshold; }

  /// set the adc range
  void setADCRange(float minADC, float maxADC)
  {
    mADCMin = minADC;
    mADCMax = maxADC;
  }

  /// set the timeBin range
  void setTimeBinRange(int first, int last)
  {
    mFirstTimeBin = first;
    mLastTimeBin = last;
  }

  /// sort the digits
  void sortDigits();

  /// clear the digits
  void clearDigits()
  {
    for (auto& digits : mDigits) {
      digits.clear();
    }
  }

  /// set in memory only mode
  void setInMemoryOnly(bool mode = true) { mInMemoryOnly = mode; }

  /// get in memory mode
  bool getInMemoryMode() const { return mInMemoryOnly; }

  /// return digits for specific sector
  std::vector<Digit>& getDigits(int sector) { return mDigits[sector]; }

  /// directly add a digit
  void addDigit(const CRU& cru, const float signal, const int rowInSector, const int padInRow, const int timeBin)
  {
    mDigits[cru.sector()].emplace_back(cru, signal, rowInSector, padInRow, timeBin);
  }

  /// initialize
  void initInputOutput();

  /// End event function
  void endEvent() final;

 private:
  std::unique_ptr<CalPad> mPedestal{}; ///< CalDet object with pedestal information
  std::unique_ptr<CalPad> mNoise{};    ///< CalDet object with noise

  TTree* mTree{nullptr};          ///< output tree
  std::unique_ptr<TFile> mFile{}; ///< output file

  std::array<std::vector<Digit>, Sector::MAXSECTOR> mDigits; ///< digit vector to be stored inside the file
  std::string mDigitFile{};                                  ///< file name for the outuput digits
  std::string mPedestalAndNoiseFile{};                       ///< file name for the pedestal and nosie file

  std::vector<std::array<int, 3>> mPadMask; ///< coordinates of pads to skip

  int mFirstTimeBin{0};      ///< first time bin used in analysis
  int mLastTimeBin{1000};    ///< first time bin used in analysis
  float mADCMin{-100};       ///< minimum adc value
  float mADCMax{1024};       ///< maximum adc value
  float mNoiseThreshold{-1}; ///< zero suppression threshold in noise sigma
  bool mInMemoryOnly{false}; ///< if processing is only done in memory, no file writing
  bool mInitialized{false};  ///< if init was called

  /// set up the output tree
  void setupOutputTree();

  /// load noise and pedestal
  void loadNoiseAndPedestal();

  /// dummy reset
  void resetEvent() final {}
};

} // namespace tpc

} // namespace o2
#endif
