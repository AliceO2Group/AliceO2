// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_RAWWRITER_H
#define ALICEO2_CPV_RAWWRITER_H

#include <gsl/span>

#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <map>
#include <vector>

#include "Rtypes.h"

#include "DetectorsRaw/RawFileWriter.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "DataFormatsCPV/CalibParams.h"
#include "DataFormatsCPV/Pedestals.h"
#include "DataFormatsCPV/BadChannelMap.h"

namespace o2
{

namespace cpv
{

static constexpr short kNcc = 24;      ///< Total number of column controllers
static constexpr short kNPAD = 48;     ///< Nuber of pads per dilogic
static constexpr short kNDilogic = 4;  ///< Number of dilogics
static constexpr short kNGasiplex = 5; ///< Number of dilogic per row
static constexpr short kNRow = 48;     ///< number of rows 16*3 mod

struct padCharge {
  short charge;
  short pad;
  padCharge() : charge(0), pad(0) {}
  padCharge(short a, short b) : charge(a),
                                pad(b)
  {
  } //for std::vector::emplace_back functionality
};

class RawWriter
{
 public:
  enum class FileFor_t {
    kFullDet,
    kLink
  };
  RawWriter() = default;
  RawWriter(const char* outputdir) { setOutputLocation(outputdir); }
  ~RawWriter() = default;

  o2::raw::RawFileWriter& getWriter() const { return *mRawWriter; }

  void setOutputLocation(const char* outputdir) { mOutputLocation = outputdir; }
  void setCcdbUrl(const char* ccdbUrl) { mCcdbUrl = ccdbUrl; }
  void setFileFor(FileFor_t filefor) { mFileFor = filefor; }

  void init();
  void digitsToRaw(gsl::span<o2::cpv::Digit> digits, gsl::span<o2::cpv::TriggerRecord> triggers);
  bool processOrbit(const gsl::span<o2::cpv::Digit> digitsbranch, const gsl::span<o2::cpv::TriggerRecord> trgs);

  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 private:
  std::vector<padCharge> mPadCharge[kNcc][kNDilogic][kNGasiplex]; ///< list of signals per event
  FileFor_t mFileFor = FileFor_t::kFullDet;                       ///< Granularity of the output files
  std::string mOutputLocation = "./";                             ///< Rawfile name
  std::string mCcdbUrl = "http://ccdb-test.cern.ch:8080";         ///< CCDB Url
  std::unique_ptr<CalibParams> mCalibParams;                      ///< CPV calibration
  std::unique_ptr<Pedestals> mPedestals;                          ///< CPV pedestals
  std::unique_ptr<BadChannelMap> mBadMap;                         ///< CPV bad channel map
  std::vector<char> mPayload;                                     ///< Payload to be written
  gsl::span<o2::cpv::Digit> mDigits;                              ///< Digits input vector - must be in digitized format including the time response
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter;             ///< Raw writer

  ClassDefNV(RawWriter, 2);
};

} // namespace cpv

} // namespace o2

#endif
