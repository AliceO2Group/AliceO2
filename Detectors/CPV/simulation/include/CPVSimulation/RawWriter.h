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
#include "CPVCalib/CalibParams.h"

namespace o2
{

namespace cpv
{

static constexpr short kNMod = 3;      ///< Total number of modules
static constexpr short kFirstMod = 2;  ///< First available module
static constexpr short kNPAD = 48;     ///< Nuber of pads per dilogic
static constexpr short kNDilogic = 10; ///< Number of dilogic per row
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
  void setFileFor(FileFor_t filefor) { mFileFor = filefor; }

  void init();
  void digitsToRaw(gsl::span<o2::cpv::Digit> digits, gsl::span<o2::cpv::TriggerRecord> triggers);
  bool processTrigger(const gsl::span<o2::cpv::Digit> digitsbranch, const o2::cpv::TriggerRecord& trg);

  int carryOverMethod(const header::RDHAny* rdh, const gsl::span<char> data,
                      const char* ptr, int maxSize, int splitID,
                      std::vector<char>& trailer, std::vector<char>& header) const;

 private:
  std::vector<padCharge> mPadCharge[kNRow][kNDilogic]; ///< list of signals per event
  FileFor_t mFileFor = FileFor_t::kFullDet;            ///< Granularity of the output files
  std::string mOutputLocation = "./";                  ///< Rawfile name
  std::unique_ptr<CalibParams> mCalibParams;           ///< CPV calibration
  std::vector<uint32_t> mPayload;                      ///< Payload to be written
  gsl::span<o2::cpv::Digit> mDigits;                   ///< Digits input vector - must be in digitized format including the time response
  std::unique_ptr<o2::raw::RawFileWriter> mRawWriter;  ///< Raw writer

  ClassDefNV(RawWriter, 1);
};

} // namespace cpv

} // namespace o2

#endif
