// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ALIGNER_H_
#define O2_ALIGNER_H_

// Helper class to pass and deploy geometry (mis)alignment

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include <string>

namespace o2
{
namespace base
{

// Global parameters for digitization
class Aligner : public o2::conf::ConfigurableParamHelper<Aligner>
{
 public:
  const std::string& getCCDB() const { return mCCDB; }
  const std::string& getDetectors() const { return mDetectors; }
  long getTimeStamp() const;
  o2::detectors::DetID::mask_t getDetectorsMask() const;

  bool isAlignmentRequested() const { return getDetectorsMask().any(); }
  void applyAlignment(long timestamp = 0, o2::detectors::DetID::mask_t addMask = o2::detectors::DetID::FullMask) const;

 private:
  std::string mCCDB = "http://ccdb-test.cern.ch:8080"; // URL for CCDB acces
  std::string mDetectors = "all";                      // comma-separated list of modules to align, "all" or "none"
  long mTimeStamp = 0;                                 // assigned TimeStamp or now() if 0

  O2ParamDef(Aligner, "align-geom");
};

} // namespace base
} // namespace o2

#endif
