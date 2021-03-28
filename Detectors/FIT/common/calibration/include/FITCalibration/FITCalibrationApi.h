// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FITCALIBRATIONAPI_H
#define O2_FITCALIBRATIONAPI_H

#include <utility>

#include "CCDB/CcdbApi.h"
#include "TH1F.h"
#include "TGraph.h"
#include "Rtypes.h"


namespace o2::fit
{


template <typename CalibrationObjectType>
class FITCalibrationApi final
{


  static constexpr const char* DEFAULT_CCDB_URL = "http://localhost:8080";


 public:
  explicit FITCalibrationApi(std::string calibrationObjectPath, const std::string& CCDBURL = DEFAULT_CCDB_URL);
  [[nodiscard]] std::string getCCDBURL() const { return mCCDBApi->getURL(); }
  [[nodiscard]] const std::string& getCalibrationObjectPath() const { return mCalibrationObjectPath; }
  void readCalibrationObject(uint64_t timestamp);
  [[nodiscard]] CalibrationObjectType& getCalibrationObject() const;

 private:
  void _checkIfObjectWasSuccessfullyRead() const;


 private:
  CalibrationObjectType* mCalibrationObject = nullptr;
  std::unique_ptr<o2::ccdb::CcdbApi> mCCDBApi;
  std::string mCalibrationObjectPath;


};


template <typename CalibrationObjectType>
FITCalibrationApi<CalibrationObjectType>::FITCalibrationApi(std::string calibrationObjectPath, const std::string& CCDBURL)
  : mCalibrationObjectPath(std::move(calibrationObjectPath))
{
    mCCDBApi = std::make_unique<o2::ccdb::CcdbApi>();
    mCCDBApi->init(CCDBURL);
}

template <typename CalibrationObjectType>
void FITCalibrationApi<CalibrationObjectType>::readCalibrationObject(uint64_t timestamp)
{
  mCalibrationObject = mCCDBApi->retrieveFromTFileAny<CalibrationObjectType>(mCalibrationObjectPath,
                                                                               std::map<std::string, std::string>(), timestamp);
  _checkIfObjectWasSuccessfullyRead();
}

template <typename CalibrationObjectType>
void FITCalibrationApi<CalibrationObjectType>::_checkIfObjectWasSuccessfullyRead() const
{
  if(!mCalibrationObject){
    throw std::runtime_error("Calibration object was not read successfully!");
  }
}

template <typename CalibrationObjectType>
CalibrationObjectType& FITCalibrationApi<CalibrationObjectType>::getCalibrationObject() const
{
  _checkIfObjectWasSuccessfullyRead();
  return *mCalibrationObject;
}




}





#endif //O2_FITCALIBRATIONAPI_H
