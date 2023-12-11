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

void readAlpideCCDB(long timestamp = -1, float thresh = 0)
{
  o2::ccdb::CcdbApi api;
  // api.init("alice-ccdb.cern.ch");
  api.init("ccdb-test.cern.ch");
  map<string, string> headers;
  map<std::string, std::string> filter;
  auto calib = api.retrieveFromTFileAny<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>>("MFT/Config/AlpideParam/", filter, timestamp, &headers);
  calib->printKeyValues();
}
