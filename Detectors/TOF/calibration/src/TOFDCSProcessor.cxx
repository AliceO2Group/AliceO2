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

#include <TOFCalibration/TOFDCSProcessor.h>
#include "DetectorsCalibration/Utils.h"
#include "Rtypes.h"
#include <deque>
#include <string>
#include <algorithm>
#include <iterator>
#include <cstring>
#include <bitset>

using namespace o2::tof;
using namespace o2::dcs;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

ClassImp(o2::tof::TOFDCSinfo);

void TOFDCSinfo::print() const
{
  LOG(info) << "First Value: timestamp = " << firstValue.first << ", value = " << firstValue.second;
  LOG(info) << "Last Value:  timestamp = " << lastValue.first << ", value = " << lastValue.second;
  LOG(info) << "Mid Value:   timestamp = " << midValue.first << ", value = " << midValue.second;
  LOG(info) << "Min Value:   timestamp = " << minValue.first << ", value = " << minValue.second;
  LOG(info) << "Max Value:   timestamp = " << maxValue.first << ", value = " << maxValue.second;
  LOG(info) << "Max Change:  timestamp[0] = " << maxChange.first[0] << ", timestamp[1] = " << maxChange.first[1] << ", value = " << maxChange.second;
  std::string updatedStr = updated ? "UPDATED" : "NOT UPDATED";
  LOG(info) << "Status = " << updatedStr;
}

//__________________________________________________________________

void TOFDCSProcessor::init(const std::vector<DPID>& pids)
{
  // fill the array of the DPIDs that will be used by TOF
  // pids should be provided by CCDB

  for (const auto& it : pids) {
    mPids[it] = false;
    mTOFDCS[it].makeEmpty();
  }

  for (int iddl = 0; iddl < NDDLS; ++iddl) {
    for (int ifeac = 0; ifeac < NFEACS; ++ifeac) {
      getStripsConnectedToFEAC(iddl, ifeac, mFeacInfo[iddl][ifeac]);
    }
  }
}

//__________________________________________________________________

int TOFDCSProcessor::process(const gsl::span<const DPCOM> dps)
{

  // first we check which DPs are missing - if some are, it means that
  // the delta map was sent
  if (mVerboseDP || mVerboseHVLV) {
    LOG(info) << "\n\n\nProcessing new DCS DP map\n-----------------";
  }
  if (!mFirstTimeSet) {
    mFirstTime = mStartValidity;
    mFirstTimeSet = true;
  }

  if (false) {
    std::unordered_map<DPID, DPVAL> mapin;
    for (auto& it : dps) {
      mapin[it.id] = it.data;
    }
    for (auto& it : mPids) {
      const auto& el = mapin.find(it.first);
      if (el == mapin.end()) {
        LOG(debug) << "DP " << it.first << " not found in map";
      } else {
        LOG(debug) << "DP " << it.first << " found in map";
      }
    }
  }

  mUpdateFeacStatus = false; // by default, we do not foresee a new entry in the CCDB for the FEAC
  mUpdateHVStatus = false;   // by default, we do not foresee a new entry in the CCDB for the HV

  // now we process all DPs, one by one
  for (const auto& it : dps) {
    // we process only the DPs defined in the configuration
    const auto& el = mPids.find(it.id);
    if (el == mPids.end()) {
      LOG(info) << "DP " << it.id << " not found in TOFDCSProcessor, we will not process it";
      continue;
    }
    processDP(it);
    mPids[it.id] = true;
  }

  if (mUpdateFeacStatus) {
    updateFEACCCDB();
  }

  if (mUpdateHVStatus) {
    updateHVCCDB();
  }

  return 0;
}

//__________________________________________________________________

int TOFDCSProcessor::processDP(const DPCOM& dpcom)
{

  // processing single DP

  auto& dpid = dpcom.id;
  const auto& type = dpid.get_type();
  auto& val = dpcom.data;
  if (mVerboseDP || mVerboseHVLV) {
    if (type == DPVAL_DOUBLE) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<double>(dpcom);
    } else if (type == DPVAL_INT) {
      LOG(info);
      LOG(info) << "Processing DP = " << dpcom << ", with value = " << o2::dcs::getValue<int32_t>(dpcom);
    }
  }
  auto flags = val.get_flags();
  if (processFlags(flags, dpid.get_alias()) == 0) {
    // now I need to access the correct element
    if (type == DPVAL_DOUBLE) {
      // for these DPs, we will store the first, last, mid value, plus the value where the maximum variation occurred
      auto& dvect = mDpsdoublesmap[dpid];
      LOG(debug) << "mDpsdoublesmap[dpid].size() = " << dvect.size();
      auto etime = val.get_epoch_time();
      if (dvect.size() == 0 ||
          etime != dvect.back().get_epoch_time()) { // we check
                                                    // that we did not get the
                                                    // same timestamp as the
                                                    // latest one
        dvect.push_back(val);
      }
    }

    if (type == DPVAL_INT) {
      // for these DPs, we need some processing
      if (std::strstr(dpid.get_alias(), "FEACSTATUS") != nullptr) { // DP is FEACSTATUS
        std::string aliasStr(dpid.get_alias());
        // extracting DDL number (regex is quite slow, using this way)
        const auto offs = std::strlen("TOF_FEACSTATUS_");
        std::size_t const nn = aliasStr.find_first_of("0123456789", offs);
        std::size_t const mm = aliasStr.find_first_not_of("0123456789", nn);
        std::string ddlStr = aliasStr.substr(nn, mm != std::string::npos ? mm - nn : mm);
        auto iddl = std::stoi(ddlStr);
        std::bitset<8> feacstatus(o2::dcs::getValue<int32_t>(dpcom));
        if (mVerboseHVLV) {
          LOG(info) << "DDL: " << iddl << ": Prev FEAC = " << mPrevFEACstatus[iddl] << ", new = " << feacstatus;
        }
        if (feacstatus == mPrevFEACstatus[iddl]) {
          if (mVerboseHVLV) {
            LOG(info) << "Same FEAC status as before, we do nothing";
          }
          return 0;
        }
        if (mVerboseHVLV) {
          LOG(info) << "Something changed in LV for DDL " << iddl << ", we need to check what";
        }
        mUpdateFeacStatus = true;
        int plate = -1, strip = -1;
        int det[5] = {iddl / 4, -1, -1, -1, -1};
        for (auto ifeac = 0; ifeac < NFEACS; ++ifeac) { // we have one bit per FEAC
          auto singlefeacstatus = feacstatus[ifeac];
          for (int istrip = 0; istrip < 6; ++istrip) {
            if (mFeacInfo[iddl][ifeac].stripInSM[istrip] == -1) {
              continue;
            }
            for (int ipadz = 0; ipadz < Geo::NPADZ; ++ipadz) {
              for (int ipadx = mFeacInfo[iddl][ifeac].firstPadX; ipadx <= mFeacInfo[iddl][ifeac].lastPadX; ++ipadx) {
                if (mVerboseHVLV) {
                  LOG(info) << "mFeacInfo[" << iddl << "][" << ifeac << "].stripInSM[" << istrip << "] = " << mFeacInfo[iddl][ifeac].stripInSM[istrip];
                }
                Geo::getStripAndModule(mFeacInfo[iddl][ifeac].stripInSM[istrip], plate, strip);
                det[1] = plate;
                det[2] = strip;
                det[3] = ipadz;
                det[4] = ipadx;
                if (mVerboseHVLV) {
                  LOG(info) << "det[0] = " << det[0] << ", det[1] = " << det[1] << ", det[2] = " << det[2] << ", det[3] = " << det[3] << ", det[4] = " << det[4];
                }
                int channelIdx = Geo::getIndex(det);
                if (mFeac[channelIdx] != singlefeacstatus) {
                  mFeac[channelIdx] = singlefeacstatus;
                }
              }
            }
          }
        } // end loop on FEACs
        if (mVerboseHVLV) {
          LOG(info) << "Updating previous FEAC status for DDL " << iddl;
        }
        mPrevFEACstatus[iddl] = feacstatus;
      } // end processing current DP, when it is of type FEACSTATUS

      if (std::strstr(dpid.get_alias(), "HVSTATUS") != nullptr) { // DP is HVSTATUS
        std::string aliasStr(dpid.get_alias());
        // extracting SECTOR and PLATE number (regex is quite slow, using this way)
        const auto offs = std::strlen("TOF_HVSTATUS_SM");
        std::size_t const nn = aliasStr.find_first_of("0123456789", offs);
        std::size_t const mm = aliasStr.find_first_not_of("0123456789", nn);
        std::size_t const oo = aliasStr.find_first_of("0123456789", mm);
        std::size_t const pp = aliasStr.find_first_not_of("0123456789", oo);
        std::string sectorStr = aliasStr.substr(nn, mm != std::string::npos ? mm - nn : mm);
        auto isect = std::stoi(sectorStr);
        std::string plateStr = aliasStr.substr(oo, pp != std::string::npos ? pp - oo : pp);
        auto iplat = std::stoi(plateStr);
        std::bitset<19> hvstatus(o2::dcs::getValue<int32_t>(dpcom));
        if (mVerboseHVLV) {
          LOG(info) << "Sector: " << isect << ", plate = " << iplat << ": Prev HV = "
                    << mPrevHVstatus[iplat][isect] << ", new = " << hvstatus;
        }
        if (hvstatus == mPrevHVstatus[iplat][isect]) {
          if (mVerboseHVLV) {
            LOG(info) << "Same HV status as before, we do nothing";
          }
          return 0;
        }
        if (mVerboseHVLV) {
          LOG(info) << "Something changed in HV for Sect " << isect << " and plate "
                    << iplat << ", we need to check what";
        }
        mUpdateHVStatus = true;
        int det[5] = {isect, iplat, -1, -1, -1};
        auto nStrips = (iplat == 2 ? Geo::NSTRIPA : (iplat == 0 || iplat == 4) ? Geo::NSTRIPC : Geo::NSTRIPB);
        for (auto istrip = 0; istrip < nStrips; ++istrip) {
          auto singlestripHV = hvstatus[istrip];
          for (int ipadz = 0; ipadz < Geo::NPADZ; ++ipadz) {
            for (int ipadx = 0; ipadx < Geo::NPADX; ++ipadx) {
              det[2] = istrip;
              det[3] = ipadz;
              det[4] = ipadx;
              int channelIdx = Geo::getIndex(det);
              if (mHV[channelIdx] != singlestripHV) {
                mHV[channelIdx] = singlestripHV;
              }
            }
          }
        } // end loop on strips
        if (mVerboseHVLV) {
          LOG(info) << "Updating previous HV status for Sector: " << isect << ", plate = " << iplat;
        }
        mPrevHVstatus[iplat][isect] = hvstatus;
      } //end processing current DP, when it is of type HVSTATUS
    }
  }
  return 0;
}

//______________________________________________________________________

uint64_t TOFDCSProcessor::processFlags(const uint64_t flags, const char* pid)
{

  // function to process the flag. the return code zero means that all is fine.
  // anything else means that there was an issue

  // for now, I don't know how to use the flags, so I do nothing

  if (flags & DataPointValue::KEEP_ALIVE_FLAG) {
    LOG(debug) << "KEEP_ALIVE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::END_FLAG) {
    LOG(debug) << "END_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::FBI_FLAG) {
    LOG(debug) << "FBI_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::NEW_FLAG) {
    LOG(debug) << "NEW_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIRTY_FLAG) {
    LOG(debug) << "DIRTY_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::TURN_FLAG) {
    LOG(debug) << "TURN_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::WRITE_FLAG) {
    LOG(debug) << "WRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::READ_FLAG) {
    LOG(debug) << "READ_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::OVERWRITE_FLAG) {
    LOG(debug) << "OVERWRITE_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::VICTIM_FLAG) {
    LOG(debug) << "VICTIM_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::DIM_ERROR_FLAG) {
    LOG(debug) << "DIM_ERROR_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_DPID_FLAG) {
    LOG(debug) << "BAD_DPID_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FLAGS_FLAG) {
    LOG(debug) << "BAD_FLAGS_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_TIMESTAMP_FLAG) {
    LOG(debug) << "BAD_TIMESTAMP_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_PAYLOAD_FLAG) {
    LOG(debug) << "BAD_PAYLOAD_FLAG active for DP " << pid;
  }
  if (flags & DataPointValue::BAD_FBI_FLAG) {
    LOG(debug) << "BAD_FBI_FLAG active for DP " << pid;
  }

  return 0;
}

//______________________________________________________________________

void TOFDCSProcessor::updateDPsCCDB()
{

  // here we create the object to then be sent to CCDB
  LOG(info) << "Finalizing";
  union Converter {
    uint64_t raw_data;
    double double_value;
  } converter0, converter1;

  for (auto& it : mPids) {
    const auto& type = it.first.get_type();
    if (type == o2::dcs::DPVAL_DOUBLE) {
      auto& tofdcs = mTOFDCS[it.first];
      if (it.second) {     // we processed the DP at least 1x
        if (mVerboseDP) {
          LOG(info) << "Processing DP " << it.first.get_alias();
        }
        it.second = false; // reset for the next period
        tofdcs.updated = true;
        auto& dpvect = mDpsdoublesmap[it.first];
        tofdcs.firstValue.first = dpvect[0].get_epoch_time();
        converter0.raw_data = dpvect[0].payload_pt1;
        tofdcs.firstValue.second = converter0.double_value;
        tofdcs.lastValue.first = dpvect.back().get_epoch_time();
        converter0.raw_data = dpvect.back().payload_pt1;
        tofdcs.lastValue.second = converter0.double_value;
        for (const auto& el : dpvect) {
          converter0.raw_data = el.payload_pt1;
          if (converter0.double_value < tofdcs.minValue.second) {
            tofdcs.minValue.first = el.get_epoch_time();
            tofdcs.minValue.second = converter0.double_value;
          }
          if (converter0.double_value > tofdcs.maxValue.second) {
            tofdcs.maxValue.first = el.get_epoch_time();
            tofdcs.maxValue.second = converter0.double_value;
          }
        }
        // now I will look for the max change
        if (dpvect.size() > 1) {
          auto deltatime = dpvect.back().get_epoch_time() - dpvect[0].get_epoch_time();
          if (deltatime < 60000) {
            // if we did not cover at least 1 minute,
            // max variation is defined as the difference between first and last value
            converter0.raw_data = dpvect[0].payload_pt1;
            converter1.raw_data = dpvect.back().payload_pt1;
            double delta = converter0.double_value - converter1.double_value;
            tofdcs.maxChange.first[0] = dpvect[0].get_epoch_time();
            tofdcs.maxChange.first[1] = dpvect.back().get_epoch_time();
            tofdcs.maxChange.second = delta;
          } else {
            for (auto i = 0; i < dpvect.size() - 1; ++i) {
              for (auto j = i + 1; j < dpvect.size(); ++j) {
                auto deltatime = dpvect[j].get_epoch_time() - dpvect[i].get_epoch_time();
                if (deltatime >= 60000) { // we compare to values coming from at least 1 minute later; epoch_time in ms
                  converter0.raw_data = dpvect[i].payload_pt1;
                  converter1.raw_data = dpvect[j].payload_pt1;
                  double delta = converter0.double_value - converter1.double_value;
                  if (std::abs(delta) > std::abs(tofdcs.maxChange.second)) {
                    tofdcs.maxChange.first[0] = dpvect[i].get_epoch_time();
                    tofdcs.maxChange.first[1] = dpvect[j].get_epoch_time();
                    tofdcs.maxChange.second = delta;
                  }
                }
              }
            }
          }
          // mid point
          auto midIdx = dpvect.size() / 2 - 1;
          tofdcs.midValue.first = dpvect[midIdx].get_epoch_time();
          converter0.raw_data = dpvect[midIdx].payload_pt1;
          tofdcs.midValue.second = converter0.double_value;
        } else {
          tofdcs.maxChange.first[0] = dpvect[0].get_epoch_time();
          tofdcs.maxChange.first[1] = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          tofdcs.maxChange.second = converter0.double_value;
          tofdcs.midValue.first = dpvect[0].get_epoch_time();
          converter0.raw_data = dpvect[0].payload_pt1;
          tofdcs.midValue.second = converter0.double_value;
        }
      } else {
        tofdcs.updated = false;
      }
      if (mVerboseDP) {
        LOG(info) << "PID " << it.first.get_alias() << " was updated to:";
        tofdcs.print();
      }
    }
  }
  if (mVerboseDP) {
    LOG(info) << "Printing object to be sent to CCDB";
    for (auto& it : mPids) {
      const auto& type = it.first.get_type();
      if (type == o2::dcs::DPVAL_DOUBLE) {
        LOG(info) << "PID = " << it.first.get_alias();
        auto& tofdcs = mTOFDCS[it.first];
        tofdcs.print();
      }
    }
    LOG(info) << "done";
  }

  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mTOFDCS, mccdbDPsInfo, "TOF/Calib/DCSDPs", md, mStartValidity, mStartValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY);

  return;
}

//______________________________________________________________________

void TOFDCSProcessor::updateFEACCCDB()
{

  // we need to update a CCDB for the FEAC status --> let's prepare the CCDBInfo

  if (mVerboseHVLV) {
    LOG(info) << "At least one FEAC changed status --> we will update CCDB";
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mFeac, mccdbLVInfo, "TOF/Calib/LVStatus", md, mStartValidity, mStartValidity + o2::ccdb::CcdbObjectInfo::MONTH);
  return;
}

//______________________________________________________________________

void TOFDCSProcessor::updateHVCCDB()
{

  // we need to update a CCDB for the HV status --> let's prepare the CCDBInfo

  if (mVerboseHVLV) {
    LOG(info) << "At least one HV changed status --> we will update CCDB";
  }
  std::map<std::string, std::string> md;
  md["responsible"] = "Chiara Zampolli";
  o2::calibration::Utils::prepareCCDBobjectInfo(mHV, mccdbHVInfo, "TOF/Calib/HVStatus", md, mStartValidity, mStartValidity + o2::ccdb::CcdbObjectInfo::MONTH);
  return;
}

//_______________________________________________________________________

void TOFDCSProcessor::getStripsConnectedToFEAC(int nDDL, int nFEAC, TOFFEACinfo& info) const
{

  //
  // Taken from AliRoot/TOF/AliTOFLvHvDataPoints.cxx
  //
  // FEAC-strip mapping:
  // return the strips and first PadX numbers
  // connected to the FEAC number nFEAC in the crate number nDDL
  //

  switch (nDDL % 4) {
    case 0:
      info.firstPadX = 0;
      info.lastPadX = Geo::NPADX / 2 - 1;

      if (nFEAC <= 2) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC;
        }
      } else if (nFEAC == 3) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC;
        }
      } else if (nFEAC == 4) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 1;
        }
      } else if (nFEAC == 5) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 1;
        }
      } else if (nFEAC == 6) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 2;
        }
      } else if (nFEAC == 7) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 2;
        }
      }

      break;
    case 1:
      info.firstPadX = Geo::NPADX / 2;
      info.lastPadX = Geo::NPADX - 1;

      if (nFEAC <= 2) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC;
        }
      } else if (nFEAC == 3) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC;
        }
      } else if (nFEAC == 4) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 1;
        }
      } else if (nFEAC == 5) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 1;
        }
      } else if (nFEAC == 6) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 1;
        }
      } else if (nFEAC == 7) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = ii + 6 * nFEAC - 2;
        }
      }

      break;
    case 2:
      info.firstPadX = Geo::NPADX / 2;
      info.lastPadX = Geo::NPADX - 1;

      if (nFEAC <= 2) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC);
        }
      } else if (nFEAC == 3) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC);
        }
      } else if (nFEAC == 4) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 1);
        }
      } else if (nFEAC == 5) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 1);
        }
      } else if (nFEAC == 6) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 2);
        }
      } else if (nFEAC == 7) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 2);
        }
      }

      break;
    case 3:
      info.firstPadX = 0;
      info.lastPadX = Geo::NPADX / 2 - 1;

      if (nFEAC <= 2) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC);
        }
      } else if (nFEAC == 3) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC);
        }
      } else if (nFEAC == 4) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 1);
        }
      } else if (nFEAC == 5) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 1);
        }
      } else if (nFEAC == 6) {
        for (int ii = 0; ii < 5; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 1);
        }
      } else if (nFEAC == 7) {
        for (int ii = 0; ii < 6; ++ii) {
          info.stripInSM[ii] = 90 - (ii + 6 * nFEAC - 2);
        }
      }

      break;
  }
  if (mVerboseHVLV) {
    for (int ii = 0; ii < 6; ++ii) {
      LOG(info) << "nDDL = " << nDDL << ", nFEAC = " << nFEAC << ", stripInSM[" << ii << "] = " << info.stripInSM[ii];
    }
  }
}
