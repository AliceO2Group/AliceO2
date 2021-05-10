// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef TOF_CHANNEL_CALIBRATOR_H_
#define TOF_CHANNEL_CALIBRATOR_H_

#include "DetectorsCalibration/TimeSlotCalibration.h"
#include "DetectorsCalibration/TimeSlot.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/CalibInfoCluster.h"
#include "DataFormatsTOF/CalibLHCphaseTOF.h"
#include "TOFBase/Geo.h"
#include "CCDB/CcdbObjectInfo.h"
#include "TOFCalibration/CalibTOFapi.h"

#include <array>
#include <boost/histogram.hpp>

#include "TGraphErrors.h"
#include "TF1.h"
#include "MathUtils/fit.h"
#include "DetectorsCalibration/Utils.h"
#include <boost/histogram.hpp>
#include <boost/histogram/ostream.hpp>
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include <boost/format.hpp>

using o2::math_utils::fitGaus;

namespace o2
{
namespace tof
{

class TOFChannelData
{

  using Slot = o2::calibration::TimeSlot<o2::tof::TOFChannelData>;
  using CalibTOFapi = o2::tof::CalibTOFapi;
  using boostHisto = boost::histogram::histogram<std::tuple<boost::histogram::axis::regular<double, boost::use_default, boost::use_default, boost::use_default>, boost::histogram::axis::integer<>>, boost::histogram::unlimited_storage<std::allocator<char>>>;

 public:
  static constexpr int NCOMBINSTRIP = o2::tof::Geo::NPADX + o2::tof::Geo::NPADS;

  TOFChannelData()
  {
    LOG(INFO) << "Default c-tor, not to be used";
  }

  TOFChannelData(int nb, float r, CalibTOFapi* cta, int nElsPerSector = o2::tof::Geo::NPADSXSECTOR) : mNBins(nb), mRange(r), mCalibTOFapi(cta), mNElsPerSector(nElsPerSector)
  {
    if (r <= 0. || nb < 1) {
      throw std::runtime_error("Wrong initialization of the histogram");
    }
    mV2Bin = mNBins / (2 * mRange);
    for (int isect = 0; isect < 18; isect++) {
      mHisto[isect] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(mNBins, -mRange, mRange, "t-texp"),
                                                       boost::histogram::axis::integer<>(0, mNElsPerSector, "channel index in sector" + std::to_string(isect))); // bin is defined as [low, high[
    }
    mEntries.resize(mNElsPerSector * 18, 0);
  }

  ~TOFChannelData() = default;

  void print() const;
  void print(int isect) const;
  void printEntries() const;
  void fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data);
  void fill(const gsl::span<const o2::tof::CalibInfoCluster> data);
  void merge(const TOFChannelData* prev);
  int findBin(float v) const;
  float integral(int chmin, int chmax, float binmin, float binmax) const;
  float integral(int chmin, int chmax, int binxmin, int binxmax) const;
  float integral(int ch, float binmin, float binmax) const;
  float integral(int ch, int binxmin, int binxmax) const;
  float integral(int ch) const;
  bool hasEnoughData(int minEntries) const;

  float getRange() const { return mRange; }
  void setRange(float r) { mRange = r; }

  int getNbins() const { return mNBins; }
  void setNbins(int nb) { mNBins = nb; }

  boostHisto& getHisto(int isect) { return mHisto[isect]; }
  const boostHisto& getHisto(int isect) const { return mHisto[isect]; }
  //const boostHisto getHisto() const { return &mHisto[0]; }
  // boostHisto* getHisto(int isect) const { return &mHisto[isect]; }

  std::vector<int> getEntriesPerChannel() const { return mEntries; }

 private:
  float mRange = o2::tof::Geo::BC_TIME_INPS * 0.5;
  int mNBins = 1000;
  float mV2Bin;
  std::array<boostHisto, 18> mHisto;
  std::vector<int> mEntries; // vector containing number of entries per channel

  CalibTOFapi* mCalibTOFapi = nullptr; // calibTOFapi to correct the t-text
  int mNElsPerSector = o2::tof::Geo::NPADSXSECTOR;

  ClassDefNV(TOFChannelData, 1);
};

template <class T>
class TOFChannelCalibrator final : public o2::calibration::TimeSlotCalibration<T, o2::tof::TOFChannelData>
{
  using TFType = uint64_t;
  using Slot = o2::calibration::TimeSlot<o2::tof::TOFChannelData>;
  using CalibTOFapi = o2::tof::CalibTOFapi;
  using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
  using CcdbObjectInfo = o2::ccdb::CcdbObjectInfo;
  using CcdbObjectInfoVector = std::vector<CcdbObjectInfo>;
  using TimeSlewingVector = std::vector<TimeSlewing>;

 protected:
  std::deque<o2::calibration::TimeSlot<o2::tof::TOFChannelData>>& getSlots() { return o2::calibration::TimeSlotCalibration<T, o2::tof::TOFChannelData>::getSlots(); }

 public:
  static double FuncDeltaOffset(double* x, double* params)
  {
    int i1 = int(x[0]) % 96;
    int i2 = int(x[0]) / 96 ? (i1 + 48) : (i1 + 1);

    if (i1 < 0) {
      return 0;
    }
    if (i2 >= Geo::NPADS) {
      return 0;
    }

    return (params[i1] - params[i2]);
  }

  static constexpr int NCOMBINSTRIP = o2::tof::Geo::NPADX + o2::tof::Geo::NPADS;

  TOFChannelCalibrator(int minEnt = 500, int nb = 1000, float r = 24400) : mMinEntries(minEnt), mNBins(nb), mRange(r){};

  ~TOFChannelCalibrator() final = default;

  bool hasEnoughData(const Slot& slot) const final
  {
    // Checking if all channels have enough data to do calibration.
    // Delegating this to TOFChannelData
    const o2::tof::TOFChannelData* c = slot.getContainer();
    LOG(DEBUG) << "Checking statistics";
    return (mTest ? true : c->hasEnoughData(mMinEntries));
  }

  void initOutput() final
  {
    // Here we initialize the vector of our output objects
    mInfoVector.clear();
    mTimeSlewingVector.clear();
    return;
  }

  void finalizeSlot(Slot& slot) final
  {
    // here we simply decide which finalize to call: for the use case with Tracks or cosmics
    mCalibWithCosmics ? finalizeSlotWithCosmics(slot) : finalizeSlotWithTracks(slot);
    return;
  }

  void finalizeSlotWithCosmics(Slot& slot)
  {
    // Extract results for the single slot
    o2::tof::TOFChannelData* c = slot.getContainer();
    LOG(INFO) << "Finalize slot for calibration with cosmics " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

    if (!mFuncDeltaOffset) {
      mFuncDeltaOffset = new TF1("fChanOffset", FuncDeltaOffset, 0, NCOMBINSTRIP, Geo::NPADS);
      mFuncDeltaOffset->FixParameter(24, 0); // fix one pad to fixed offset (as reference)
      for (int ichLocal = 0; ichLocal < Geo::NPADS; ichLocal++) {
        if (ichLocal == 24) {
          continue; //already fixed
          mFuncDeltaOffset->SetParLimits(ichLocal, -mRange, mRange);
        }
      }
    }

    // for the CCDB entry
    std::map<std::string, std::string> md;
    TimeSlewing& ts = mCalibTOFapi->getSlewParamObj(); // we take the current CCDB object, since we want to simply update the offset

    float xp[NCOMBINSTRIP], exp[NCOMBINSTRIP], deltat[NCOMBINSTRIP], edeltat[NCOMBINSTRIP], fracUnderPeak[Geo::NPADS];

    for (int sector = 0; sector < Geo::NSECTORS; sector++) {
      int offsetsector = sector * Geo::NSTRIPXSECTOR * Geo::NPADS;
      for (int istrip = 0; istrip < Geo::NSTRIPXSECTOR; istrip++) {
        int offsetstrip = istrip * Geo::NPADS + offsetsector;
        int goodpoints = 0;

        memset(&fracUnderPeak[0], 0, sizeof(fracUnderPeak));

        for (int ipair = 0; ipair < NCOMBINSTRIP; ipair++) {
          int chinsector = ipair + istrip * NCOMBINSTRIP;
          int ich = chinsector + sector * Geo::NSTRIPXSECTOR * NCOMBINSTRIP;
          auto entriesInPair = c->integral(ich);
          if (entriesInPair < mMinEntries && entriesInPair != 0) {
            LOG(DEBUG) << "pair " << ich << " will not be calibrated since it has only " << entriesInPair << " entries (min = " << mMinEntries << ")";
            continue;
          }
          // make the slice of the 2D histogram so that we have the 1D of the current channel
          std::vector<float> fitValues;
          std::vector<float> histoValues;
          std::vector<int> entriesPerChannel = c->getEntriesPerChannel();
          if (entriesPerChannel.at(ich) == 0) {
            continue; // skip always since a channel with 0 entries is normal, it will be flagged as problematic
            if (mTest) {
              LOG(INFO) << "Skipping channel " << ich << " because it has zero entries, but it should not be"; // should become error!
              continue;
            } else {
              throw std::runtime_error("We found one channel with no entries, we cannot calibrate!");
            }
          }

          // more efficient way
          auto histo = c->getHisto(sector);
          for (unsigned j = chinsector; j <= chinsector; ++j) {
            for (unsigned i = 0; i < c->getNbins(); ++i) {
              const auto& v = histo.at(i, j);
              LOG(DEBUG) << "channel = " << ich << ", in sector = " << sector << " (where it is channel = " << chinsector << ") bin = " << i << " value = " << v;
              histoValues.push_back(v);
            }
          }

          double fitres = fitGaus(c->getNbins(), histoValues.data(), -(c->getRange()), c->getRange(), fitValues);

          if (fitres >= 0) {
            LOG(DEBUG) << "Pair " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
          } else {
            //LOG(INFO) << "Pair " << ich << " :: Fit failed with result = " << fitres;
            continue;
          }

          if (fitValues[2] < 0) {
            fitValues[2] = -fitValues[2];
          }

          float intmin = fitValues[1] - 5 * fitValues[2]; // mean - 5*sigma
          float intmax = fitValues[1] + 5 * fitValues[2]; // mean + 5*sigma

          if (intmin < -mRange) {
            intmin = -mRange;
          }
          if (intmax < -mRange) {
            intmax = -mRange;
          }
          if (intmin > mRange) {
            intmin = mRange;
          }
          if (intmax > mRange) {
            intmax = mRange;
          }

          xp[goodpoints] = ipair + 0.5;      // pair index
          exp[goodpoints] = 0.0;             // error on pair index (dummy since it is on the pair index)
          deltat[goodpoints] = fitValues[1]; // delta between offsets from channels in pair (from the fit) - in ps
          edeltat[goodpoints] = 20;          // TODO: for now put by default to 20 ps since it was seen to be reasonable; but it should come from the fit: who gives us the error from the fit ??????
          goodpoints++;
          int ch1 = ipair % 96;
          int ch2 = ipair / 96 ? ch1 + 48 : ch1 + 1;
          float fractionUnderPeak = entriesInPair > 0 ? c->integral(ich, intmin, intmax) / entriesInPair : 0;
          // we keep as fractionUnderPeak of the channel the largest one that is found in the 3 possible pairs with that channel (for both channels ch1 and ch2 in the pair)
          if (fracUnderPeak[ch1] < fractionUnderPeak) {
            fracUnderPeak[ch1] = fractionUnderPeak;
          }
          if (fracUnderPeak[ch2] < fractionUnderPeak) {
            fracUnderPeak[ch2] = fractionUnderPeak;
          }
        } // end loop pairs

        // fit strip offset
        if (goodpoints == 0) {
          //LOG(INFO) << "We did not find any good point for strip " << istrip << " in sector " << sector;
          continue;
        }
        LOG(DEBUG) << "We found " << goodpoints << " good points for strip " << istrip << " in sector " << sector << " --> we can fit the TGraph";
        TGraphErrors g(goodpoints, xp, deltat, exp, edeltat);
        g.Fit(mFuncDeltaOffset, "Q0");

        //update calibrations
        for (int ichLocal = 0; ichLocal < Geo::NPADS; ichLocal++) {
          int ich = ichLocal + offsetstrip;
          ts.updateOffsetInfo(ich, mFuncDeltaOffset->GetParameter(ichLocal));
          ts.setFractionUnderPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, fracUnderPeak[ichLocal]);
          ts.setSigmaPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, abs(mFuncDeltaOffset->GetParError(ichLocal)));
        }

      } // end loop strips
    }   // end loop sectors

    auto clName = o2::utils::MemFileHelper::getClassName(ts);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    mInfoVector.emplace_back("TOF/ChannelCalib", clName, flName, md, slot.getTFStart(), 99999999999999);
    mTimeSlewingVector.emplace_back(ts);
  }

  void finalizeSlotWithTracks(Slot& slot)
  {
    // Extract results for the single slot
    o2::tof::TOFChannelData* c = slot.getContainer();
    LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

    // for the CCDB entry
    std::map<std::string, std::string> md;
    TimeSlewing& ts = mCalibTOFapi->getSlewParamObj(); // we take the current CCDB object, since we want to simply update the offset

    for (int ich = 0; ich < Geo::NCHANNELS; ich++) {
      // make the slice of the 2D histogram so that we have the 1D of the current channel
      int sector = ich / Geo::NPADSXSECTOR;
      int chinsector = ich % Geo::NPADSXSECTOR;
      auto entriesInChannel = c->integral(ich);
      if (entriesInChannel < mMinEntries) {
        LOG(DEBUG) << "channel " << ich << " will not be calibrated since it has only " << entriesInChannel << " entries (min = " << mMinEntries << ")";
        continue;
      }
      std::vector<float> fitValues;
      std::vector<float> histoValues;
      std::vector<int> entriesPerChannel = c->getEntriesPerChannel();
      if (entriesPerChannel.at(ich) == 0) {
        continue; // skip always since a channel with 0 entries is normal, it will be flagged as problematic
        if (mTest) {
          LOG(DEBUG) << "Skipping channel " << ich << " because it has zero entries, but it should not be"; // should become error!
          continue;
        } else {
          throw std::runtime_error("We found one channel with no entries, we cannot calibrate!");
        }
      }

      // more efficient way
      auto histo = c->getHisto(sector);
      for (unsigned j = chinsector; j <= chinsector; ++j) {
        for (unsigned i = 0; i < c->getNbins(); ++i) {
          const auto& v = histo.at(i, j);
          LOG(DEBUG) << "channel = " << ich << ", in sector = " << sector << " (where it is channel = " << chinsector << ") bin = " << i << " value = " << v;
          histoValues.push_back(v);
        }
      }

      double fitres = fitGaus(c->getNbins(), histoValues.data(), -(c->getRange()), c->getRange(), fitValues);

      if (fitres >= 0) {
        LOG(DEBUG) << "Channel " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
      } else {
        //        LOG(INFO) << "Channel " << ich << " :: Fit failed with result = " << fitres;
        continue;
      }

      if (fitValues[2] < 0) {
        fitValues[2] = -fitValues[2];
      }

      float fractionUnderPeak;
      float intmin = fitValues[1] - 5 * fitValues[2]; // mean - 5*sigma
      float intmax = fitValues[1] + 5 * fitValues[2]; // mean + 5*sigma

      if (intmin < -mRange) {
        intmin = -mRange;
      }
      if (intmax < -mRange) {
        intmax = -mRange;
      }
      if (intmin > mRange) {
        intmin = mRange;
      }
      if (intmax > mRange) {
        intmax = mRange;
      }

      fractionUnderPeak = entriesInChannel > 0 ? c->integral(ich, intmin, intmax) / entriesInChannel : 0;
      // now we need to store the results in the TimeSlewingObject
      ts.setFractionUnderPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, fractionUnderPeak);
      ts.setSigmaPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, abs(fitValues[2]));
      ts.updateOffsetInfo(ich, fitValues[1]);
    }
    auto clName = o2::utils::MemFileHelper::getClassName(ts);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    mInfoVector.emplace_back("TOF/ChannelCalib", clName, flName, md, slot.getTFStart(), 99999999999999);
    mTimeSlewingVector.emplace_back(ts);
  }

  Slot& emplaceNewSlot(bool front, TFType tstart, TFType tend) final
  {
    auto& cont = getSlots();
    auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
    int nElements = mCalibWithCosmics ? NCOMBINSTRIP * Geo::NSTRIPXSECTOR : Geo::NPADSXSECTOR; // if we calibrate with cosmics, we pass the number of possible combinations per sector; otherwise, the number of pads per sector
    slot.setContainer(std::make_unique<TOFChannelData>(mNBins, mRange, mCalibTOFapi, nElements));
    return slot;
  }

  const TimeSlewingVector& getTimeSlewingVector() const { return mTimeSlewingVector; }
  const CcdbObjectInfoVector& getTimeSlewingInfoVector() const { return mInfoVector; }
  CcdbObjectInfoVector& getTimeSlewingInfoVector() { return mInfoVector; }

  void setIsTest(bool isTest) { mTest = isTest; }
  bool isTest() const { return mTest; }

  void setCalibTOFapi(CalibTOFapi* api) { mCalibTOFapi = api; }
  CalibTOFapi* getCalibTOFapi() const { return mCalibTOFapi; }

  void setRange(float r) { mRange = r; }
  float getRange() const { return mRange; }

  void setDoCalibWithCosmics(bool doCalibWithCosmics = true) { mCalibWithCosmics = doCalibWithCosmics; }
  bool doCalibWithCosmics() const { return mCalibWithCosmics; }

 private:
  int mMinEntries = 0; // min number of entries to calibrate the TimeSlot
  int mNBins = 0;      // bins of the histogram with the t-text per channel
  float mRange = 0.;   // range of the histogram with the t-text per channel
  bool mTest = false;  // flag to be used when running in test mode: it simplify the processing (e.g. does not go through all channels)
  TF1* mFuncDeltaOffset = nullptr;

  CalibTOFapi* mCalibTOFapi = nullptr; // CalibTOFapi needed to get the previous calibrations read from CCDB (do we need that it is a pointer?)

  // output
  CcdbObjectInfoVector mInfoVector;     // vector of CCDB Infos , each element is filled with the CCDB description of the accompanying TimeSlewing object
  TimeSlewingVector mTimeSlewingVector; // vector of TimeSlewing, each element is filled in "process"
                                        // when we finalize one slot (multiple can be finalized
                                        // during the same "process", which is why we have a vector).
                                        // Each element is to be considered the output of the device,
                                        // and will go to the CCDB. Note that for the channel offset
                                        // we still fill the TimeSlewing object

  bool mCalibWithCosmics = false; // flag to indicate whether we are calibrating with cosmics

  ClassDefOverride(TOFChannelCalibrator, 1);
};

} // end namespace tof
} // end namespace o2

#endif /* TOF_CHANNEL_CALIBRATOR_H_ */
