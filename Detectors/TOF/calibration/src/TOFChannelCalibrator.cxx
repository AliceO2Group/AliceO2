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

#include "TOFCalibration/TOFChannelCalibrator.h"
#include "Framework/Logger.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <TStopwatch.h>
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Math/WrappedMultiTF1.h"

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace o2
{
namespace tof
{

using Slot = o2::calibration::TimeSlot<o2::tof::TOFChannelData>;
using TimeSlewing = o2::dataformats::CalibTimeSlewingParamTOF;
using clbUtils = o2::calibration::Utils;
using boost::histogram::indexed;
using namespace o2::tof;
//using boost::histogram::algorithm; // not sure why it does not work...

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::dataformats::CalibInfoTOF> data)
{
  // check on multiplicity to avoid noisy events (SAFE MODE)
  static int ntf = 0;
  static float sumDt = 0;

  if (mSafeMode) {
    float sumdt = 0;
    int ngood = 0;
    for (int j = 0; j < data.size(); j++) {
      float dtraw = data[j].getDeltaTimePi();
      float dt = mCalibTOFapi->getTimeCalibration(data[j].getTOFChIndex(), data[j].getTot());
      if (dt > -50000 && dt < 50000) {
        sumdt += dt;
        ngood++;
      }
    }
    if (ngood) {
      sumdt /= ngood;
      sumDt += sumdt;
      ntf++;
    }

    if (ntf && (sumdt > 5000 + sumDt / ntf || sumdt < -5000 + sumDt / ntf)) { // skip TF since it is very far from average behaviour (probably noise if detector already partial calibrated)
      return;
    }
  }

  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getTOFChIndex();
    int sector = ch / Geo::NPADSXSECTOR;
    int chInSect = ch % Geo::NPADSXSECTOR;
    auto dt = data[i].getDeltaTimePi();

    auto tot = data[i].getTot();
    // TO BE DISCUSSED: could it be that the LHCphase is too old? If we ar ein sync mode, it could be that it is not yet created for the current run, so the one from the previous run (which could be very old) is used. But maybe it does not matter much, since soon enough a calibrated LHC phase should be produced
    auto corr = mCalibTOFapi->getTimeCalibration(ch, tot); // we take into account LHCphase, offsets and time slewing
    auto dtcorr = dt - corr;

    int used = o2::tof::Utils::addMaskBC(data[i].getMask(), data[i].getTOFChIndex()); // fill the current BC candidate mask and return the one used
    dtcorr -= used * o2::tof::Geo::BC_TIME_INPS;

    // add calib info for computation of LHC phase
    Utils::addCalibTrack(dtcorr);

    // uncomment to enable auto correction of LHC phase
    //    dtcorr -= Utils::mLHCPhase;

    LOG(debug) << "Residual LHCphase = " << Utils::mLHCPhase;

#ifdef DEBUGGING
    mChannelDist->Fill(ch, dtcorr);
#endif

    if (!mPerStrip) {
      mHisto[sector](dtcorr, chInSect); // we pass the calibrated time
      mEntries[ch] += 1;
    } else {
      int istrip = ch / 96;
      int istripInSector = chInSect / 96;
      int halffea = (chInSect % 96) / 12;
      int choffset = (istrip - istripInSector) * 96;
      //int minch = istripInSector * 96;
      //int maxch = minch + 96;
      int minch = istripInSector * 96 + halffea * 12;
      int maxch = minch + 12;
      for (int ich = minch; ich < maxch; ich++) {
        mHisto[sector](dtcorr, ich); // we pass the calibrated time
        mEntries[ich + choffset] += 1;
      }
    }
  }
}

//_____________________________________________
void TOFChannelData::fill(const gsl::span<const o2::tof::CalibInfoCluster> data)
{
  // fill container
  for (int i = data.size(); i--;) {
    auto ch = data[i].getCH();
    auto dch = data[i].getDCH(); // this is a char! if you print it, you need to cast it to int
    auto dt = data[i].getDT();
    auto tot1 = data[i].getTOT1();
    auto tot2 = data[i].getTOT2();

    // we order them so that the channel number of the first cluster is smaller than
    // the one of the second cluster
    if (dch < 0) {
      ch += dch;
      dt = -dt;
      dch = -dch;
      float inv = tot1;
      tot1 = tot2;
      tot2 = inv;
    }

    int sector = ch / Geo::NPADSXSECTOR;
    int absoluteStrip = ch / Geo::NPADS;
    int stripInSect = absoluteStrip % Geo::NSTRIPXSECTOR;
    int shift = 0;
    if (dch == 1) {
      shift = 0; // 2nd channel is on the right
    } else if (dch == 48) {
      shift = 1; // 2nd channel is at the top
    } else {
      continue;
    }
    int chOnStrip = ch % 96;
    int comb = 96 * shift + chOnStrip; // index of the current pair of clusters on the strip; there are in total 96 + 48

    auto corr1 = mCalibTOFapi->getTimeCalibration(ch, tot1);       // we take into account LHCphase, offsets and time slewing
    auto corr2 = mCalibTOFapi->getTimeCalibration(ch + dch, tot2); // we take into account LHCphase, offsets and time slewing
    LOG(debug) << "inserting in channel " << ch << ", " << ch + dch << ": dt = " << dt << ", tot1 = " << tot1 << ", tot2 = " << tot2 << ", corr1 = " << corr1 << ", corr2 = " << corr2 << ", corrected dt = " << dt - corr1 + corr2;

    dt -= corr1 - corr2;

    int combInSect = comb + stripInSect * NCOMBINSTRIP;

    LOG(debug) << "ch = " << ch << ", sector = " << sector << ", absoluteStrip = " << absoluteStrip << ", stripInSect = " << stripInSect << ", shift = " << shift << ", dch = " << (int)dch << ", chOnStrip = " << chOnStrip << ", comb = " << comb;

    mHisto[sector](dt, combInSect); // we pass the difference of the *calibrated* times
    mEntries[comb + NCOMBINSTRIP * absoluteStrip] += 1;

#ifdef DEBUGGING
    mChannelDist->Fill(comb + NCOMBINSTRIP * absoluteStrip, dt);
#endif
  }
}

//_____________________________________________
void TOFChannelData::merge(const TOFChannelData* prev)
{
  // merge data of 2 slots
  for (int isect = 0; isect < Geo::NSECTORS; isect++) {
    mHisto[isect] += prev->getHisto(isect);
  }
  for (auto iel = 0; iel < mEntries.size(); iel++) {
    mEntries[iel] += prev->mEntries[iel];
  }
}

//_____________________________________________
bool TOFChannelData::hasEnoughData(int minEntries) const
{
  // true if all channels can be fitted --> have enough statistics

  // We consider that we have enough entries if the mean of the number of entries in the channels
  // with at least one entry is greater than the cut at "minEntries"
  // Channels/pairs with zero entries are assumed to be off --> we do not consider them

  //printEntries();
  int nValid = 0;
  float mean = 0;
  int smallestElementIndex = -1;
  int smallestEntries = 1e5;
  int largestElementIndex = -1;
  int largestEntries = 0;
  for (auto i = 0; i < mEntries.size(); ++i) {
    if (mEntries[i] != 0) { // skipping channels/pairs if they have zero entries (most likely they are simply off)
      mean += mEntries[i];
      ++nValid;
      if (mEntries[i] < smallestEntries) {
        smallestEntries = mEntries[i];
        smallestElementIndex = i;
      }
      if (mEntries[i] > largestEntries) {
        largestEntries = mEntries[i];
        largestElementIndex = i;
      }
    }
  }
  if (nValid == 0) {
    LOG(info) << "hasEnough = false: all channels/pairs are empty";
    return false;
  }

  LOG(debug) << "mean = " << mean << ", nvalid = " << nValid;
  mean /= nValid;

  LOG(debug) << "minElement is at position " << smallestElementIndex << " and has " << smallestEntries << " entries";
  LOG(debug) << "maxElement is at position " << largestElementIndex << " and has " << largestEntries << " entries";
  float threshold = minEntries + 5 * std::sqrt(minEntries);
  bool enough = mean < threshold ? false : true;
  if (enough) {
    LOG(info) << "hasEnough: " << (int)enough << " ("
              << nValid << " valid channels found (should be " << mEntries.size() << ") with mean = "
              << mean << " with cut at = " << threshold << ") ";
  }
  return enough;
}

//_____________________________________________
void TOFChannelData::print() const
{
  LOG(info) << "Printing histograms:";
  std::ostringstream os;
  for (int isect = 0; isect < Geo::NSECTORS; isect++) {
    LOG(info) << "Sector: " << isect;
    os << mHisto[isect];
    auto nentriesInSec = boost::histogram::algorithm::sum(mHisto[isect]);
    LOG(info) << "Number of entries in histogram: " << boost::histogram::algorithm::sum(mHisto[isect]);
    int cnt = 0;
    if (nentriesInSec != 0) {
      for (auto&& x : indexed(mHisto[isect])) {
        if (x.get() > 0) {
          const auto i = x.index(0); // current index along first axis --> t-texp
          const auto j = x.index(1); // current index along second axis --> channel
          const auto b0 = x.bin(0);  // current bin interval along first axis --> t-texp
          const auto b1 = x.bin(1);  // current bin interval along second axis --> channel
          LOG(info) << "bin " << cnt << ": channel = " << j << " in [" << b1.lower() << ", " << b1.upper()
                    << "], t-texp in [" << b0.lower() << ", " << b0.upper() << "], has entries = " << x.get();
        }
        cnt++;
      }
    }
    LOG(info) << cnt << " bins inspected";
  }
}

//_____________________________________________
void TOFChannelData::print(int isect) const
{
  LOG(info) << "*** Printing histogram " << isect;
  std::ostringstream os;
  int cnt = 0;
  os << mHisto[isect];
  LOG(info) << "Number of entries in histogram: " << boost::histogram::algorithm::sum(mHisto[isect]);
  for (auto&& x : indexed(mHisto[isect])) { // does not work also when I use indexed(*(mHisto[sector]))
    cnt++;
    LOG(debug) << " c " << cnt << " i " << x.index(0) << " j " << x.index(1) << " b0 " << x.bin(0) << " b1 " << x.bin(1) << " val= " << *x << "|" << x.get();
    if (x.get() > 0) {
      LOG(info) << "x = " << x.get() << " c " << cnt;
    }
  }
  LOG(info) << cnt << " bins inspected";
}

//_____________________________________________
void TOFChannelData::printEntries() const
{
  // to print number of entries per channel
  for (int i = 0; i < mEntries.size(); ++i) {
    if (mEntries.size() > tof::Geo::NCHANNELS) {
      LOG(info) << "pair of channels " << i << " has " << mEntries[i] << " entries";
    } else {
      LOG(info) << "channel " << i << " has " << mEntries[i] << " entries";
    }
  }
}

//_____________________________________________
int TOFChannelData::findBin(float v) const
{
  // find the bin along the x-axis (with t-texp) where the value "v" is; this does not depend on the channel
  // (axis 1), nor on the sector, so we use sector0

  if (v == mRange) {
    v -= 1.e-1;
  }

  LOG(debug) << "In FindBin, v = : " << v;
  LOG(debug) << "bin0 limits: lower = " << mHisto[0].axis(0).bin(0).lower() << ", upper = " << mHisto[0].axis(0).bin(0).upper();
  LOG(debug) << "bin1000 limits: lower = " << mHisto[0].axis(0).bin(mNBins - 1).lower() << ", upper = " << mHisto[0].axis(0).bin(mNBins - 1).upper();
  LOG(debug) << "v = " << v << " is in bin " << mHisto[0].axis(0).index(v);

  return mHisto[0].axis(0).index(v);
}

//_____________________________________________
float TOFChannelData::integral(int chmin, int chmax, float binmin, float binmax) const
{
  // calculates the integral in [chmin, chmax] and in [binmin, binmax]

  if (binmin < -mRange || binmax > mRange || chmin < 0 || chmax >= Geo::NSECTORS * mNElsPerSector) {
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  }
  if (binmax < binmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / mNElsPerSector;
  if (sector != chmax / mNElsPerSector) {
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");
  }

  int chinsectormin = chmin % mNElsPerSector;
  int chinsectormax = chmax % mNElsPerSector;

  float res2 = 0;
  //TStopwatch t3;
  int ind = -1;
  int binxmin = findBin(binmin);
  int binxmax = findBin(binmax);
  LOG(debug) << "binxmin = " << binxmin << ", binxmax = " << binxmax;
  //t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  }
  //t3.Stop();
  LOG(debug) << "Time for integral looping over axis (result = " << res2 << "):";
  //t3.Print();

  return res2;

}

//_____________________________________________
float TOFChannelData::integral(int ch, float binmin, float binmax) const
{
  // calculates the integral along one fixed channel and in [binmin, binmax]

  return integral(ch, ch, binmin, binmax);
}

//_____________________________________________
float TOFChannelData::integral(int chmin, int chmax, int binxmin, int binxmax) const
{
  // calculates the integral in [chmin, chmax] and in [binmin, binmax]

  if (binxmin < 0 || binxmax > mNBins || chmin < 0 || chmax >= Geo::NSECTORS * mNElsPerSector) {
    throw std::runtime_error("Check your bins, we cannot calculate the integrals in under/overflows bins");
  }
  if (binxmax < binxmin || chmax < chmin) {
    throw std::runtime_error("Check your bin limits!");
  }

  int sector = chmin / mNElsPerSector;
  if (sector != chmax / mNElsPerSector) {
    throw std::runtime_error("We cannot integrate over channels that belong to different sectors");
  }

  int chinsectormin = chmin % mNElsPerSector;
  int chinsectormax = chmax % mNElsPerSector;

  float res2 = 0;
  //TStopwatch t3;
  //t3.Start();
  for (unsigned j = chinsectormin; j <= chinsectormax; ++j) {
    for (unsigned i = binxmin; i <= binxmax; ++i) {
      const auto& v = mHisto[sector].at(i, j);
      res2 += v;
    }
  }
  //t3.Stop();
  LOG(debug) << "Time for integral looping over axis (result = " << res2 << "):";
  //t3.Print();
  return res2;

}

//_____________________________________________
float TOFChannelData::integral(int ch, int binxmin, int binxmax) const
{
  // calculates the integral along one fixed channel and in [binmin, binmax]

  return integral(ch, ch, binxmin, binxmax);
}

//_____________________________________________
float TOFChannelData::integral(int ch) const
{
  // calculates the integral along one fixed channel and in the full x-range

  return mEntries.at(ch);
}

//_____________________________________________
void TOFChannelData::resetAndReRange(float range)
{
  // empty the container and redefine the range

  setRange(range);
  std::fill(mEntries.begin(), mEntries.end(), 0);
  mV2Bin = mNBins / (2 * mRange);
  for (int isect = 0; isect < 18; isect++) {
    mHisto[isect] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(mNBins, -mRange, mRange, "t-texp"),
                                                     boost::histogram::axis::integer<>(0, mNElsPerSector, "channel index in sector" + std::to_string(isect)));
  }
  return;
}

//-------------------------------------------------------------------
// TOF Channel Calibrator
//-------------------------------------------------------------------

template <typename T>
void TOFChannelCalibrator<T>::finalizeSlotWithCosmics(Slot& slot)
{
  // Extract results for the single slot

  o2::tof::TOFChannelData* c = slot.getContainer();
  LOG(info) << "Finalize slot for calibration with cosmics " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  // for the CCDB entry
  std::map<std::string, std::string> md;
  TimeSlewing& ts = mCalibTOFapi->getSlewParamObj(); // we take the current CCDB object, since we want to simply update the offset
                                                     //  ts.bind();

  int nbins = c->getNbins();
  float range = c->getRange();
  std::vector<int> entriesPerChannel = c->getEntriesPerChannel();

#ifdef WITH_OPENMP
  if (mNThreads < 1) {
    mNThreads = std::min(omp_get_max_threads(), NMAXTHREADS);
  }
  LOG(debug) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#else
  mNThreads = 1;
#endif
  for (int sector = 0; sector < Geo::NSECTORS; sector++) {
    TMatrixD mat(3, 3);
    int ithread = 0;
#ifdef WITH_OPENMP
    ithread = omp_get_thread_num();
#endif

    LOG(info) << "Processing sector " << sector << " with thread " << ithread;
    double xp[NCOMBINSTRIP], exp[NCOMBINSTRIP], deltat[NCOMBINSTRIP], edeltat[NCOMBINSTRIP];

    std::array<double, 3> fitValues;
    std::vector<float> histoValues;

    auto& histo = c->getHisto(sector);

    int offsetPairInSector = sector * Geo::NSTRIPXSECTOR * NCOMBINSTRIP;
    int offsetsector = sector * Geo::NSTRIPXSECTOR * Geo::NPADS;
    for (int istrip = 0; istrip < Geo::NSTRIPXSECTOR; istrip++) {
      LOG(info) << "Processing strip " << istrip;
      double fracUnderPeak[Geo::NPADS] = {0.};
      bool isChON[96] = {false};
      int offsetstrip = istrip * Geo::NPADS + offsetsector;
      int goodpoints = 0;
      int allpoints = 0;

      TLinearFitter localFitter(1, mStripOffsetFunction.c_str());

      int offsetPairInStrip = istrip * NCOMBINSTRIP;

      localFitter.StoreData(kFALSE);
      localFitter.ClearPoints();
      for (int ipair = 0; ipair < NCOMBINSTRIP; ipair++) {
        int chinsector = ipair + offsetPairInStrip;
        int ich = chinsector + offsetPairInSector;
        auto entriesInPair = entriesPerChannel.at(ich);
        xp[allpoints] = ipair + 0.5; // pair index

        if (entriesInPair == 0) {
          localFitter.AddPoint(&(xp[allpoints]), 0.0, 1.0);
          allpoints++;
          continue; // skip always since a channel with 0 entries is normal, it will be flagged as problematic
        }
        if (entriesInPair < mMinEntries) {
          LOG(debug) << "pair " << ich << " will not be calibrated since it has only " << entriesInPair << " entries (min = " << mMinEntries << ")";
          localFitter.AddPoint(&(xp[allpoints]), 0.0, 1.0);
          allpoints++;
          continue;
        }
        fitValues.fill(-99999999);
        histoValues.clear();

        // make the slice of the 2D histogram so that we have the 1D of the current channel
        for (unsigned i = 0; i < nbins; ++i) {
          const auto& v = histo.at(i, chinsector);
          LOG(debug) << "channel = " << ich << ", in sector = " << sector << " (where it is channel = " << chinsector << ") bin = " << i << " value = " << v;
          histoValues.push_back(v);
        }

        double fitres = entriesInPair - 1;
        fitres = fitGaus(nbins, histoValues.data(), -range, range, fitValues, nullptr, 2., true);
        if (fitres >= 0) {
          LOG(debug) << "Pair " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
        } else {
          LOG(debug) << "Pair " << ich << " :: Fit failed with result = " << fitres;
          localFitter.AddPoint(&(xp[allpoints]), 0.0, 1.0);
          allpoints++;
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

        xp[allpoints] = ipair + 0.5;      // pair index
        exp[allpoints] = 0.0;             // error on pair index (dummy since it is on the pair index)
        deltat[allpoints] = fitValues[1]; // delta between offsets from channels in pair (from the fit) - in ps
        float integral = c->integral(ich, intmin, intmax);
        edeltat[allpoints] = 20 + fitValues[2] / sqrt(integral); // TODO: for now put by default to 20 ps since it was seen to be reasonable; but it should come from the fit: who gives us the error from the fit ??????
        localFitter.AddPoint(&(xp[allpoints]), deltat[allpoints], edeltat[allpoints]);
        goodpoints++;
        allpoints++;
        int ch1 = ipair % 96;
        int ch2 = ipair / 96 ? ch1 + 48 : ch1 + 1;
        isChON[ch1] = true;
        isChON[ch2] = true;

        float fractionUnderPeak = entriesInPair > 0 ? integral / entriesInPair : 0;
        // we keep as fractionUnderPeak of the channel the largest one that is found in the 3 possible pairs with that channel (for both channels ch1 and ch2 in the pair)
        if (fracUnderPeak[ch1] < fractionUnderPeak) {
          fracUnderPeak[ch1] = fractionUnderPeak;
        }
        if (fracUnderPeak[ch2] < fractionUnderPeak) {
          fracUnderPeak[ch2] = fractionUnderPeak;
        }

#ifdef DEBUGGING
//        mFitCal->Fill(ipair + offsetPairInStrip + offsetPairInSector, fitValues[1]);
#endif

      } // end loop pairs

      // fit strip offset
      if (goodpoints == 0) {
        continue;
      }
      LOG(debug) << "We found " << goodpoints << " good points for strip " << istrip << " in sector " << sector << " --> we can fit the TGraph";

      bool isFirst = true;
      int nparams = 0;
      LOG(debug) << "N parameters before fixing = " << localFitter.GetNumberFreeParameters();

      // we fix to zero the parameters that have no entry, plus the first one that we find, which we will use as reference for the other offsets
      for (int i = 0; i < 96; ++i) {
        if (isChON[i]) {
          if (isFirst) {
            localFitter.FixParameter(i, 0.);
            isFirst = false;
          } else {
            nparams++;
          }
        } else {
          localFitter.FixParameter(i, 0.);
        }
      }

      LOG(debug) << "Strip = " << istrip << " fitted by thread = " << ithread << ", goodpoints = " << goodpoints << ", number of free parameters = "
                 << localFitter.GetNumberFreeParameters() << ",  NDF = " << goodpoints - localFitter.GetNumberFreeParameters();

      if (goodpoints <= localFitter.GetNumberFreeParameters()) {
        LOG(debug) << "Skipped";
        continue;
      }

      LOG(debug) << "N real params = " << nparams << ", fitter has " << localFitter.GetNumberFreeParameters() << " free parameters, " << localFitter.GetNumberTotalParameters() << " total parameters, " << localFitter.GetNpoints() << " points";
      LOG(info) << "Sector = " << sector << ", strip = " << istrip << " fitted by thread = " << ithread << ": ready to fit";
      localFitter.Eval();

      LOG(info) << "Sector = " << sector << ", strip = " << istrip << " fitted by thread = " << ithread << " with Chi/NDF " << localFitter.GetChisquare() << "/" << goodpoints - localFitter.GetNumberFreeParameters();
      LOG(debug) << "Strip = " << istrip << " fitted by thread = " << ithread << " with Chi/NDF " << localFitter.GetChisquare() << "/" << goodpoints - localFitter.GetNumberFreeParameters();

      //      if(localFitter.GetChisquare() > (goodpoints - localFitter.GetNumberFreeParameters())*10){
      //        continue;
      //      }

      //update calibrations
      for (int ichLocal = 0; ichLocal < Geo::NPADS; ichLocal++) {
        int ich = ichLocal + offsetstrip;
        ts.updateOffsetInfo(ich, localFitter.GetParameter(ichLocal));
#ifdef DEBUGGING
        mFitCal->Fill(ich, localFitter.GetParameter(ichLocal));
#endif
        ts.setFractionUnderPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, fracUnderPeak[ichLocal]);
        ts.setSigmaPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, abs(std::sqrt(localFitter.GetCovarianceMatrixElement(ichLocal, ichLocal))));
      }

    } // end loop strips
  }   // end loop sectors

  auto clName = o2::utils::MemFileHelper::getClassName(ts);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  auto startValidity = slot.getStartTimeMS();
  auto endValidity = o2::ccdb::CcdbObjectInfo::MONTH * 2;
  ts.setStartValidity(startValidity);
  ts.setEndValidity(endValidity);
  mInfoVector.emplace_back("TOF/Calib/ChannelCalib", clName, flName, md, startValidity, endValidity);
  mTimeSlewingVector.emplace_back(ts);

#ifdef DEBUGGING
  TFile fout("debug_tof_cal.root", "RECREATE");
  mFitCal->Write();
  mChannelDist->Write();
  fout.Close();
#endif
}

//_____________________________________________

template <typename T>
void TOFChannelCalibrator<T>::finalizeSlotWithTracks(Slot& slot)
{
  // Extract results for the single slot
  o2::tof::TOFChannelData* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  int nbins = c->getNbins();
  float range = c->getRange();
  std::vector<int> entriesPerChannel = c->getEntriesPerChannel();

  // for the CCDB entry
  std::map<std::string, std::string> md;
  TimeSlewing ts = mCalibTOFapi->getSlewParamObj(); // we take the current CCDB object, since we want to simply update the offset
  ts.bind();

#ifdef WITH_OPENMP
  if (mNThreads < 1) {
    mNThreads = std::min(omp_get_max_threads(), NMAXTHREADS);
  }
  LOG(debug) << "Number of threads that will be used = " << mNThreads;
#pragma omp parallel for schedule(dynamic) num_threads(mNThreads)
#else
  mNThreads = 1;
#endif
  for (int sector = 0; sector < Geo::NSECTORS; sector++) {
    TMatrixD mat(3, 3);
    int ithread = 0;
#ifdef WITH_OPENMP
    ithread = omp_get_thread_num();
#endif
    LOG(info) << "Processing sector " << sector << " with thread " << ithread;
    auto& histo = c->getHisto(sector);

    std::array<double, 3> fitValues;
    std::vector<float> histoValues;
    for (int chinsector = 0; chinsector < Geo::NPADSXSECTOR; chinsector++) {
      // make the slice of the 2D histogram so that we have the 1D of the current channel
      int ich = chinsector + sector * Geo::NPADSXSECTOR;
      auto entriesInChannel = entriesPerChannel.at(ich);
      if (entriesInChannel == 0) {
        ts.setChannelOffset(ich, 0.0);
        continue; // skip always since a channel with 0 entries is normal, it will be flagged as problematic
      }

      if (entriesInChannel < mMinEntries) {
        LOG(debug) << "channel " << ich << " will not be calibrated since it has only " << entriesInChannel << " entries (min = " << mMinEntries << ")";
        ts.setChannelOffset(ich, 0.0);
        continue;
      }

      LOG(debug) << "channel " << ich << " will be calibrated since it has " << entriesInChannel << " entries (min = " << mMinEntries << ")";
      fitValues.fill(-99999999);
      histoValues.clear();
      // more efficient way
      int imax = nbins / 2;
      double maxval = 0;
      double binwidth = 2 * range / nbins;
      int binrange = int(1500 / binwidth) + 1;
      float minRange = -range;
      float maxRange = range;
      int nbinsUsed = 0;
      for (unsigned j = chinsector; j <= chinsector; ++j) {
        for (unsigned i = 0; i < nbins; ++i) { // find peak
          const auto& v = histo.at(i, j);
          if (v > maxval) {
            maxval = v;
            imax = i;
          }
        }

        float renorm = 1.; // to avoid fit problem when stats is too large (bad chi2)
        if (maxval > 10) {
          renorm = 10. / maxval;
        }

        for (unsigned i = 0; i < nbins; ++i) {
          const auto& v = histo.at(i, j);
          LOG(debug) << "channel = " << ich << ", in sector = " << sector << " (where it is channel = " << chinsector << ") bin = " << i << " value = " << v;
          if (i >= imax - binrange && i < imax + binrange) {
            histoValues.push_back(v * renorm);
            nbinsUsed++;
          } // not count for entries far from the peak (fit optimization)
        }
      }

      minRange = (imax - nbins / 2 - binrange) * binwidth;
      maxRange = (imax - nbins / 2 + binrange) * binwidth;

      double fitres = fitGaus(nbinsUsed, histoValues.data(), minRange, maxRange, fitValues, nullptr, 2., false);
      LOG(info) << "channel = " << ich << " fitted by thread = " << ithread;
      if (fitres > -3) {
        LOG(info) << "Channel " << ich << " :: Fit result " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
      } else {
#ifdef DEBUGGING
        FILE* f = fopen(Form("%d.cal", ich), "w");
        for (int i = 0; i < histoValues.size(); i++) {
          fprintf(f, "%d %f %f\n", i, minRange + binwidth * i, histoValues[i]);
        }
        fclose(f);
#endif
        LOG(info) << "Channel " << ich << " :: Fit failed with result = " << fitres;
        ts.setFractionUnderPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, -1);
        ts.setSigmaPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, 99999);
        ts.setChannelOffset(ich, 0.0);
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

      int tobeused = o2::tof::Utils::getMaxUsedChannel(ich);
      fitValues[1] += tobeused * o2::tof::Geo::BC_TIME_INPS; // adjust by adding the right BC

      if (abs(fitValues[1]) > mRange) {
        ts.setFractionUnderPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, -1);
        ts.setSigmaPeak(ich / Geo::NPADSXSECTOR, ich % Geo::NPADSXSECTOR, 99999);
      }

      bool isProb = ts.getFractionUnderPeak(ich) < 0.5 || ts.getSigmaPeak(ich) > 1000;

      if (!isProb) {
        ts.updateOffsetInfo(ich, fitValues[1]);
      } else {
        ts.setChannelOffset(ich, 0.0);
      }

#ifdef DEBUGGING
      mFitCal->Fill(ich, fitValues[1]);
#endif
      LOG(debug) << "udpdate channel " << ich << " with " << fitValues[1] << " offset in ps";
    } // end loop channels in sector
  }   // end loop over sectors
  auto clName = o2::utils::MemFileHelper::getClassName(ts);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  auto startValidity = slot.getStaticStartTimeMS();
  auto endValidity = startValidity + o2::ccdb::CcdbObjectInfo::MONTH * 2;
  ts.setStartValidity(startValidity);
  ts.setEndValidity(endValidity);
  mInfoVector.emplace_back("TOF/Calib/ChannelCalib", clName, flName, md, startValidity, endValidity);
  mTimeSlewingVector.emplace_back(ts);

#ifdef DEBUGGING
  TFile fout("debug_tof_cal.root", "RECREATE");
  mFitCal->Write();
  mChannelDist->Write();
  fout.Close();
#endif

  Utils::printFillScheme();
}

//_____________________________________________

template class TOFChannelCalibrator<o2::dataformats::CalibInfoTOF>;
template class TOFChannelCalibrator<o2::tof::CalibInfoCluster>;

} // end namespace tof
} // end namespace o2
