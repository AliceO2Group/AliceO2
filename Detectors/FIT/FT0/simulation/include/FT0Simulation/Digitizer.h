// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FT0_DIGITIZER_H
#define ALICEO2_FT0_DIGITIZER_H

#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsFT0/Digit.h"
#include "DataFormatsFT0/ChannelData.h"
#include "DataFormatsFT0/MCLabel.h"
#include "FT0Simulation/Detector.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "FT0Simulation/DigitizationParameters.h"
#include <TH1F.h>
#include <bitset>
#include <vector>
#include <deque>
#include <optional>
#include <set>

namespace o2
{
namespace ft0
{
class Digitizer
{
 public:
  Digitizer(const DigitizationParameters& params, Int_t mode = 0) : mMode(mode), parameters(params) { initParameters(); };
  ~Digitizer() = default;

  void process(const std::vector<o2::ft0::HitType>* hits, std::vector<o2::ft0::Digit>& digitsBC,
               std::vector<o2::ft0::ChannelData>& digitsCh,
               o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void flush(std::vector<o2::ft0::Digit>& digitsBC,
             std::vector<o2::ft0::ChannelData>& digitsCh,
             o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void flush_all(std::vector<o2::ft0::Digit>& digitsBC,
                 std::vector<o2::ft0::ChannelData>& digitsCh,
                 o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& label);
  void initParameters();
  void printParameters();
  void setEventID(Int_t id) { mEventID = id; }
  void setSrcID(Int_t id) { mSrcID = id; }
  const o2::InteractionRecord& getInteractionRecord() const { return mIntRecord; }
  o2::InteractionRecord& getInteractionRecord(o2::InteractionRecord& src) { return mIntRecord; }
  void setInteractionRecord(const o2::InteractionTimeRecord& src) { mIntRecord = src; }
  uint32_t getOrbit() const { return mIntRecord.orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  double measure_amplitude(const std::vector<double>& times);
  void init();
  void finish();

  std::optional<double> get_time(const std::vector<double>& times);

  void setContinuous(bool v = true) { mIsContinuous = v; }
  bool isContinuous() const { return mIsContinuous; }
  struct BCCache {
    struct particle {
      int hit_ch;
      double hit_time;
      friend bool operator<(particle a, particle b)
      {
        return (a.hit_ch != b.hit_ch) ? (a.hit_ch < b.hit_ch) : (a.hit_time < b.hit_time);
      }
    };
    // using particle = std::pair<int, double>;
    std::vector<particle> hits;
    std::set<ft0::MCLabel> labels;
  };

 private:
  // digit info
  // parameters
  Int_t mMode;                          //triggered or continuos
  o2::InteractionTimeRecord mIntRecord; // Interaction record (orbit, bc)
  Int_t mEventID;
  Int_t mSrcID;              // signal, background or QED
  bool mIsContinuous = true; // continuous (self-triggered) or externally-triggered readout

  o2::InteractionRecord firstBCinDeque = 0;
  std::deque<BCCache> mCache;

  DigitizationParameters parameters;

  void storeBC(BCCache& bc,
               std::vector<o2::ft0::Digit>& digitsBC,
               std::vector<o2::ft0::ChannelData>& digitsCh,
               o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>& labels);

  ClassDefNV(Digitizer, 1);
};
inline double sinc(const double x)
{
  return (std::abs(x) < 1e-12) ? 1 : std::sin(x) / x;
}

template <typename Float>
Float signalForm_i(Float x)
{
  using namespace std;
  return x > 0 ? -(exp(-0.83344945 * x) - exp(-0.45458 * x)) / 7.8446501 : 0.;
  //return -(exp(-0.83344945 * x) - exp(-0.45458 * x)) * (x >= 0) / 7.8446501; // Maximum should be 7.0/250 mV
};

inline float signalForm_integral(float x)
{
  using namespace std;
  double a = -0.45458, b = -0.83344945;
  if (x < 0)
    x = 0;
  return -(exp(b * x) / b - exp(a * x) / a) / 7.8446501;
};
} // namespace ft0
} // namespace o2

#endif
