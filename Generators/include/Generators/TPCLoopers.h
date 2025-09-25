// Copyright 2024-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author M+Giacalone - September 2025

#ifndef ALICEO2_EVENTGEN_TPCLOOPERS_H_
#define ALICEO2_EVENTGEN_TPCLOOPERS_H_

#ifdef GENERATORS_WITH_TPCLOOPERS
#include <onnxruntime_cxx_api.h>
#endif
#include <vector>
#include <rapidjson/document.h>
#include "TRandom3.h"
#include <SimulationDataFormat/DigitizationContext.h>
#include "TParticle.h"

#ifdef GENERATORS_WITH_TPCLOOPERS
// Static Ort::Env instance for multiple onnx model loading
extern Ort::Env global_env;

// This class is responsible for loading the scaler parameters from a JSON file
// and applying the inverse transformation to the generated data.
// Inferenced output is scaled (min-max normalization or robust scaling for outlier features) during training,
// so we need to revert this transformation to get physical values.
struct Scaler {
  std::vector<double> normal_min;
  std::vector<double> normal_max;
  std::vector<double> outlier_center;
  std::vector<double> outlier_scale;

  void load(const std::string& filename);

  std::vector<double> inverse_transform(const std::vector<double>& input);

 private:
  std::vector<double> jsonArrayToVector(const rapidjson::Value& jsonArray);
};

// This class loads the ONNX model and generates samples using it.
class ONNXGenerator
{
 public:
  ONNXGenerator(Ort::Env& shared_env, const std::string& model_path);

  std::vector<double> generate_sample();

 private:
  Ort::Env& env;
  Ort::Session session;
  TRandom3 rand_gen;
};
#endif // GENERATORS_WITH_TPCLOOPERS

namespace o2
{
namespace eventgen
{

#ifdef GENERATORS_WITH_TPCLOOPERS
/**
 * Generator for TPC Loopers based on pre-trained ONNX models.
 * Currently it generates loopers as electron-positron pairs and Compton electrons
 * according to specified distributions and parameters.
 * This can be extended to other types of background processes in the future (e.g. slow neutron spallation products, saturation tail).
 * Multiple configuration options are available:
 * - Flat gas: loopers are generated uniformly per event taking a reference value which can be either the LHC orbit time or the average interaction time record interval from the collision context.
 *   ==> Current automatic setup (default) sets the interaction rate automatically from the collision context and the reference value per orbit is calculated from an external file.
 *   ==> Number of loopers per orbit can be adjusted via a specific parameter.
 * - Poisson + Gaussian sampling: number of loopers are sampled from Poissonian (for pairs) and Gaussian (for Compton electrons) distributions based on provided parameters.
 *   ==> flat gas must be disabled to use this option.
 * - Fixed number of loopers per event
 *   ==> flat gas must be disabled to use this option and Poissonian/Gaussian parameters file should be set to None
 */
class GenTPCLoopers
{
 public:
  GenTPCLoopers(std::string model_pairs = "tpcloopmodel.onnx", std::string model_compton = "tpcloopmodelcompton.onnx",
                std::string poisson = "poisson.csv", std::string gauss = "gauss.csv", std::string scaler_pair = "scaler_pair.json",
                std::string scaler_compton = "scaler_compton.json");

  Bool_t generateEvent();

  Bool_t generateEvent(double time_limit);

  std::vector<TParticle> importParticles();

  unsigned int PoissonPairs();

  unsigned int GaussianElectrons();

  void SetNLoopers(unsigned int nsig_pair, unsigned int nsig_compton);

  void SetMultiplier(const std::array<float, 2>& mult);

  void setFlatGas(Bool_t flat, Int_t number = -1, Int_t nloopers_orbit = -1);

  void setFractionPairs(float fractionPairs);

  void SetRate(const std::string& rateFile, bool isPbPb, int intRate = 50000);

  void SetAdjust(float adjust = 0.f);

  unsigned int getNLoopers() const { return (mNLoopersPairs + mNLoopersCompton); }

 private:
  std::unique_ptr<ONNXGenerator> mONNX_pair = nullptr;
  std::unique_ptr<ONNXGenerator> mONNX_compton = nullptr;
  std::unique_ptr<Scaler> mScaler_pair = nullptr;
  std::unique_ptr<Scaler> mScaler_compton = nullptr;
  double mPoisson[3] = {0.0, 0.0, 0.0};    // Mu, Min and Max of Poissonian
  double mGauss[4] = {0.0, 0.0, 0.0, 0.0}; // Mean, Std, Min, Max
  std::vector<std::vector<double>> mGenPairs;
  std::vector<std::vector<double>> mGenElectrons;
  unsigned int mNLoopersPairs = -1;
  unsigned int mNLoopersCompton = -1;
  std::array<float, 2> mMultiplier = {1., 1.};
  bool mPoissonSet = false;
  bool mGaussSet = false;
  // Random number generator
  TRandom3 mRandGen;
  int mCurrentEvent = 0;                                          // Current event number, used for adaptive loopers
  TFile* mContextFile = nullptr;                                  // Input collision context file
  o2::steer::DigitizationContext* mCollisionContext = nullptr;    // Pointer to the digitization context
  std::vector<o2::InteractionTimeRecord> mInteractionTimeRecords; // Interaction time records from collision context
  Bool_t mFlatGas = false;                                        // Flag to indicate if flat gas loopers are used
  Bool_t mFlatGasOrbit = false;                                   // Flag to indicate if flat gas loopers are per orbit
  Int_t mFlatGasNumber = -1;                                      // Number of flat gas loopers per event
  double mIntTimeRecMean = 1.0;                                   // Average interaction time record used for the reference
  double mTimeLimit = 0.0;                                        // Time limit for the current event
  double mTimeEnd = 0.0;                                          // Time limit for the last event
  float mLoopsFractionPairs = 0.08;                               // Fraction of loopers from Pairs
  int mInteractionRate = 50000;                                   // Interaction rate in Hz
};
#endif // GENERATORS_WITH_TPCLOOPERS

} // namespace eventgen
} // namespace o2

#endif // ALICEO2_EVENTGEN_TPCLOOPERS_H_