
#ifndef FluenceWeightCalculator_h
#define FluenceWeightCalculator_h
#include <vector>
#include <string>
#include <memory>
#include "TGraph.h"
//
// Static container class for damage weight funnctions in form of TGraphs
// The weights can be read from a csv file and stored in the graphs.
//
class FluenceWeightCalculator
{
 public:
  FluenceWeightCalculator() = delete;
  static void InitWeights(const std::string& filename);
  static void InitWeightsFromCSV(const std::string& filename);
  static double GetWeight(const int pdg, const double ekin);

 private:
  static std::unique_ptr<TGraph> neutronG;
  static std::unique_ptr<TGraph> protonG;
  static std::unique_ptr<TGraph> pionG;
};
#endif
