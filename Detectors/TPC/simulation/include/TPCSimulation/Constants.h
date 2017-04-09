/// \file Constants.h
/// \brief Definition of constants that should go to a (yet to be defined) parameter space
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
///
#ifndef AliceO2_TPC_Constants_H_
#define AliceO2_TPC_Constants_H_

#include <vector>
#include <array>

namespace o2 {
namespace TPC {

// gas parameters
constexpr float WION = 37.3e-9;           ///< Effective ionization potential of Ne-CO2-N2 (90-10-5), value from TDR
constexpr float IPOT = 20.77e-9;          ///< Effective ionization potential of Ne-CO2 (90-10), value from ORIGINAL TDR
constexpr float SCALEWIONG4 = 0.85;       ///< scale factor to tune WION for GEANT4 
constexpr float FANOFACTORG4 = 0.7;       ///< parameter for smearing the number of ionizations (nel) using GEANT4
constexpr float ATTCOEF = 250.;           ///< Attachement coefficient of Ne-CO2-N2 (90-10-5), value from TDR
constexpr float OXYCONT = 5.e-6;          ///< Oxygen content, value from current AliRoot TPCParam
constexpr float DRIFTV = 2.58;            ///< Drift velocity of Ne-CO2-N2 (90-10-5), value from TDR
constexpr float SIGMAOVERMU = 0.78;       ///< Sigma over mu of Ne-CO2-N2 (90-10-5), gives deviation from exponential gain fluctuations. Value from JINST 11 (2016) P10017
constexpr float DIFFT = 0.0209;           ///< Transverse diffusion of Ne-CO2-N2 (90-10-5), value from TDR
constexpr float DIFFL = 0.0221;           ///< Longitudinal diffusion of Ne-CO2-N2 (90-10-5), value from TDR
constexpr float NPRIM = 14.;              ///< Number of priimary electrons per MIP and cm in Ne-CO2-N2 (90-10-5), value from current AliRoot TPCParam
const std::vector<float> BBPARAM{0.76176e-1, 10.632, 0.13279e-4, 1.8631, 1.9479};   ///< Bethe-Bloch parameters of Ne-CO2-N2 (90-10-5), value from current AliRoot TPCParam

// ROC parameters
constexpr float EFFGAINGEM1 = 9.1;        ///< Effective gain in GEM1, value from TDR addendum
constexpr float EFFGAINGEM2 = 0.88;       ///< Effective gain in GEM2, value from TDR addendum
constexpr float EFFGAINGEM3 = 1.66;       ///< Effective gain in GEM3, value from TDR addendum
constexpr float EFFGAINGEM4 = 144;        ///< Effective gain in GEM4, value from TDR addendum
constexpr std::array<float, 4> GAIN{{EFFGAINGEM1, EFFGAINGEM2, EFFGAINGEM3, EFFGAINGEM4}};
constexpr std::array<float, 4> MULTIPLICATION{{14.f, 8.f, 53.f, 240.f}};
constexpr std::array<float, 4> COLLECTION{{1.f, 0.2, 0.25, 1.f}};
constexpr std::array<float, 4> EXTRACTION{{0.65, 0.55, 0.12, 0.6}};

constexpr float CPAD = 0.1;               ///< Capacitance of a single pad in pF
constexpr float TPCLENGTH = 250.;         ///< Maximal drift length in the TPC

//electronics parameters
constexpr int mNShapedPoints = 8;         ///< Number of points taken into account for shaping (8 chosen to fit into SSE registers)
constexpr float ADCSAT = 1024;            ///< ADC saturation
constexpr float QEL = 1.602e-19;          ///< Electron charge
constexpr float CHIPGAIN = 20;            ///< mV/fC - should be a switch as it may be either 20 or 30, depending on the configuration
constexpr float ADCDYNRANGE = 2200;       ///< Dynamic range of the ADC in mV
constexpr float PEAKINGTIME = 160e-3;     ///< Peaking time of the SAMPA, in us

constexpr float ZBINWIDTH = 0.19379844961; ///< Width of a z bin in us

}
}

#endif // AliceO2_TPC_Constants_H_
