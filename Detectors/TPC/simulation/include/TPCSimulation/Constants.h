/// \file Constants.h
/// \brief Definition of constants that should go to a (yet to be defined) parameter space
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef AliceO2_TPC_Constants_H_
#define AliceO2_TPC_Constants_H_

namespace AliceO2 {
namespace TPC {

// gas parameters
constexpr Float_t WION = 37.3e-6;           ///< Effective ionization potential of Ne-CO2-N2 (90-10-5), value from TDR
constexpr Float_t ATTCOEF = 250.;           ///< Attachement coefficient of Ne-CO2-N2 (90-10-5), value from TDR
constexpr Float_t OXYCONT = 5.e-6;          ///< Oxygen content, value from current AliRoot TPCParam
constexpr Float_t DRIFTV = 2.58;            ///< Drift velocity of Ne-CO2-N2 (90-10-5), value from TDR
constexpr Float_t SIGMAOVERMU = 0.78;       ///< Sigma over mu of Ne-CO2-N2 (90-10-5), gives deviation from exponential gain fluctuations. Value from JINST 11 (2016) P10017
constexpr Float_t DIFFT = 0.0209;           ///< Transverse diffusion of Ne-CO2-N2 (90-10-5), value from TDR
constexpr Float_t DIFFL = 0.0221;           ///< Longitudinal diffusion of Ne-CO2-N2 (90-10-5), value from TDR
constexpr Float_t NPRIM = 14.;              ///< Number of priimary electrons per MIP and cm in Ne-CO2-N2 (90-10-5), value from current AliRoot TPCParam
const std::vector<float> BBPARAM{0.76176e-1, 10.632, 0.13279e-4, 1.8631, 1.9479};   ///< Bethe-Bloch parameters of Ne-CO2-N2 (90-10-5), value from current AliRoot TPCParam

// ROC parameters
constexpr Float_t EFFGAINGEM1 = 9.1;        ///< Effective gain in GEM1, value from TDR addendum
constexpr Float_t EFFGAINGEM2 = 0.88;       ///< Effective gain in GEM2, value from TDR addendum
constexpr Float_t EFFGAINGEM3 = 1.66;       ///< Effective gain in GEM3, value from TDR addendum
constexpr Float_t EFFGAINGEM4 = 144;        ///< Effective gain in GEM4, value from TDR addendum
constexpr Float_t CPAD = 0.1;               ///< Capacitance of a single pad in pF
constexpr Float_t TPCLENGTH = 250.;         ///< Maximal drift length in the TPC

//electronics parameters
constexpr Float_t ADCSAT = 1023;            ///< ADC saturation
constexpr Float_t QEL = 1.602e-19;          ///< Electron charge
constexpr Float_t CHIPGAIN = 20;            ///< 
constexpr Float_t ADCDYNRANGE = 2000;       ///< Dynamic range of the ADC
constexpr Float_t PEAKINGTIME = 160e-3;     ///< Peaking time of the SAMPA, in us

constexpr Float_t ZBINWIDTH = 0.19379844961; ///< Width of a z bin in us

}
}

#endif // AliceO2_TPC_Constants_H_
