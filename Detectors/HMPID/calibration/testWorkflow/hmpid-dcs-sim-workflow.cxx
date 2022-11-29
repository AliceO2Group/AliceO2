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

// // we need to add workflow options before including Framework/runDataProcessing
// void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
// {
// // option allowing to set parameters
// }

// ------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <fmt/format.h>
#include "Framework/ConfigParamSpec.h"
#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"max-chambers", VariantType::Int, 0, {"max chamber number to use DCS variables, 0-6"}},
  };

  std::swap(workflowOptions, options);
}

void createHints(const char* type, int i, double mean, double sd, std::vector<o2::dcs::test::HintType>& dphints)
{
  double lLimit = mean - 2 * sd;
  double uLimit = mean + 2 * sd;
  if (i > 9) {
    const char* specifier = Form("HMP_TRANPLANT_MEASURE_%i_%s", i, type);
    dphints.emplace_back(o2::dcs::test::DataPointHint<double>{specifier, lLimit, uLimit});
  } else {
    const char* specifier = Form("HMP_TRANPLANT_MEASURE_0%i_%s", i, type);
    dphints.emplace_back(o2::dcs::test::DataPointHint<double>{specifier, lLimit, uLimit});
  }
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  std::vector<o2::dcs::test::HintType> dphints;

  // ==| Environment Pressure  (mBar) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_ENV_PENV", 1003., 1023.});

  // ==|(CH4) Chamber Pressures  (mBar) |=================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_MP_[0..6]_GAS_PMWPC", 3., 5.});

  //==| Temperature C6F14 IN/OUT / RADIATORS  (C) |=================================

  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_MP_[0..6]_LIQ_LOOP_RAD_[0..2]_IN_TEMP", 21.5, 22.5});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_MP_[0..6]_LIQ_LOOP_RAD_[0..2]_OUT_TEMP", 24.5, 25.5});

  // ===| HV / SECTORS (V) |=========================================================
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"HMP_MP_[0..6]_SEC_[0..5]_HV_VMON", 2049.5, 2050.5});

  // string for DPs of Refractive Index Parameters =============================================================

  // measured values for IR-parameters
  // 30 entries of 8 values:
  // Argon Ref mean    Argon Ref std       Argon Cell mean    Argon Cell std      C6F14 Ref mean     C6F14 Ref std       C6F14 Cell mean     C6F14 Cell std
  const double irVals[30][8] =
    {{-0.86698055267334, 0.015992261469364, -2.9767239093781, 0.055669896304607, -0.86270183324814, 0.015952149406075, -0.059273429214954, 0.0011367546394467}, // FIRST
     {-0.34526389837265, 0.0061066164635122, -1.599116563797, 0.029814273118973, -0.34129247069359, 0.0062515325844288, -0.11364602297544, 0.0020333647262305},
     {-0.19329330325127, 0.0031397438142449, -0.85721117258072, 0.015860633924603, -0.1903311163187, 0.0034912948030978, -0.20784763991833, 0.0037625974509865},
     {-0.14834992587566, 0.002474470064044, -0.65886175632477, 0.012137032113969, -0.14846935868263, 0.0025894897989929, -0.33942249417305, 0.0061758663505316},
     {-0.14286313951015, 0.0027808723971248, -0.63008558750153, 0.011596300639212, -0.14201314747334, 0.0024778251536191, -0.48442330956459, 0.0088781854137778},
     {-0.15407188236713, 0.0029800452757627, -0.68659967184067, 0.012656413950026, -0.15283487737179, 0.0029571398627013, -0.6342813372612, 0.011700733564794},
     {-0.17301347851753, 0.0031401996966451, -0.77723491191864, 0.014357348904014, -0.17033703625202, 0.0031950008124113, -0.77303212881088, 0.014290297403932},
     {-0.19241452217102, 0.0036375673953444, -0.89939516782761, 0.016652518883348, -0.19368402659893, 0.0038244123570621, -0.9212738275528, 0.017078565433621},
     {-0.22112882137299, 0.0039031782653183, -1.0648462772369, 0.01976865530014, -0.22205835580826, 0.0039840959943831, -1.1093875169754, 0.02061739563942},
     {-0.2481614202261, 0.0044772434048355, -1.2268204689026, 0.022809866815805, -0.24697144329548, 0.0043729245662689, -1.2903002500534, 0.024012215435505}, // 10
     {-0.26668339967728, 0.0048548625782132, -1.3464559316635, 0.025057515129447, -0.26911398768425, 0.0049738502129912, -1.4212131500244, 0.026473663747311},
     {-0.28008028864861, 0.005099821370095, -1.431206703186, 0.026657309383154, -0.28283646702766, 0.0050928434357047, -1.5145578384399, 0.028227487578988},
     {-0.29250073432922, 0.0052133435383439, -1.4960530996323, 0.027876731008291, -0.29394751787186, 0.0053529520519078, -1.587148308754, 0.029599368572235},
     {-0.30391928553581, 0.0056737358681858, -1.5516122579575, 0.028916034847498, -0.30461722612381, 0.0054301782511175, -1.648961186409, 0.030760537832975},
     {-0.31080573797226, 0.0055016754195094, -1.5923854112625, 0.029680499807, -0.31138452887535, 0.0053659677505493, -1.6954737901688, 0.031633801758289}, // 15
     {-0.31008964776993, 0.0056457673199475, -1.6111241579056, 0.030030036345124, -0.31404292583466, 0.0057512698695064, -1.7176169157028, 0.032046440988779},
     {-0.30684891343117, 0.0053859171457589, -1.6295021772385, 0.030381938442588, -0.31102648377419, 0.0058898706920445, -1.7385073900223, 0.03244848921895},
     {-0.3116267323494, 0.0054278862662613, -1.6659815311432, 0.031067481264472, -0.31384035944939, 0.0056781014427543, -1.779548406601, 0.033212583512068},
     {-0.31930908560753, 0.0057652872055769, -1.7229619026184, 0.032144580036402, -0.32055324316025, 0.0059450049884617, -1.8417125940323, 0.034386739134789},
     {-0.33134245872498, 0.0058694803155959, -1.7995973825455, 0.033577345311642, -0.33342209458351, 0.0060769249685109, -1.9229571819305, 0.035908468067646}, // 20
     {-0.3437374830246, 0.0064436765387654, -1.8924849033356, 0.035330656915903, -0.34587496519089, 0.0062040886841714, -2.0227761268616, 0.03778387606144},
     {-0.36247292160988, 0.0063365683890879, -1.9954553842545, 0.037263486534357, -0.36232820153236, 0.0065949787385762, -2.1335067749023, 0.039873410016298},
     {-0.37738528847694, 0.0068919118493795, -2.1079571247101, 0.039387684315443, -0.37793877720833, 0.0066521903499961, -2.2520573139191, 0.042100977152586},
     {-0.39373001456261, 0.0070624663494527, -2.2194654941559, 0.041477311402559, -0.39541909098625, 0.0072884988039732, -2.3708393573761, 0.044334270060062},
     {-0.41024819016457, 0.0073348702862859, -2.3269090652466, 0.043495532125235, -0.41210383176804, 0.0075397142209113, -2.4846012592316, 0.046468034386635}, // 25
     {-0.42143520712852, 0.0077510792762041, -2.4250814914703, 0.045346949249506, -0.42174956202507, 0.0076608234085143, -2.588326215744, 0.048415776342154},
     {-0.43363136053085, 0.0077972891740501, -2.5094130039215, 0.046931173652411, -0.43422821164131, 0.0079285996034741, -2.6780755519867, 0.05010611563921},
     {-0.44121220707893, 0.0078951977193356, -2.5789725780487, 0.048236511647701, -0.44251444935799, 0.0080582965165377, -2.7519564628601, 0.05148757994175},
     {-0.44413486123085, 0.0082251932471991, -2.6322927474976, 0.049238469451666, -0.44755265116692, 0.0082092136144638, -2.8076829910278, 0.052542366087437},
     {-0.44769379496574, 0.0082640117034316, -2.6672098636627, 0.049895957112312, -0.45071032643318, 0.0082084992900491, -2.8445172309875, 0.053234227001667}}; // 30

  double argonRefMean, argonRefSD;   // Argon Ref current (mA)
  double argonCellMean, argonCellSD; // Argon Cell current (mA)
  double freonRefMean, freonRefSD;   // C6F14 Ref current (mA)
  double freonCellMean, freonCellSD; // C6F14 Cell current (mA)
  int iR = 162;

  for (int i = 0; i < 30; ++i) {

    createHints("WAVELENGHT", i, iR, 0.025, dphints);
    iR += 2;

    argonRefMean = irVals[i][0];
    argonRefSD = irVals[i][1];
    createHints("ARGONREFERENCE", i, argonRefMean, argonRefSD, dphints);

    argonCellMean = irVals[i][2];
    argonCellSD = irVals[i][3];
    createHints("ARGONCELL", i, argonCellMean, argonCellSD, dphints);

    freonRefMean = irVals[i][4];
    freonRefSD = irVals[i][5];
    createHints("C6F14REFERENCE", i, freonRefMean, freonRefSD, dphints);

    freonCellMean = irVals[i][6];
    freonCellSD = irVals[i][7];
    createHints("C6F14CELL", i, freonCellMean, freonCellSD, dphints);
  }

  o2::framework::WorkflowSpec specs;

  specs.emplace_back(o2::dcs::test::getDCSRandomDataGeneratorSpec(dphints, "HMP"));
  return specs;
}
