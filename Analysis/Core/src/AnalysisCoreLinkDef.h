// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class StepTHn + ;
#pragma link C++ class StepTHnT < TArrayF> + ;
#pragma link C++ class StepTHnT < TArrayD> + ;
#pragma link C++ typedef StepTHnF;
#pragma link C++ typedef StepTHnD;

#pragma link C++ class CorrelationContainer + ;
#pragma link C++ class TrackSelection + ;
#pragma link C++ class TriggerAliases + ;

#pragma link C++ class VarManager + ;
#pragma link C++ class HistogramManager + ;
#pragma link C++ class AnalysisCut + ;
#pragma link C++ class AnalysisCompositeCut + ;

// #pragma link C++ class JetFinder+;
