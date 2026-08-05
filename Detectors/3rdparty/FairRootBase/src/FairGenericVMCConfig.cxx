/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
// -------------------------------------------------------------------------
// -----               FairGenericVMCConfigurce file                   -----
// -----            Created 2019.02.19  by R. Karabowicz               -----
// -------------------------------------------------------------------------
#include "FairGenericVMCConfig.h"

#include "FairLogger.h"
#include "FairRunSim.h"

#include <TROOT.h>

FairGenericVMCConfig::FairGenericVMCConfig() {}

FairGenericVMCConfig::~FairGenericVMCConfig() {}

void FairGenericVMCConfig::Setup(const char* mcEngine)
{
    LOG(warning) << "FairGenericVMCConfig::Setup() Using gConfig.C macro DEPRACATED.";
    LOG(warning) << "Check FairRoot/examples/common/gconfig/ for current YAML implementation.";

    TString lUserCuts = FairRunSim::Instance()->GetUserCuts();
    TString lUserConfig = FairRunSim::Instance()->GetUserConfig();

    TString work = getenv("VMCWORKDIR");
    TString work_config = work + "/gconfig/";
    work_config.ReplaceAll("//", "/");

    TString Lib_config = getenv("GEANT4VMC_MACRO_DIR");
    Lib_config.ReplaceAll("//", "/");
    if (!Lib_config.EndsWith("/") && !Lib_config.IsNull()) {
        Lib_config += "/";
    }

    TString config_dir = getenv("CONFIG_DIR");
    config_dir.ReplaceAll("//", "/");

    Bool_t AbsPath = kFALSE;

    TString LibMacro;
    TString LibFunction;
    TString ConfigMacro;
    TString cuts = lUserCuts;
    //----------------------------------------------Geant4 Config-----------------------------------------
    if (strcmp(mcEngine, "TGeant4") == 0) {
        TString g4LibMacro = "g4libs.C";
        TString g4Macro;
        if (lUserConfig.IsNull()) {
            g4Macro = "g4Config.C";
            lUserConfig = g4Macro;
        } else {
            if (lUserConfig.Contains("/")) {
                AbsPath = kTRUE;
            }
            g4Macro = lUserConfig;
            LOG(info) << "---------------User config is used: " << g4Macro.Data();
        }
        if (TString(gSystem->FindFile(config_dir.Data(), g4LibMacro))
            != TString("")) {   // be carfull after this call the string g4LibMacro is empty if not found!!!!
            LOG(info) << "---User path for Configuration (g4libs.C) is used: " << config_dir.Data();
            LibMacro = g4LibMacro;
        } else if (gSystem->AccessPathName((Lib_config + "g4libs.C").Data()) == false) {
            // Note: file is existing if AccessPathName return false
            LOG(info) << "---G4VMC macro  path for Configuration (g4libs.C) is used: " << Lib_config.Data();
            LibMacro = Lib_config + "g4libs.C";
        }
        LibFunction = "g4libs()";
        if (!AbsPath && TString(gSystem->FindFile(config_dir.Data(), g4Macro)) != TString("")) {
            LOG(info) << "---User path for Configuration (g4Config.C) is used: " << config_dir.Data();
            ConfigMacro = g4Macro;
        } else {
            if (AbsPath) {
                ConfigMacro = lUserConfig;
            } else {
                ConfigMacro = work_config + lUserConfig;
            }
        }
        //----------------------------------------------Geant3 Config-----------------------------------------
    } else if (strcmp(mcEngine, "TGeant3") == 0) {
        TString g3LibMacro = "g3libs.C";
        TString g3Macro = "g3Config.C";
        if (lUserConfig.IsNull()) {
            g3Macro = "g3Config.C";
            lUserConfig = g3Macro;
            LOG(info) << "-------------- Standard Config is called ------------------------------------";
        } else {
            if (lUserConfig.Contains("/")) {
                AbsPath = kTRUE;
            }
            g3Macro = lUserConfig;
            LOG(info) << "---------------User config is used: " << g3Macro.Data();
        }
        if (TString(gSystem->FindFile(config_dir.Data(), g3LibMacro)) != TString("")) {
            LOG(info) << "---User path for Configuration (g3libs.C) is used: " << config_dir.Data();
            LibMacro = g3LibMacro;
        } else if (gSystem->AccessPathName((work_config + "g3libs.C").Data()) == false) {
            // Note: file is existing if AccessPathName return false
            LOG(info) << "---VMCWORKDIR path for Configuration (g3libs.C) is used: " << work_config.Data();
            LibMacro = work_config + "g3libs.C";
        }
        LibFunction = "g3libs()";
        if (!AbsPath && TString(gSystem->FindFile(config_dir.Data(), g3Macro)) != TString("")) {
            LOG(info) << "---User path for Configuration (g3Config.C) is used: " << config_dir.Data();
            ConfigMacro = g3Macro;
        } else {
            if (AbsPath) {
                ConfigMacro = lUserConfig;
            } else {
                ConfigMacro = work_config + lUserConfig;
            }
        }
        //----------------------------------------------Fluka Config-----------------------------------------
    } else if (strcmp(mcEngine, "TFluka") == 0) {
        TString flLibMacro = "fllibs.C";
        TString flMacro = "flConfig.C";
        if (lUserConfig.IsNull()) {
            flMacro = "flConfig.C";
            lUserConfig = flMacro;
        } else {
            if (lUserConfig.Contains("/")) {
                AbsPath = kTRUE;
            }
            flMacro = lUserConfig;
            LOG(info) << "---------------User config is used: " << flMacro.Data();
        }
        if (TString(gSystem->FindFile(config_dir.Data(), flLibMacro)) != TString("")) {
            LOG(info) << "---User path for Configuration (fllibs.C) is used: " << config_dir.Data();
        } else {
            flLibMacro = work_config + "fllibs.C";
        }
        LibMacro = flLibMacro;
        LibFunction = "fllibs()";
        if (!AbsPath && TString(gSystem->FindFile(config_dir.Data(), flMacro)) != TString("")) {
            LOG(info) << "---User path for Configuration (flConfig.C) is used: " << config_dir.Data();
            ConfigMacro = flMacro;
        } else {
            if (AbsPath) {
                ConfigMacro = lUserConfig;
            } else {
                ConfigMacro = work_config + lUserConfig;
            }
        }
    }
    //----------------------------------------------SetCuts------------------------------------------------
    if (TString(gSystem->FindFile(config_dir.Data(), cuts)) != TString("")) {
        LOG(info) << "---User path for Cuts and Processes (SetCuts.C) is used: " << config_dir.Data();
    } else {
        cuts = work_config + lUserCuts;
    }
    //--------------------------------------Now load the Config and Cuts------------------------------------
    if (!LibMacro.IsNull()) {
        gROOT->LoadMacro(LibMacro.Data());
        gROOT->ProcessLine(LibFunction.Data());
    }

    gROOT->LoadMacro(ConfigMacro.Data());
    gROOT->ProcessLine("Config()");

    gROOT->LoadMacro(cuts);
    gROOT->ProcessLine("SetCuts()");
}

ClassImp(FairGenericVMCConfig);
