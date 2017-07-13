// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file    EveInitializer.cxx
/// @author  Jeremi Niedziela
///

#include "EveInitializer.h"

#include "EveEventManager.h"

//#include <AliEveTrackCounter.h>
//#include <EveEventManagerEditor.h>
//#include <AliEveMultiView.h>
//#include <AliEveMacroExecutor.h>
//#include <AliEveMacro.h>
//#include <AliEveGeomGentle.h>
//#include <AliEveDataSourceOffline.h>

//#include <AliCDBManager.h>

//#include <TGrid.h>
//#include <TROOT.h>
//#include <TInterpreter.h>
//#include <TEveWindowManager.h>
//#include <TGTab.h>
//#include <TPRegexp.h>
//#include <TFolder.h>
//#include <TSystemDirectory.h>
//#include <TGFileBrowser.h>
//#include <TEveMacro.h>
//#include <TEveBrowser.h>
//#include <TRegexp.h>

//#include <cstring>
//#include <iostream>
//#include <string>
//#include <vector>

using namespace std;

EveInitializer::EveInitializer(EveEventManager::EDataSource defaultDataSource)
{
  TEnv settings;
  GetConfig(&settings);
  
  bool autoloadEvents   = settings.GetValue("events.autoload.set",false);   // set autoload by default
  bool fullscreen       = settings.GetValue("fullscreen.mode",false);       // hide left and bottom tabs
  TString ocdbStorage   = settings.GetValue("OCDB.default.path","local://$ALICE_ROOT/OCDB");// default path to OCDB
  cout<<"EveInitializer -- OCDB path:"<<ocdbStorage<<endl;
  
  
  
  shared_ptr<EveEventManager> man(EveEventManager::Instance());
  
  new EveEventManager(defaultDataSource);
  
  man->SetCdbUri(ocdbStorage);
  
  if (gSystem->Getenv("ALICE_ROOT") != 0)
  {
    gInterpreter->AddIncludePath(Form("%s/MUON", gSystem->Getenv("ALICE_ROOT")));
    gInterpreter->AddIncludePath(Form("%s/MUON/mapping", gSystem->Getenv("ALICE_ROOT")));
  }
  
  AliEveDataSourceOffline *dataSourceOffline  = (AliEveDataSourceOffline*)man->GetDataSourceOffline();
  
  ImportMacros();
  gEve->AddEvent(EveEventManager::Instance());
  
  TEveUtil::AssertMacro("VizDB_scan.C");
  
  TEveBrowser *browser = gEve->GetBrowser();
  browser->ShowCloseTab(kFALSE);
  
  //==============================================================================
  // Geometry, scenes, projections and viewers
  //==============================================================================
  
  SetupGeometry();
  AddMacros();
  
  //==============================================================================
  // Additional GUI components
  //==============================================================================
  
  TEveWindowSlot *slot = TEveWindow::CreateWindowInTab(browser->GetTabRight());
  
  // QA viewer
  /*
   slot = TEveWindow::CreateWindowInTab(browser->GetTabRight());
   slot->StartEmbedding();
   new AliQAHistViewer(gClient->GetRoot(), 600, 400, kTRUE);
   slot->StopEmbedding("QA histograms");
   */
  //    browser->GetTabRight()->SetTab(1);
  browser->StartEmbedding(TRootBrowser::kBottom);
  
  EveEventManagerWindow::GetInstance()->SetCurrentDataSource(defaultDataSource);
  browser->StopEmbedding("EventCtrl");
  
  slot = TEveWindow::CreateWindowInTab(browser->GetTabRight());
  TEveWindowTab *store_tab = slot->MakeTab();
  store_tab->SetElementNameTitle("WindowStore",
                                 "Undocked windows whose previous container is not known\n"
                                 "are placed here when the main-frame is closed.");
  gEve->GetWindowManager()->SetDefaultContainer(store_tab);
  
  
  //==============================================================================
  // AliEve objects - global tools
  //==============================================================================
  
  AliEveTrackCounter* g_trkcnt = new AliEveTrackCounter("Primary Counter");
  gEve->AddToListTree(g_trkcnt, kFALSE);
  
  
  //==============================================================================
  // Final stuff
  //==============================================================================
  
  
  // A refresh to show proper window.
  //    gEve->GetViewers()->SwitchColorSet();
  
  SetupBackground();
  
  browser->MoveResize(0, 0, gClient->GetDisplayWidth(),gClient->GetDisplayHeight() - 32);
  gEve->Redraw3D(true);
  gSystem->ProcessEvents();
  
  
  gEve->EditElement(g_trkcnt);
  gEve->Redraw3D();
  
  SetupCamera();
  
  // Fullscreen
  if(fullscreen){
    ((TGWindow*)gEve->GetBrowser()->GetTabLeft()->GetParent())->Resize(1,0);
    ((TGWindow*)gEve->GetBrowser()->GetTabBottom()->GetParent())->Resize(0,1);
    gEve->GetBrowser()->Layout();
  }
  
  gEve->FullRedraw3D();
  gSystem->ProcessEvents();
  gEve->Redraw3D(true);
  
  man->SetAutoLoad(autoloadEvents);// set autoload by default
  
  if(defaultDataSource == EveEventManager::kSourceOffline)
  {
    ((AliEveDataSourceOffline*)man->GetDataSourceOffline())->GotoEvent(0);
    if(settings.GetValue("momentum.histograms.all.events.show",false))
    {
      man->GetMomentumHistogramsDrawer()->DrawAllEvents();
    }
  }
}

void EveInitializer::SetupGeometry()
{
  AliEveMultiView *mv = AliEveMultiView::Instance();
  AliEveGeomGentle *geomGentle = new AliEveGeomGentle();
  
  // read all files with names matching "geom_list_XYZ.txt"
  vector<string> detectorsList;
  
  TEnv settings;
  GetConfig(&settings);
  string geomPath = settings.GetValue("simple.geom.path","${ALICE_ROOT}/EVE/resources/geometry/run2/");
  string alirootBasePath = gSystem->Getenv("ALICE_ROOT");
  size_t alirootPos = geomPath.find("${ALICE_ROOT}");
  
  if(alirootPos != string::npos){
    geomPath.replace(alirootPos,alirootPos+13,alirootBasePath);
  }
  
  TSystemDirectory dir(geomPath.c_str(),geomPath.c_str());
  TList *files = dir.GetListOfFiles();
  
  if (files)
  {
    TRegexp e("simple_geom_[A-Z,0-9][A-Z,0-9][A-Z,0-9].root");
    TRegexp e2("[A-Z,0-9][A-Z,0-9][A-Z,0-9]");
    
    TSystemFile *file;
    TString fname;
    TIter next(files);
    
    while ((file=(TSystemFile*)next()))
    {
      fname = file->GetName();
      if(fname.Contains(e))
      {
        TString detName = fname(e2);
        detName.Resize(3);
        detectorsList.push_back(detName.Data());
      }
    }
  }
  else{
    cout<<"\n\nEveInitializer -- geometry files not found!!!"<<endl;
    cout<<"Searched directory was:"<<endl;
    dir.Print();
  }
  
  for(int i=0;i<detectorsList.size();i++)
  {
    if(settings.GetValue(Form("%s.draw",detectorsList[i].c_str()), true))
    {
      if(detectorsList[i]=="TPC" || detectorsList[i]=="MCH" || detectorsList[i]=="MTR"|| detectorsList[i]=="MID" || detectorsList[i]=="MFT" || detectorsList[i]=="AD0" || detectorsList[i]=="FMD")
      {
        // don't load MUON+MFT and AD and standard TPC to R-Phi view
        mv->InitSimpleGeom(geomGentle->GetSimpleGeom((char*)detectorsList[i].c_str()),true,false);
      }
      else if(detectorsList[i]=="RPH")
      {
        // special TPC geom from R-Phi view
        mv->InitSimpleGeom(geomGentle->GetSimpleGeom("RPH"),false,true,false);
      }
      else
      {
        mv->InitSimpleGeom(geomGentle->GetSimpleGeom((char*)detectorsList[i].c_str()));
      }
    }
  }
}

void EveInitializer::AddMacros()
{
  //==============================================================================
  // Registration of per-event macros
  //==============================================================================
  
  
  // check which macros are available
  TEnv settings;
  GetConfig(&settings);
  vector<string> detectorsList;
  TSystemDirectory dir(Form("%s/EVE/macros/data/",gSystem->Getenv("ALICE_ROOT")),
                       Form("%s/EVE/macros/data/",gSystem->Getenv("ALICE_ROOT")));
  
  TList *files = dir.GetListOfFiles();
  
  if (files)
  {
    TRegexp e("data_vis_[A-Z,0-9][A-Z,0-9][A-Z,0-9].C");
    TRegexp e2("[A-Z,0-9][A-Z,0-9][A-Z,0-9]");
    
    TSystemFile *file;
    TString fname;
    TIter next(files);
    
    while ((file=(TSystemFile*)next()))
    {
      fname = file->GetName();
      if(fname.Contains(e))
      {
        TString detName = fname(e2);
        detName.Resize(3);
        detectorsList.push_back(detName.Data());
      }
    }
  }
  
  AliEveMacroExecutor *exec = EveEventManager::Instance()->GetExecutor();
  exec->RemoveMacros(); // remove all old macros
  
  for(int i=0;i<detectorsList.size();i++)
  {
    const char *detector = detectorsList[i].c_str();
    cout<<"Adding macros for "<<detector<<endl;
    
    // add macro for hits
    if(settings.GetValue(Form("%s.hits",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("Hits %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kHits"
                                     ));
    }
    // add macro for digits
    if(settings.GetValue(Form("%s.digits",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("Digits %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kDigits"
                                     ));
    }
    // add macro for raw data
    if(settings.GetValue(Form("%s.raw",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("Raw %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kRaw"
                                     ));
    }
    // add macro for raw data
    if(settings.GetValue(Form("%s.clusters",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("Clusters %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kClusters"
                                     ));
    }
    // add macro for ESD
    if(settings.GetValue(Form("%s.esd",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("ESD %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kESD"
                                     ));
    }
    // add macro for AOD
    if(settings.GetValue(Form("%s.aod",detector),false))
    {
      exec->AddMacro(new AliEveMacro(
                                     Form("AOD %s",detector),
                                     Form("data_vis_%s.C",detector),
                                     Form("data_vis_%s",detector),
                                     "EveEventManager::kAOD"
                                     ));
    }
  }
  // what's below should be removed
  
  bool showMuon         = settings.GetValue("MUON.show", true);              // show MUON's geom
  bool showEMCal        = settings.GetValue("EMCal.show", false);            // show EMCal and PHOS histograms
  bool drawClusters     = settings.GetValue("clusters.show",false);          // show clusters
  bool drawRawData      = settings.GetValue("rawData.show",false);           // show raw data
  bool drawHits         = settings.GetValue("hits.show",false);              // show hits
  bool drawDigits       = settings.GetValue("digits.show",false);            // show digits
  
  if(drawHits)
  {
    cout<<"Adding old macros for hits"<<endl;
    exec->AddMacro(new AliEveMacro("SIM Hits ITS", "its_hits.C",    "its_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits TPC", "tpc_hits.C",    "tpc_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits T0",  "t0_hits.C",     "t0_hits",     ""));
    exec->AddMacro(new AliEveMacro("SIM Hits FMD", "fmd_hits.C",    "fmd_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits ACORDE", "acorde_hits.C",    "acorde_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits EMCAL", "emcal_hits.C",    "emcal_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits TOF",  "tof_hits.C",     "tof_hits",     ""));
    exec->AddMacro(new AliEveMacro("SIM Hits TRD", "trd_hits.C",    "trd_hits",    ""));
    exec->AddMacro(new AliEveMacro("SIM Hits VZERO", "vzero_hits.C",    "vzero_hits",    ""));
  }
  if(drawDigits){
    cout<<"Adding old macros for digits"<<endl;
    exec->AddMacro(new AliEveMacro("DIG ITS",     "its_digits.C",  "its_digits",  ""));
    exec->AddMacro(new AliEveMacro("DIG TPC",     "tpc_digits.C",  "tpc_digits",  ""));
    exec->AddMacro(new AliEveMacro("DIG TOF",     "tof_digits.C",  "tof_digits",  ""));
    exec->AddMacro(new AliEveMacro("DIG HMPID",   "hmpid_digits.C","hmpid_digits",""));
    exec->AddMacro(new AliEveMacro("DIG FMD",     "fmd_digits.C",  "fmd_digits",  ""));
  }
  if(drawRawData)
  {
    cout<<"Adding old macros for raw data"<<endl;
    exec->AddMacro(new AliEveMacro("RAW ITS",     "its_raw.C",     "its_raw",     ""));
    exec->AddMacro(new AliEveMacro("RAW TPC",     "tpc_raw.C",     "tpc_raw",     ""));
    exec->AddMacro(new AliEveMacro("RAW TOF",     "tof_raw.C",     "tof_raw",     ""));
    exec->AddMacro(new AliEveMacro("RAW HMPID",   "hmpid_raw.C",   "hmpid_raw",   ""));
    exec->AddMacro(new AliEveMacro("RAW T0",      "t0_raw.C",      "t0_raw",      ""));
    exec->AddMacro(new AliEveMacro("RAW FMD",     "fmd_raw.C",     "fmd_raw",     ""));
    exec->AddMacro(new AliEveMacro("RAW VZERO",   "vzero_raw.C",   "vzero_raw",   ""));
    exec->AddMacro(new AliEveMacro("RAW ACORDE",  "acorde_raw.C",  "acorde_raw",  ""));
  }
  if(drawClusters)
  {
    cout<<"Adding old macros for clusters"<<endl;
    exec->AddMacro(new AliEveMacro("REC Clusters ITS", "its_clusters.C", "its_clusters",""));
    exec->AddMacro(new AliEveMacro("REC Clusters TPC", "tpc_clusters.C", "tpc_clusters",""));
    exec->AddMacro(new AliEveMacro("REC Clusters TRD", "trd_clusters.C", "trd_clusters",""));
    exec->AddMacro(new AliEveMacro("REC Clusters TOF", "tof_clusters.C", "tof_clusters",""));
    exec->AddMacro(new AliEveMacro("REC Clusters HMPID", "hmpid_clusters.C","hmpid_clusters",""));
    exec->AddMacro(new AliEveMacro("REC Clusters PHOS", "phos_clusters.C","phos_clusters",""));
  }
  if (showMuon)
  {
    cout<<"Adding old macros for muon"<<endl;
    if(drawRawData){
      exec->AddMacro(new AliEveMacro("RAW MUON", "muon_raw.C", "muon_raw", ""));
    }
    if(drawDigits){
      exec->AddMacro(new AliEveMacro("DIG MUON", "muon_digits.C", "muon_digits", ""));
    }
    if(drawClusters){
      exec->AddMacro(new AliEveMacro("REC Clusters MUON", "muon_clusters.C", "muon_clusters", ""));
    }
    
  }
  if(showEMCal){
    cout<<"Adding old macros for emcal"<<endl;
    exec->AddMacro(new AliEveMacro("ESD EMCal", "emcal_esdclustercells.C", "emcal_esdclustercells", ""));
  }
}

void EveInitializer::SetupCamera()
{
  // move and rotate sub-views
  TEnv settings;
  GetConfig(&settings);
  AliEveMultiView *mv = AliEveMultiView::Instance();
  gEve->GetBrowser()->GetTabRight()->SetTab(1);
  
  TGLViewer *glView3D   = mv->Get3DView()->GetGLViewer();
  TGLViewer *glViewRPhi = mv->GetRPhiView()->GetGLViewer();
  TGLViewer *glViewRhoZ = mv->GetRhoZView()->GetGLViewer();
  
  double angleHorizontal = settings.GetValue("camera.3D.rotation.horizontal",-0.4);// horizontal rotation of 3D view
  double angleVertical   = settings.GetValue("camera.3D.rotation.vertical",1.0);   // vertical rotation of 3D view
  double zoom3D          = settings.GetValue("camera.3D.zoom",1.0);// zoom of 3D view
  double zoomRPhi        = settings.GetValue("camera.R-Phi.zoom",1.0);// zoom of R-Phi view
  double zoomRhoZ        = settings.GetValue("camera.Rho-Z.zoom",1.0);// zoom of Rho-Z view
  
  glView3D->CurrentCamera().Reset();
  glViewRPhi->CurrentCamera().Reset();
  glViewRhoZ->CurrentCamera().Reset();
  
  glView3D->CurrentCamera().RotateRad(angleHorizontal, angleVertical);
  glView3D->CurrentCamera().Dolly(zoom3D, kFALSE, kTRUE);
  glViewRPhi->CurrentCamera().Dolly(zoomRPhi, kFALSE, kTRUE);
  glViewRhoZ->CurrentCamera().Dolly(zoomRhoZ, kFALSE, kTRUE);
}

void EveInitializer::SetupBackground()
{
  TEnv settings;
  GetConfig(&settings);
  TEveViewerList *viewers = gEve->GetViewers();
  
  for(TEveElement::List_i i = viewers->BeginChildren(); i != viewers->EndChildren(); i++)
  {
    TEveViewer* view = ((TEveViewer*)*i);
    
    if((strcmp(view->GetName(),"3D View MV")!=0) &&
       (strcmp(view->GetName(),"RPhi View")!=0) &&
       (strcmp(view->GetName(),"RhoZ View")!=0)){
      continue;
    }
    view->GetGLViewer()->SetClearColor(settings.GetValue("background.color",1));
  }
}

void EveInitializer::ImportMacros()
{
  // Put macros in the list of browsables, add a macro browser to
  // top-level GUI.
  
  TString  hack = gSystem->pwd(); // Problem with TGFileBrowser cding
  
  TString macdir("$(ALICE_ROOT)/EVE/macros/data");
  gSystem->ExpandPathName(macdir);
  
  TFolder* f = gEve->GetMacroFolder();
  void* dirhandle = gSystem->OpenDirectory(macdir.Data());
  if (dirhandle != 0)
  {
    const char* filename;
    TPMERegexp re("\\.C$");
    TObjArray names;
    while ((filename = gSystem->GetDirEntry(dirhandle)) != 0)
    {
      if (re.Match(filename))
      names.AddLast(new TObjString(filename));
    }
    names.Sort();
    
    for (Int_t ii=0; ii<names.GetEntries(); ++ii)
    {
      TObjString * si = (TObjString*) names.At(ii);
      f->Add(new TEveMacro(Form("%s/%s", macdir.Data(), (si->GetString()).Data())));
    }
  }
  gSystem->FreeDirectory(dirhandle);
  
  gROOT->GetListOfBrowsables()->Add(new TSystemDirectory(macdir.Data(), macdir.Data()));
  
  {
    TEveBrowser   *br = gEve->GetBrowser();
    TGFileBrowser *fb = 0;
    fb = br->GetFileBrowser();
    fb->GotoDir(macdir);
    {
      br->StartEmbedding(0);
      fb = br->MakeFileBrowser();
      fb->BrowseObj(f);
      fb->Show();
      br->StopEmbedding();
      br->SetTabTitle("Macros", 0);
      br->SetTab(0, 0);
    }
  }
  gSystem->cd(hack);
}

void EveInitializer::GetConfig(TEnv *settings)
{
  TEveException kEH("EveInitializer::GetConfig");
  
  if(settings->ReadFile(Form("%s/.eve_config",gSystem->Getenv("HOME")), kEnvUser) < 0)
  {
    if(settings->ReadFile(Form("%s/eve_config",gSystem->Getenv("HOME")), kEnvUser) < 0)
    {
      Warning(kEH," could not find eve_config in home directory! Trying default one in $ALICE_ROOT/EVE/EveBase/");
      if(settings->ReadFile(Form("%s/EVE/EveBase/eve_config",gSystem->Getenv("ALICE_ROOT")), kEnvUser) < 0)
      {
        Error(kEH,"could not find eve_config file!.");
        exit(0);
      }
    }
  }
}



