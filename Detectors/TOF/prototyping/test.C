{
  /*
  TString dir("/home/noferini/Soft/alice/O2/O2");
  TString geom_dir = dir + "/Detectors/Geometry/";
  gSystem->Setenv("GEOMPATH",geom_dir.Data());

  FairRunSim* run = new FairRunSim();
  run->SetUseFairLinks(kTRUE);

  new TGeant3TGeo("C++ Interface to Geant3");
  new TGeoManager;

  run->SetMaterials("media.geo"); // Materials

  printf("prova %i -> \n",gGeoManager->GetListOfVolumes()->GetEntries());
 
  o2::field::MagneticField *magField = new o2::field::MagneticField("Maps","Maps", -1., -1., o2::field::MagFieldParam::k5kG);
  run->SetField(magField);


  o2::Passive::Cave *cave = new o2::Passive::Cave();
  cave->SetGeometryFileName("cave.geo");
  run->AddModule(cave);

  o2::TOF::Detector* tof = new o2::TOF::Detector("TOF", kTRUE);
  run->AddModule(tof);

  FairPrimaryGenerator* primGen = new FairPrimaryGenerator();

 FairBoxGenerator* boxGen = new FairBoxGenerator(211, 10);

  //boxGen->SetThetaRange(0.0, 90.0);
  boxGen->SetEtaRange(-0.9,0.9);
  boxGen->SetPRange(0.1, 5);
  boxGen->SetPhiRange(0., 360.);
  boxGen->SetDebug(kTRUE);

  primGen->AddGenerator(boxGen);

  run->SetGenerator(primGen);

  run->Init();
*/


  new TGeant3TGeo("C++ Interface to Geant3");
  new TGeoManager;

  printf("prova2 %i -> \n",gGeoManager->GetListOfVolumes()->GetEntries());

  o2::TOF::Detector* tof = new o2::TOF::Detector("TOF", kTRUE);
  tof->ConstructGeometry();

  gGeoManager->SetTopVolume(gGeoManager->GetVolume("FTOA"));

  printf("prova2 %i -> %x\n",gGeoManager->GetListOfVolumes()->GetEntries(),gGeoManager->GetVolume("FTOA"));

  gGeoManager->Export("TOFgeom.root");
}
