{
  new TGeant3TGeo("C++ Interface to Geant3");
  new TGeoManager;

  printf("prova2 %i -> \n", gGeoManager->GetListOfVolumes()->GetEntries());

  o2::tof::Detector* tof = new o2::tof::Detector("TOF", kTRUE);
  tof->ConstructGeometry();

  gGeoManager->SetTopVolume(gGeoManager->GetVolume("FTOA"));

  printf("prova2 %i -> %x\n", gGeoManager->GetListOfVolumes()->GetEntries(), gGeoManager->GetVolume("FTOA"));

  gGeoManager->Export("TOFgeom.root");
}
