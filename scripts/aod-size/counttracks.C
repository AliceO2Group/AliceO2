void countperfile(const char *filename, size_t &ntracks, size_t &nevents, size_t &nbc) {
  TFile *file = TFile::Open(filename);
  if (!file) {
    printf("counttracks could not open file %s\n", filename);
  }
  ntracks = 0;
  nevents = 0;
  nbc = 0;
  TIter next(file->GetListOfKeys());
  TKey *key;
  while ((key = (TKey*)next())) {
    TString tname = TString::Format("%s/O2track", key->GetName());
    TString ename = TString::Format("%s/O2collision", key->GetName());
    TString bcname = TString::Format("%s/O2bc", key->GetName());

    TTree *tree_tracks = (TTree*) file->Get(tname);
    TTree *tree_events = (TTree*) file->Get(ename);
    TTree *tree_bc = (TTree*) file->Get(bcname);
    if (tree_events) {
      ntracks += tree_tracks->GetEntries();
      nevents += tree_events->GetEntries();
      if (tree_bc) nbc += tree_bc->GetEntries();
    }
  }
  printf("events: %zu  bc: %zu  tracks: %zu\n", nevents, nbc, ntracks);
}

void counttracks(const char *filenames)
{
  TString sfname(filenames);
  if (sfname.BeginsWith("alien"))
    TGrid::Connect("alien:");
  TObjArray *list = sfname.Tokenize(" ");
  size_t ntracks, nevents, nbc;
  size_t ntrackstot=0, neventstot=0, nbctot=0;
  for (int i = 0; i < list->GetEntriesFast(); i++) {
    TString filename = list->At(i)->GetName();
    if (filename.Length() < 2)  continue;
    //printf("%s\n", filename.Data());
    countperfile(filename.Data(), ntracks, nevents, nbc);
    ntrackstot += ntracks;
    neventstot += nevents;
    nbctot += nbc;
  }
  printf("eventstot: %zu  bctot: %zu  trackstot: %zu\n", neventstot, nbctot, ntrackstot);
}
