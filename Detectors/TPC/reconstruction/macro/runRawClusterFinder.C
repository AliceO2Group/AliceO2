void runRawClusterFinder(TString fileInfo, TString pedestalFile, TString outputFileName="clusters.root", Int_t maxEvents=-1, ClustererType clustererType=ClustererType::HW)
{
   RawClusterFinder::ProcessEvents(fileInfo, pedestalFile, outputFileName, maxEvents, clustererType);
}
