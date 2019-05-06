void dump()
{
  AliHLTSystem* pHLT = AliHLTPluginBase::GetInstance();
  AliHLTConfiguration overrideClusterTransformation("TPC-ClusterTransformation", "TPCClusterTransformation", "TPC-HWCFDecoder", "-use-orig-transform -do-mc");
  AliHLTConfiguration dumper("Dumper", "GPUDump", "TPC-ClusterTransformation TRD-tracklet-reader", "");
  AliHLTConfiguration overrideTracker("TPC-TR", "TPCCATracker", "TPC-ClusterTransformation Dumper", "-GlobalTracking -SearchWindowDZDR 2.5");
}
