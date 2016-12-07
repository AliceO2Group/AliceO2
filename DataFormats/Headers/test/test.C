test() {
gROOT->LoadMacro("DataHeader.cxx++");
AliceO2::Base::ROOTobjectHeader m;
AliceO2::Base::DataHeader d;
auto b = AliceO2::Base::BaseHeader::compose(d,m);
AliceO2::Base::BaseHeader* h = AliceO2::Base::BaseHeader::get(b.first.get());
hexDump("",h,b.second);
h->get(AliceO2::Base::ROOTobjectHeader::sHeaderType);
AliceO2::Base::BaseHeader* hr = h->get(AliceO2::Base::ROOTobjectHeader::sHeaderType);
hexDump("",hr,hr->size());
h->get(AliceO2::Base::ROOTobjectHeader::sHeaderType);
h->get("oiuoiu");
h->get("ROOTmtd");
h->get("BaseHDR");
h->get("BaseHDe");
string s{"asdgasdgasgd"};
auto bs = AliceO2::Base::BaseHeader::compose(s);
hexDump("",bs.first.get(),bs.second);
}
