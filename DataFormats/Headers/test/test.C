test() {
gROOT->LoadMacro("DataHeader.cxx++");
o2::Base::ROOTobjectHeader m;
o2::Base::DataHeader d;
auto b = o2::Base::BaseHeader::compose(d,m);
o2::Base::BaseHeader* h = AliceO2::Base::BaseHeader::get(b.first.get());
hexDump("",h,b.second);
h->get(o2::Base::ROOTobjectHeader::sHeaderType);
o2::Base::BaseHeader* hr = h->get(AliceO2::Base::ROOTobjectHeader::sHeaderType);
hexDump("",hr,hr->size());
h->get(o2::Base::ROOTobjectHeader::sHeaderType);
h->get("oiuoiu");
h->get("ROOTmtd");
h->get("BaseHDR");
h->get("BaseHDe");
string s{"asdgasdgasgd"};
auto bs = o2::Base::BaseHeader::compose(s);
hexDump("",bs.first.get(),bs.second);
}
