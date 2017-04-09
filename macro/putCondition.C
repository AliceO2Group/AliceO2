using namespace o2::CDB;
void putCondition()
{
  Manager *cdb = Manager::Instance();
  cdb->setDefaultStorage("local://");
  ConditionId id("DET/Align/Data", 190000, 191000);
  TH1F *h1 = new TH1F("aHisto", "yeah", 100, 0, 10);
  ConditionMetaData *md = new ConditionMetaData("any comment");
  md->addDateToComment();
  Condition *e = new Condition(h1, id, md);
  cdb->putObject(e);
}
