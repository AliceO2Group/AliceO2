#!/bin/sh

CRED="/Users/laurent/MCH Mapping-0a71181801a9.json"
CHAMBERS="CH1R CH1L CH2R CH2L CH3R CH3L CH4R CH4L CH5R CH5L CH6R CH6L CH7R CH7L CH8L
CH8R CH9L CH9R CH10R CH10L"

rm cru.map
rm fec.map
for chamber in $CHAMBERS; do
   echo "Generating code for ${chamber}"
   ./elecmap.py -gs "MCH Electronic Mapping" -s ${chamber} \
   --credentials="$CRED" -c  ${chamber} \
  --cru_map ${chamber}.cru.map \
  --fec_map ${chamber}.fec.map
  cat ${chamber}.fec.map >> fec.map.tmp
  cat ${chamber}.cru.map >> cru.map.tmp
  rm ${chamber}.fec.map
  rm ${chamber}.cru.map
  sleep 10
done

sort cru.map.tmp -o cru.map
sort fec.map.tmp -o fec.map

rm cru.map.tmp
rm fec.map.tmp

# generate SolarCrate.cxx

./elecmap.py -gs "MCH Electronic Mapping" --credentials="$CRED" --dcs-to-solar -s "DCS Alias To Solar Crate"

