option="--dummy-elecmap"
nevents=30

# first orbit of the run
orbitFirst=0
# first orbit to sample
orbitFirstSampled=0
# interactionRate (default 50000)
interactionRate=50000

echo "=== Performing simulation (aka hit creation)"
o2-sim-serial --seed 4242 -n ${nevents} -g fwmugen -e TGeant3 -j 1 -m MCH &> sim-serial.log

echo "=== Digitizing..."
o2-sim-digitizer-workflow \
--configKeyValues "HBFUtils.orbitFirst=${orbitFirst};HBFUtils.orbitFirstSampled=${orbitFirstSampled}" \
--onlyDet MCH -b &> sim-digitizer.log

echo "=== Convert sim digits to mch binary format"
o2-mch-sim-digits-reader-workflow --mch-digit-infile mchdigits.root | \
o2-mch-digits-writer-workflow -b --binary-file-format 3 --outfile \
digits.sim.out &> /dev/null

echo "=== Textual dump of sim digits"
o2-mch-digits-file-dumper --infile digits.sim.out \
--print-digits &> digits.sim.txt

echo "=== Conversion digits -> raw (MCH)"
o2-mch-digits-to-raw --input-file mchdigits.root \
--configKeyValues="MCHCoDecParam.sampaBcOffset=12345" \
--output-dir ./raw/MCH --file-per-link $option &> \
digits-to-raw.log

echo "=== Conversion raw -> digit + real digits to mch binary format"
o2-raw-file-reader-workflow --input-conf raw/MCH/MCHraw.cfg -b | \
o2-mch-raw-to-digits-workflow ${option} \
--configKeyValues="MCHCoDecParam.sampaBcOffset=12345" -b | \
o2-mch-digits-writer-workflow -b --binary-file-format 3 --outfile \
digits.real.out &> raw-to-digits.log

echo "=== Textual dump of real digits"
o2-mch-digits-file-dumper --infile digits.real.out \
--print-digits &> digits.real.txt

