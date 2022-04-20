#!/usr/bin/perl

# Run as: perl size_ao2d.pl [file1] [file2] ...
# file1, file2, ... have to be local or remote AO2D.root files
# Remote file locations should be specified as: alien:///alice/.../AO2D.root

use warnings;
use strict;

my @tableNames = ("O2bc",
                  "O2fdd",
                  "O2collision", 
                  "O2track.par", 
                  "O2track.parcov", 
                  "O2track.extra", 
                  "O2zdc", 
                  "O2v0",
                  "O2fv0a",
                  "O2fv0c",
                  "O2ft0", 
                  "O2cascade", 
                  "O2calo", 
                  "O2calotrigger", 
                  "O2muon", 
                  "O2muoncluster",
                  "O2mcparticle",
                  "O2mccollision",
                  "O2mccalolabel",
                  "O2mctracklabel",
                  "O2mccollisionlabel"
                  #"O2tof"
                  #"O2mcparticle", "O2mccollision", "O2mccalolabel", "O2mctracklabel", "O2mccollisionlabel"
                  #"DbgEventExtra"
                  );

# my %totalSize = map { $_ => 0 } @tableNames;
# my %comprSize = map { $_ => 0 } @tableNames;

my %fields = ( 
"par" => ["CollisionsID", "TrackType",
                  "X", "Alpha",
                  "Y", "Z", "Snp", "Tgl",
                  "Signed1Pt",
         ],
"parcov" => [ "SigmaY", "SigmaZ", "SigmaSnp", "SigmaTgl", "Sigma1Pt", "RhoZY", "RhoSnpY",
                  "RhoSnpZ", "RhoTglY", "RhoTglZ", "RhoTglSnp", "Rho1PtY", "Rho1PtZ",
                  "Rho1PtSnp", "Rho1PtTgl",
            ],
"extra" => [      "TPCInnerParam", "Flags", "ITSClusterMap",
                  "TPCNClsFindable", "TPCNClsFindableMinusFound", "TPCNClsFindableMinusCrossedRows",
                  "TPCNClsShared", "TRDPattern", "ITSChi2NCl",
                  "TPCChi2NCl", "TRDChi2", "TOFChi2",
                  "TPCSignal", "TRDSignal", "TOFSignal", "Length", "TOFExpMom", "TrackEtaEMCAL", "TrackPhiEMCAL"
           ]
);                  

my $totalU = 0;
my $totalC = 0;

for (@tableNames) {
  my $table = $_;
  my $O2table = undef;
  if (/(.*)\.(.*)/) {
    #tree name (O2bc, ...)
    $table = $1;
    # fields name (par, parcov or extra)
    $O2table = $2;  
    #print "### 1 = $table   2 = $O2table  ### \n";
  }

  my $uncompressed = 0;
  my $compressed = 0;

  my @branch_names;
  my @uncompressed_br = (0) x 200;
  my @compressed_br = (0) x 200;
  my $arr_ind = 0;
  my $nbranches = 0;
  my $pass = 0;

  for (@ARGV) {
  
    my @treeInfo = `root -b -q treeinfo.C'("$_", "$table")'`;
    
    $arr_ind = -1;
  # print %size;

    my $adding = 0;
    for (@treeInfo) 
    {
      if (/\*Br/) {
        $arr_ind += 1;
        $adding = 0;
        my @elem = split ":";
        # branch name
        $elem[1] =~ s/\s//g;
        $branch_names[$arr_ind] = "${table}.$elem[1]" if ($pass == 0);
        
        if (defined $O2table) {
          for (@{$fields{$O2table}}) {
            $adding = 1 if ("f$_" eq $elem[1]);
          }
          #print $O2table, " branch=", $elem[1], "  adding=",$adding,"\n";
        } else { 
          $adding = 1;
        }
      }
      
      $uncompressed += $1 if ($adding && /Total  Size=\s+(\d+) bytes/);
      $compressed   += $1 if ($adding && /File Size  =\s+(\d+)/);

      $uncompressed_br[$arr_ind] += $1 if (/Total  Size=\s+(\d+) bytes/);
      $compressed_br[$arr_ind] += $1 if (/File Size  =\s+(\d+)/);  
    }
    $pass += 1;
  }
  
  print "$_ \t $uncompressed \t $compressed \t";
  print ($uncompressed / $compressed) if ($compressed > 0);
  print "\n";

  # print branch Compression
  for my $i (0 .. $arr_ind)
  {
    #print "$branch_names[$i] \t $uncompressed_br[$i] \t $compressed_br[$i] \t";
    #print ($uncompressed_br[$i] / $compressed_br[$i]) if ($compressed_br[$i] > 0);
    #print "\n";
  }

  
  $totalU += $uncompressed;
  $totalC += $compressed;
#   last;
}

print "Total \t $totalU \t $totalC \t " . ($totalU / $totalC) . "\n";

# *Br    0 :fCollisionsID : fCollisionsID/I                                    *
# *Entries :   992709 : Total  Size=    4077499 bytes  File Size  =     139185 *
# *Baskets :      992 : Basket Size=       4096 bytes  Compression=  29.10     *
