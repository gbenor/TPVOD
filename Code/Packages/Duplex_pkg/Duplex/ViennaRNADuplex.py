import RNA
from collections import Counter
#from SeedFeatures import *
from Duplex.InteractionRichPresentation import *
import os
import pandas as pd


class Duplex(object):
    def __init__(self, structure, i,j, energy):
        self.structure = structure
        self.i = i
        self.j = j
        self.energy = energy



class ViennaRNADuplex(object):

    def __init__(self, o_mir, o_mrna_extended,tmp_dir, constraint=None):
        self.mir = o_mir
        self.mrna = o_mrna_extended
        self.constraint = constraint
        self.tmp_dir = tmp_dir

        v_dp = RNA.duplexfold(self.mir, self.mrna)
        c_dp = self.RNAplex(self.mir, self.mrna, constraint=constraint)
        self.equal = self.duplex_equal(v_dp, c_dp)

        # if constraint:
        #     if self.equal:
        #         print ("although the constraint, the duplex are equal.")
        #     else:
        #         print ("great success. the duplex are not equal")
        # else:
        #     assert self.equal, "The duplex are not equal {} {} {} {} {} {}".format(v_dp.i, c_dp.i,v_dp.j, c_dp.j,v_dp.structure, c_dp.structure)

        self.duplex = c_dp
        # self.duplex = c_dp if constraint else v_dp
        self.duplex_score = -self.duplex.energy

        (mir_pairing, mrna_pairing) = self.duplex.structure.split('&')
        self.mir_coor = (self.duplex.i - len(mir_pairing), self.duplex.i)
        self.mrna_coor = (self.duplex.j - 1, self.duplex.j + len(mrna_pairing) - 1)
        self.mir_idx = self.find_pairing(mir_pairing, '(')
        self.mrna_idx = self.find_pairing(mrna_pairing, ')')
        self.active_mir = self.mir[self.mir_coor[0]:self.mir_coor[1]]
        self.active_mrna = self.mrna[self.mrna_coor[0]:self.mrna_coor[1]]
        self.mrna_idx = self.mrna_idx[::-1]
        self.mir_len = len(self.active_mir)
        self.mrna_len = len(self.active_mrna)

        self.num_of_pairs = len(self.find_pairing(mir_pairing, '('))

        self.IRP = self.parse_interaction()

    def get_features(self):
        d0 = 1 if self.constraint else 0
        d1 = 1 if self.equal else 0
        return pd.DataFrame(data=[[d0, d1]], columns=["constraint", "duplex_RNAplex_equals"])

    def duplex_equal (self, d1, d2):
        i =  d1.i == d2.i
        j =  d1.j == d2.j
        structure =  d1.structure == d2.structure
        return i and j and structure



    def RNAplex (self, mir, mrna, constraint=None):
        plex_in_file = self.tmp_dir / 'plex.in'
        plex_out_file = self.tmp_dir / 'plex.out'

        f = open(plex_in_file, 'w')
        f.write("> query1\n")
        f.write(mir + "\n")
        f.write("> target1\n")
        f.write(mrna + "\n")
        if constraint:
            f.write(constraint + "\n")
        f.close()
        consflag = "-C" if constraint else ""


        cmd = "RNAplex {consflag} < {infile} > {outfile}".format(consflag=consflag, infile=plex_in_file,  outfile=plex_out_file)
   #     cmd = "RNAduplex {consflag} < {infile} > {outfile}".format(consflag=consflag, infile=plex_in_file,  outfile=plex_out_file)

        os.system(cmd)

        f = open(plex_out_file, 'r')
        plexout = f.readlines()
        print (plexout)
        f.close()

        plexout = plexout[2]
        structure = plexout.split()[0]
        # energy = plexout.split()[-1]
        # energy = float(energy[1:-1])

        # ij = plexout[plexout.find("i:"):plexout.find("i:")+12]
        # ij = ij.split()[0]
        # ij = ij.split(',')
        # i = int(ij[0][2:])
        # j = int(ij[1][2:])
        try:
            energy = plexout.split()[4]
            energy = float(energy[1:-1])
        except ValueError:
            energy = plexout.split()[5]
            energy = float(energy[1:-1])


        mir = plexout.split()[1]
        mrna = plexout.split()[3]

        i=int(mir.split(',')[1])
        j=int(mrna.split(',')[0])


        return Duplex (structure, i, j, energy)






    def find_pairing(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def parse_interaction (self):
        mir = self.active_mir
        mrna = self.active_mrna

        mrna_bulge = ""
        mrna_inter = ""
        mir_inter = ""
        mir_bulge = ""
        mir_i = 0
        mrna_i = self.mrna_len - 1
        if (self.mir_coor[0] > 0):
            mir_bulge+=self.mir[:self.mir_coor[0]]
            mir_inter+=" "*self.mir_coor[0]
            mrna_inter+=" "*self.mir_coor[0]
            mrna_addition_len = self.mrna_coor[1]+self.mir_coor[0] - self.mrna_coor[1]
            mrna_bulge_additon=self.mrna[self.mrna_coor[1]:self.mrna_coor[1]+self.mir_coor[0]]
            mrna_bulge_additon = mrna_bulge_additon + "#" * (mrna_addition_len - len(mrna_bulge_additon))

            # if mrna_bulge_additon == "":
            #     mrna_bulge_additon="#"*self.mir_coor[0]
            mrna_bulge+=mrna_bulge_additon[::-1]


        for i in range(len(self.mir_idx)) :
            #deal with the bulge
            mir_bulge_idx = range (mir_i, self.mir_idx[i])
            mir_bulge+=mir[mir_i:self.mir_idx[i]]
            mrna_bulge_idx= range (mrna_i, self.mrna_idx[i], -1)
            mrna_bulge+=mrna[mrna_i:self.mrna_idx[i]: -1]
            c_pos = max (len(mrna_bulge_idx), len(mir_bulge_idx))
            mrna_inter += " " * c_pos
            mir_inter += " " * c_pos
            mrna_bulge+=" "*(c_pos - len(mrna_bulge_idx))
            mir_bulge+=" "*(c_pos - len(mir_bulge_idx))
            #deal with the interaction
            mir_bulge+=" "
            mir_inter+=mir[self.mir_idx[i]]
            mrna_bulge+=" "
            mrna_inter+=mrna[self.mrna_idx[i]]
            #update the idx
            mir_i=self.mir_idx[i] + 1
            mrna_i= self.mrna_idx[i] - 1
        #deal with the tail
        # if (mir_i<=len(mir)) :
        #     mir_bulge+=mir[mir_i:]
        # if (mrna_i >=0) :
        #     mrna_bulge+=mrna[mrna_i::-1]
        full_mir = self.mir
        if (mir_i<=len(full_mir)) :
            mir_bulge+=full_mir[mir_i:]
            addition = full_mir[mir_i:]
            mrna_bulge+="*"*len(addition)
        # if (mrna_i >=0) :
        #     mrna_bulge+=mrna[mrna_i::-1]

        return InteractionRichPresentation (mrna_bulge, mrna_inter, mir_inter, mir_bulge)


    def tostring(self):
        classstr = ""
        classstr += " {} \n".format(self.duplex.structure)
        classstr += self.IRP.__str__()
        classstr += "\n"

        classstr = classstr + "energy:  {} \n".format(-1*self.duplex_score)

     #   classstr = classstr + "site            ({}): {} \n".format(len(self.site), self.site)
        classstr = classstr + "active_mrna[-1] ({}): {} \n".format(len(self.active_mrna[::-1]), self.active_mrna[::-1])
        classstr = classstr + "active_mir      ({}): {} \n".format(len(self.active_mir), self.active_mir)
        classstr = classstr + "full_mir        ({}): {} \n".format(len(self.mir), self.mir)
        classstr = classstr + "offset = {} \n".format(self.mir_coor[0])





        #s = SeedFeatures(self)
        #classstr = classstr + "canonic = {} \n".format(s.canonic_seed)

   #     mir_seed, pairs_in_seed = self.extract_seed()
    #    classstr = classstr + "mir seed = {} \n".format(mir_seed)
     #   classstr = classstr + "pairs_in_seed = {} num_of_pairs = {} \n".format(pairs_in_seed, len(pairs_in_seed))

        return classstr


    def __str__(self):
        return self.tostring()
