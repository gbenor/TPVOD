class InteractionRichPresentation (object) :


    def __init__(self, mrna_bulge, mrna_inter, mir_inter, mir_bulge):
        self.mrna_bulge = mrna_bulge
        self.mrna_inter = mrna_inter
        self.mir_inter = mir_inter
        self.mir_bulge = mir_bulge
        self.verify_corectness()
        self.site = self.extract_site()
        self.mir_bulges_count, self.mrna_bulges_count = self.count_bulges()

    def verify_corectness (self):
        def check_pairs (p1, p2):
            for i in range (min(len(p1), len(p2))):
                assert (p1[i]==' ' or p2[i]==' '), "IRP Error. both interaction and bulge have nt value. one of them should be empty (space)."
        check_pairs(self.mrna_bulge, self.mrna_inter)
        check_pairs(self.mir_bulge, self.mir_inter)

    def interaction_iterator (self):
        i=0
        while i<len(self.mir_inter) :
            if self.mir_inter[i]!=' ':
                yield (i ,self.mrna_inter[i]+self.mir_inter[i])
            i+=1

    def mir_iterator (self):
        #make sure the iterator won't access out of range of the mir variables
        mir_len = max (len(self.mir_inter), len(self.mir_bulge))
        mir_inter = self.mir_inter + " "*mir_len
        mir_bulge = self.mir_bulge + " "*mir_len

        i=0
        while i<mir_len:
            if mir_inter[i]!=' ' or mir_bulge[i]!=' ':
                yield (i, self.mix_inter_bulge(mir_inter[i], mir_bulge[i]))
            i+=1

    def mir_pairing_iterator (self):
        for i, mir in self.mir_iterator():
            try:
                mrna = self.mrna_inter[i]
            except IndexError:
                mrna = " "
            yield mrna+mir


    def extract_seed (self, start=1, end=8) :
        #notice: if there a bulge in the mrna, it will skip it. the function return 8 consecutive mirna nt.
        mrna_bulge=""
        mrna_inter=""
        mir_inter=""
        mir_bulge=""
        mir_i = self.mir_iterator()
        for i in range(start-1):
            next(mir_i)
        s, mir = next(mir_i)
        for k in range(end-start):
            e, mir = next(mir_i)
        e+=1
        mrna_bulge += self.mrna_bulge[s:e]
        mrna_inter += self.mrna_inter[s:e]
        mir_inter += self.mir_inter[s:e]
        mir_bulge += self.mir_bulge[s:e]
        return InteractionRichPresentation(mrna_bulge, mrna_inter, mir_inter, mir_bulge)

    def extract_site(self):
        def first_char (s1, s2):
            i = 0
            while s1[i]==" " and s2[i]==" ":
                i+=1
            return i
        # make sure the iterator won't access out of range of the mir variables
        mrna_site =""
        mir_len = max(len(self.mir_inter), len(self.mir_bulge))
        mir_start = first_char(self.mir_inter, self.mir_bulge)
        for i in range (mir_start, mir_len):
            try:
                mrna_site+=self.mix_inter_bulge(self.mrna_inter[i], self.mrna_bulge[i])
            except IndexError:
                #probablly the bulge is longer
                mrna_site+=self.mrna_bulge[i]
        mrna_site = mrna_site.replace (" ","")
        return mrna_site

    def count_bulges (self) :
        mir_bulges_count = len(self.mir_bulge.split())
        mrna_bulges_count = len(self.mrna_bulge.split())
        return  mir_bulges_count, mrna_bulges_count

    def mix_inter_bulge(self, i, b):
        if i!=' ' and b!=' ':
            raise Exception ("both interaction and bulge have nt value. one of them should be empty (space).")
        return chr(ord(i) + ord(b) - ord(' '))


    def tostring(self):
        classstr = ""
        classstr = classstr + "target_bulge:       {}\n".format(self.mrna_bulge)
        classstr = classstr + "target_interaction: {}\n".format(self.mrna_inter)
        classstr = classstr + "mirna_interaction:  {}\n".format(self.mir_inter)
        classstr = classstr + "mirna_bulge:        {}\n".format(self.mir_bulge)
        classstr +="\n"
        classstr = classstr + "site ({}):          {}\n".format(len(self.site), self.site)

        #classstr = classstr + "mrna_bulges_count: {} \nmir_bulges_count:  {}\n".format(self.mrna_bulges_count,self.mir_bulges_count)

        return classstr

    def replace_T_U (self) :
        self.mrna_bulge = self.mrna_bulge.upper().replace('T', 'U')
        self.mrna_inter = self.mrna_inter.upper().replace('T', 'U')
        self.mir_inter = self.mir_inter.upper().replace('T', 'U')
        self.mir_bulge = self.mir_bulge.upper().replace('T', 'U')

    def set_site(self, site):
        self.site = site


    def __str__(self):
        return self.tostring()


