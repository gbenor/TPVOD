import subprocess


def interaction_formatting (s1, s2):
    t = ""
    for i in range(len(s1)):
        if s1[i]!=" ":
            t+="-"
        elif s2[i]!=" ":
            t+="|"

    return t

def rnaHybrid (mrna, mirna):
    cmd = "RNAhybrid -s 3utr_human {} {}".format(mrna, mirna)
    print (cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    a = p.stdout.readlines()
    a = [l.decode(encoding='unicode-escape') for l in a]
    b = [[a[l], a[l + 1], a[l + 2], a[l + 3]] for l in range(len(a)) if a[l].startswith('target 5')]
    end = b[0][0].find("3") - 1
    f = [t[10:end] for t in b[0]]
    f = [t[::-1] for t in f]
    tar = interaction_formatting(f[0], f[1])[1:]
    tar += "-" * max(0, (22 - len(tar)))
    #tar = tar[:22]

    mir = interaction_formatting(f[3], f[2])[1:]
    mir += "-" * max(0, (22 - len(mir)))
    #mir = mir[:22]
    return tar, mir
