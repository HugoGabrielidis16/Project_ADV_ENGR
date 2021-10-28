

def affichage(l):
    for i in range(len(l)):
        print(l[i])


def difference(l1,l2):
    s = 0
    for i in range(len(l1)):
        if l1[i] != l2[i]:
            s += 1
    return s