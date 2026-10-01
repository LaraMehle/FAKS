from collections import Counter

def naloga1(vhod: list, vhodS: list) -> tuple[list, list, float]:
    MAX_SLOVAR = 4096
    izhod = []
    izhodS = []
    R = float('nan')

    slovar = {chr(i): i for i in range(256)}  # ASCII slovar
    obratni_slovar = {i: chr(i) for i in range(256)}

    if vhodS == []:
        nov_vhod = vhod.copy()

        while len(slovar) < MAX_SLOVAR:
            pari = Counter()
            for i in range(len(nov_vhod) - 1):
                par = (nov_vhod[i], nov_vhod[i + 1])
                pari[par] += 1

            if not pari: # Če ni parov, prekinemo
                break
            
            if all(stevilo == 1 for stevilo in pari.values()): # Če so vsi pari enako pogosti, prekinemo
                break

            najpogostejsi = pari.most_common(1)[0][0]

            if len(slovar) < MAX_SLOVAR:
                nov_indeks = len(slovar)
                slovar[najpogostejsi] = nov_indeks
                obratni_slovar[nov_indeks] = najpogostejsi


            # Zamenjamo najpogostejši par z novim indeksom
            nov_niz = []
            i = 0
            while i < len(nov_vhod) - 1:
                par = (nov_vhod[i], nov_vhod[i + 1])
                if par in slovar:
                    nov_niz.append(slovar[par])
                    i += 2
                else:
                    nov_niz.append(nov_vhod[i])
                    i += 1

            # Doda zadnji znak, če je potrebno
            if i < len(nov_vhod):
                nov_niz.append(nov_vhod[i])

            nov_vhod = nov_niz

        # Končno kodiranje
        koncni_izhod = []
        for x in nov_vhod:
            if x in slovar:
                koncni_izhod.append(slovar[x])
            elif isinstance(x, str):
                koncni_izhod.append(ord(x))
            else:
                koncni_izhod.append(x)
        #print("Končno kodiran vhod:", koncni_izhod)

        izhod = koncni_izhod
        izhodS = list(slovar.items())
        R = (8 * len(vhod)) / (12 * len(koncni_izhod))

    else:
        trenutni = len(vhodS) - 1
        temp = vhod.copy()

        #Gremo od konca proti začetku slovarja in dekodiramo
        while trenutni > 255:
            trenutniZnak = vhodS[trenutni]
            dekodirani = []

            for i in range(len(temp)):
                if temp[i] == trenutni: # Našli smo znak ki ga želimo dekodirati
                    for element in trenutniZnak:
                            dekodirani.append(element)
                else:
                    dekodirani.append(temp[i])

            temp = dekodirani
            trenutni -= 1

        #Končno dekodiranje
        znaki = []
        for koda in temp:
            if koda < 256:
                znaki.append(chr(koda))
            else:
                znaki.append(koda) 
        

        izhod = znaki
        izhodS = []
        R = (8 * len(izhod)) / (12 * len(vhod))

    return (izhod, izhodS, R)
