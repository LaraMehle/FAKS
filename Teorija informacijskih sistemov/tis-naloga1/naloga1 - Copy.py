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

"""
  list je niz znakov na vhodu, če je vhodS prazen seznam bo kodirala če pa ni prazen pa bo dekodirala torej seznamS je slovar
  vedno vrnemo izhod in pa kompresijsko razmerje
  python3 test_naloga1.py primeri --> s tem testiraš
  lahko tudi samo python3 test_naloga1.py primeri 1 --> s tem testiraš samo prvi primer
  velikost 3. primera je velikost problema ki bo testiran na ocenjevanju

  json je vhod, vhod s je prazen torej kodiramo, izhod so kode in pa slovar, in kompresijsko razmerje 1.primer
  2. primer je dekodiranje, 3. primer je kodiranje
  2. primer -> vhod so kode torej indeksi, potem je slovar. Pri preverjanju izhoda so neki u torej unicode znaki --> če je pa znak v standardnem ASCII pa napiše normalen znak
  pretvarjanje med znakom in številko je samo navaden typecast

  NAMIG: prav pride counter iz collections

  Izvedemo kodiranje ali dekodiranje z algoritmom BPE. 
  Najvecja dolzina vhodS je 4096.

  Parameters
  ----------
  vhod : list
    Seznam vhodnih znakov: bodisi znaki abecede (ASCII)
    (ko kodiramo) bodisi indeksi 
    (ko dekodiramo).
  vhodS : list 
    Seznam ASCII kod in parov indeksov 
    ce je []: kodiramo vhod
    sicer: dekodiramo vhod

  Returns
  -------
  (izhod, izhodS, R) : tuple[list, list, float]
      izhod : list
          Ce kodiramo: "izhod" je kodiran "vhod", 
          Ce dekodiramo: "izhod" je dekodiran "vhod"
      izhodS : list
          Ce kodiramo: seznam ASCII kod in parov indeksov
          Ce dekodiramo: []
      R : float
Kompresijsko razmerje
  """

