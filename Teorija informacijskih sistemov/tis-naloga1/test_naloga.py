from collections import Counter

def naloga1(vhod: list, vhodS: list) -> tuple[list, list, float]:
    MAX_SLOVAR = 4096
    izhod = []
    izhodS = []
    R = float('nan')

    if vhodS == []:
        # Encoding part
        slovar = {chr(i): i for i in range(256)}  # ASCII slovar
        obratni_slovar = {i: chr(i) for i in range(256)}
        
        nov_vhod = vhod.copy()

        while len(slovar) < MAX_SLOVAR:
            pari = Counter()
            for i in range(len(nov_vhod) - 1):
                par = (nov_vhod[i], nov_vhod[i + 1])
                pari[par] += 1

            if not pari:  # Če ni parov, prekinemo
                break
            
            if all(stevilo == 1 for stevilo in pari.values()):  # Če so vsi pari enako pogosti, prekinemo
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
                if par == najpogostejsi:
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
            if isinstance(x, str):
                koncni_izhod.append(ord(x))
            else:
                koncni_izhod.append(x)

        izhod = koncni_izhod
        izhodS = list(slovar.items())
        R = (8 * len(vhod)) / (12 * len(koncni_izhod))

    else:
        # Decoding part
        def expand_token(token):
            if token < 256:
                return [token]
            else:
                pair = vhodS[token]
                expanded = []
                for element in pair:
                    expanded.extend(expand_token(element))
                return expanded
        
        # Process every token in the input
        result = []
        for token in vhod:
            result.extend(expand_token(token))
        
        izhod = result
        izhodS = []
        R = float('nan')

    return (izhod, izhodS, R)