import math
from collections import Counter
from collections import defaultdict

def entropija(pogostosti: list) -> float:
    stVseh = sum(pogostosti)
    result = 0
    for pogostost in pogostosti:
        if pogostost > 0:
            p = pogostost / stVseh
            result -= p * math.log2(p)
    return result

def naloga2(znacilke: dict, razredi: list, koraki: int) -> tuple:
    N = len(razredi)
    selected = [] #izbrane značilke
    vseZnacilke = list(znacilke.keys())

    for i in range(koraki):
        minH = float('inf')
        najboljsa = None

        for znacilka in vseZnacilke:
            if znacilka in selected: # če je že bila izbrana, jo preskočimo
                continue

            groups = defaultdict(lambda: Counter())
            for j in range(N):
                features = selected + [znacilka]
                values = []
                for f in features:
                    values.append(znacilke[f][j])
                key = tuple(values)
                groups[key][razredi[j]] += 1
            
            trenutenH = 0.0
            for counter in groups.values():
                count_sum = sum(counter.values())
                H_branch = entropija(list(counter.values()))
                trenutenH += (count_sum / N) * H_branch
            
            if trenutenH < minH:
                minH = trenutenH
                najboljsa = znacilka
        if najboljsa is None:
            break
        selected.append(najboljsa)

    
    koncnaEntropija = minH if najboljsa is not None else 0.0


    majority = {}
    counts = defaultdict(lambda: Counter())
    for j in range(N):
        key = tuple(znacilke[f][j] for f in selected)
        counts[key][razredi[j]] += 1

    for key, counter in counts.items():
        majority[key] = counter.most_common(1)[0][0]

    correct = 0
    for j in range(N):
        key = tuple(znacilke[f][j] for f in selected)
        pred = majority.get(key)
        if pred == razredi[j]:
            correct += 1
    tocnost = correct / N

    return (koncnaEntropija, tocnost)
