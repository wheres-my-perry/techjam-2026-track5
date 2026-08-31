import csv, collections, sys
def bucket(l):
    return "<=341" if l<=341 else "342-512" if l<=512 else "513-768" if l<=768 else "769-1024" if l<=1024 else ">1024"
for name in sys.argv[1:]:
    c=collections.Counter(); gens=set()
    for r in csv.DictReader(open(f"data/manifests/{name}.csv",newline="")):
        c[(bucket(int(r["long"])), r["label"])]+=1
        if r["label"]=="1": gens.add(r["generator"])
    print(f"{name}: {sum(c.values())} rows, {len(gens)} fake generators")
    for b in ["<=341","342-512","513-768","769-1024",">1024"]:
        rr, ff = c[(b,"0")], c[(b,"1")]
        if rr or ff:
            print(f"   {b:10s} real {rr:6d}  fake {ff:6d}   ratio {ff/max(1,rr):.2f}")
