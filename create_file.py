f = open("data.csv",'w')

st = ""


m = 2
c = 2
for x in range(1,101):
    y = m*x + c 
    st+=str(x)+","+str(y)+"\n"

f.write(st)

print("Done")

