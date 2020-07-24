class UnionFind():
   def __init__(self, n):
       self.n = n
       self.root = [-1]*(n+1)
       self.rnk = [0]*(n+1)
   def Find_Root(self, x):
       if(self.root[x] < 0):
           return x
       else:
           self.root[x] = self.Find_Root(self.root[x])
           return self.root[x]
   def Unite(self, x, y):
       x = self.Find_Root(x)
       y = self.Find_Root(y)
       if(x == y):
           return
       elif(self.rnk[x] > self.rnk[y]):
           self.root[x] += self.root[y]
           self.root[y] = x
       else:
           self.root[y] += self.root[x]
           self.root[x] = y
           if(self.rnk[x] == self.rnk[y]):
               self.rnk[y] += 1
   def isSameGroup(self, x, y):
       return self.Find_Root(x) == self.Find_Root(y)
   def Count(self, x):
       return -self.root[self.Find_Root(x)]

def krascal(n, D):
    h = UnionFind(n)
    D.sort(key=lambda x: x[2])
    cnt = 0
    for x, y, c in D:
        if not h.isSameGroup(x, y):
            print(rdic[x], "-", rdic[y])
            h.Unite(x, y)
            cnt += c
    return cnt


if __name__ == '__main__':
    dic = {"Osaka": 0, "Kyoto": 1, "Kanazawa": 2, "Nagoya": 3, "Sizuoka": 4, "Toyama": 5, "Nagano": 6, "Takasaki": 7, "Tokyo": 8, "Yokohama": 9}
    rdic = list(dic.keys())
    dist = [["Osaka", "Kyoto", 50], ["Kyoto", "Kanazawa", 250], ["Kyoto", "Nagoya", 200], ["Nagoya", "Sizuoka", 120], ["Nagoya", "Kanazawa", 220], ["Kanazawa", "Toyama", 45], ["Toyama", "Nagano", 120], ["Nagano", "Sizuoka", 100], ["Nagano", "Takasaki", 60], ["Takasaki", "Tokyo", 60], ["Sizuoka", "Yokohama", 75], ["Yokohama", "Tokyo", 30]]
    for i in range(len(dist)):
        dist[i][0] = dic[dist[i][0]]
        dist[i][1] = dic[dist[i][1]]
        

    print("sum cost", krascal(10, dist))

