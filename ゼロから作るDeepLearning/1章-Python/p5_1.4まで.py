# データ型 type()
print(type(10)) # int
print(type(2.7)) # float
print(type("Hello")) # str

# 変数
x = 10
print(x) # 10
x = 100
print(x) # 100
y = 3.14
print(x*y) # 314.0
print(type(x*y)) # float

# リスト
a = [1, 2, 3, 4, 5]
# 虚無みを感じたので飛ばす

# class

class Man:
    def __init__(self,Name):
        self.name = Name
    def hello(self):
        print("Hello",self.name,"!")
        print(self.name,"is genius.")
    def goodbye(self):
        print("Good-bye",self.name,"!")

m = Man("masa-aa")
m.hello()
m.goodbye()