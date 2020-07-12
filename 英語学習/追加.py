# 変える

dic={"数字":"number", "数学":"math", "論理":"logic", "57は":"素数"}

# 何も入力せずEnterを押すと終わる.
# ここより下は触らない.
#-------------------------------------------------------------------------------
print()


while True:
    s = input("日本語: ")
    t = input("英語: ")
    print()
    if s and t:
        dic[s] = t
    else:
        break
print()
print("dic　=",dic)
print()
