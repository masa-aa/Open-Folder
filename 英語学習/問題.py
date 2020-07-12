from random import randint

dic = {"とにかく":"anyway", "出荷する":"ship", "参加する、出席する":"attend", "成功した、うまくいく":"successful",
"社員、従業員":"employee", "拡大する":"expand", "（本、雑誌等の）冊・部":"copy",
"近い":"close", "週に一度の":"weekly", "〜を（…よりも）好む":"prefer", "著名する":"sign",
"休憩":"break", "書類、記録する":"document", "四半紀":"quarter", "部門、売り場":"department",
"受け入れる":"accept", "許す":"allow", "接続する、つなぐ":"connect", "最近":"recently",
"もう一つの":"another", "〜しそうだ":"likely", "割合、料金":"rate", "地元の":"local",
"退職する":"retire", "職場":"workplace", "業績、仕事ぶり、性能、公演":"performance", "出身の、その地で生まれた":"native",
"現代的な、近代的な":"modern", "満足した":"satisfied", "著者":"author", "経験豊かな":"experienced",
"支店":"branch", "計画等を作り出す":"develop", "ほとんど、もう少しで":"nearly", "想像する、作り出す":"create", "請求書":"bill",
"郵便物、郵便配達":"mail", "合計、総計":"total", "紹介する、導入する":"introduce", "請求する、料金":"charge",
"利益":"profit", "分野、競技場":"field", "影響する":"affect", "数字、人物":"figure", "作品":"work", "元の、最初の":"original",
"近所、地域":"neighborhood", "環境":"environment", "消費者":"consumer", "展示会、展示物、展示":"exhibition"} # ここに追加する
num = 10 # 問題数
Hint = True # 一文字目のヒント

#ここより下は触らない.
#-------------------------------------------------------------------------------

key = list(dic.keys())
n = len(key)
correct = 0
i = 0
use = set()
miss=[]
print()

while i < n and i < num:
    a = key[randint(0,n-1)]
    if not(a in use):
        if Hint:
            answer = input(a+"("+dic[a][0]+")"+"    ")
        else:
            answer = input(a+"    ")
        use.add(a)
        i += 1
        if answer == dic[a]:
            print("正解")
            print()
            correct+=1
        else:
            print("x 正解は " + dic[a])
            print()
            miss.append(a)
print("正解数は",correct,"問です。")
print()
if i - correct > 0:
    print("間違えた単語は")
    print()
    for x in miss:
        print(x,":",dic[x])
        print()
    print("の",i - correct,"問です。")
else:
    print("全問正解おめでとう！！")
print()
