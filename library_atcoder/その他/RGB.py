def color(*colors):
    """0 <= r,g,b <= 255 を受け取ってカラーコードを返す．"""
    res = "#"
    for c in colors:
        res += ("0" + hex(c)[2:])[-2:]
    return res
