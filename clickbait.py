words = ["shocking","breaking","secret","exposed","you won't believe"]

def score(text):
    s = 0
    for w in words:
        if w in text.lower():
            s += 1
    return s