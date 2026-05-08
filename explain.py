def explain(pred, click):

    if pred == "REAL":
        msg = "News appears legitimate based on language patterns."
    else:
        msg = "News may contain misleading information."

    if click > 0:
        msg += " Clickbait patterns detected."

    return msg