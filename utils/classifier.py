def classify_from_score(score):

    if score > 80:
        return "ATENTO"

    elif score > 60:
        return "MULTIPLAS TAREFAS"

    elif score > 40:
        return "DESATENCAO"

    elif score > 20:
        return "RISCO"

    else:
        return "CRITICO"
