def human_classify(hallux):
    if hallux < 17 or hallux > 50:
        return 'SS'
    elif hallux > 26:
        return 'RT'
    else:
        return 'CH'