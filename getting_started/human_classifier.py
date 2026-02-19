def human_classify(hallux, weight):
    if hallux < 16 or hallux > 50:
        return 'Sharp Shinned'
    elif weight > 620 or hallux > 27:
        return 'Red Tailed'
    else:
        return 'Coopers'