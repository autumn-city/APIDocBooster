from sklearn.metrics import cohen_kappa_score

def get_agreement_score_block(label_1, label_2):
    return cohen_kappa_score(label_1, label_2)

def get_agreement_score_relevance(label_1, label_2):
    rele_1 = []
    rele_2 = []
    for item in label_1:
        if item == 0:
            rele_1.append(0)
        else:
            rele_1.append(1)
    for item in label_2:
        if item == 0:
            rele_2.append(0)
        else:
            rele_2.append(1)           
    return cohen_kappa_score(rele_1, rele_2)

if __name__ == '__main__':
    label_1 = [0,0,3,3,3,3,3,3,0,3,3,1,1,3,3,0,1,0,1,0,3,3,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
    label_2 = [0,0,0,0,0,2,2,0,0,0,1,1,1,3,3,0,1,0,1,0,0,3,3,0,0,0,3,0,0,0,3,3,3,0,0,1,0,0,3,0,0,3,0,0,3,0,0,0,0,0]
    relevance_agreement = get_agreement_score_relevance(label_1, label_2)
    block_agreement = get_agreement_score_block(label_1, label_2)
    print('the relevance agreement is : ',relevance_agreement)
    print('the block agreement is : ',block_agreement)
