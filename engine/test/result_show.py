import pickle

filename = '../sofa_test_result.pkl'
with open(filename,'rb') as f:
    result = pickle.load(f)

total_number = len(result)
r20_t20 = 0
r10_t5 = 0
r10_t2 = 0
r5_t5 = 0
r5_t2 = 0
r5,r10,t5,t2 = 0,0,0,0
for item in result:
    rotation, translation = item
    translation = translation
    if rotation<20 and translation < 20:
        r20_t20+=1
    if rotation<10 and translation<5:
        r10_t5+=1
    if rotation<10 and translation<2:
        r10_t2+=1
    if rotation<5 and translation<5:
        r5_t5+=1
    if rotation<5 and translation<2:
        r5_t2+=1
    if rotation<10:
        r10+=1
    if rotation<5:
        r5+=1
    if translation<5:
        t5+=1
    if translation<2:
        t2+=1
    total_number+=1

print(
    f"20'10cm: {100*r20_t20/total_number:.2f}%\n"
    f"10':{100*r10/total_number:.2f}% \n "
    f"5':{100*r5/total_number:.2f}% \n "
    f"5cm:{100*t5/total_number:.2f}% \n"
    f"2cm:{100*t2/total_number:.2f}% \n" 
    f"10'5cm:{100*r10_t5/total_number:.2f}% \n"
    f"10'2cm:{100*r10_t2/total_number:.2f}% \n"
    f"5'5cm:{100*r5_t5/total_number:.2f}% \n"
    f"5'2cm:{100*r5_t2/total_number:.2f}% \n"
)
